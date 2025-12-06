/*
Copyright 2024 TriOrb Inc.

The major design pattern of this plugin was abstracted
from Fixstars Corporation, which is subject to the same license.
Here is the original copyright notice:

Copyright 2023 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Implementation of the article:
//     Iago Suarez, Ghesn Sfeir, Jose M. Buenaposada, and Luis Baumela.
//     Revisiting binary local image description for resource limited devices.
//     IEEE Robotics and Automation Letters, 2021.

#include "cuda_orb_internal.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace cv
{
namespace cuda
{
namespace gpu
{

static constexpr float CV_DEGREES_TO_RADS = 0.017453292519943295f;
static constexpr float CV_PI_F = 3.14159265358979323846f;
static constexpr float CV_2PI_F = 6.2831853071795864769f;

static __device__ inline float getPixel(const int *integral, int integralStep, int width, int height, int x, int y, bool wrapHorizontal)
{
        if (wrapHorizontal)
        {
                const int mod = width;
                x = ((x % mod) + mod) % mod;
        }
        else
        {
                x = (x < 0) ? 0 : ((x >= width) ? (width - 1) : x);
        }

        y = (y < 0) ? 0 : ((y >= height) ? (height - 1) : y);

        const int idx = (y + 1) * integralStep + (x + 1);
        const int idxL = (y + 1) * integralStep + x;
        const int idxT = y * integralStep + (x + 1);
        const int idxTL = y * integralStep + x;

        return static_cast<float>(integral[idx] - integral[idxL] - integral[idxT] + integral[idxTL]);
}

static __device__ inline float sampleBilinear(const int *integral, int integralStep, int width, int height, float x, float y, bool wrapHorizontal)
{
        if (wrapHorizontal)
        {
                const float mod = static_cast<float>(width);
                x = fmodf(x, mod);
                if (x < 0)
                        x += mod;
        }

        const int x0 = static_cast<int>(floorf(x));
        const int y0 = static_cast<int>(floorf(y));

        const float dx = x - x0;
        const float dy = y - y0;

        const int x1 = wrapHorizontal ? ((x0 + 1) % width) : (x0 + 1);

        const float i00 = getPixel(integral, integralStep, width, height, x0, y0, wrapHorizontal);
        const float i10 = getPixel(integral, integralStep, width, height, x1, y0, wrapHorizontal);
        const float i01 = getPixel(integral, integralStep, width, height, x0, y0 + 1, wrapHorizontal);
        const float i11 = getPixel(integral, integralStep, width, height, x1, y0 + 1, wrapHorizontal);

        const float i0 = i00 + dx * (i10 - i00);
        const float i1 = i01 + dx * (i11 - i01);
        return i0 + dy * (i1 - i0);
}

static __device__ inline void generatePattern(int idx, float halfPatch, float &x1, float &y1, float &x2, float &y2)
{
        // Lightweight deterministic pseudo-random pattern derived from idx
        const float t1 = 0.1234f * idx;
        const float t2 = 0.9876f * idx + 1.2345f;

        x1 = cosf(t1) * halfPatch;
        y1 = sinf(t1) * halfPatch;
        x2 = cosf(t2) * halfPatch * 0.9f;
        y2 = sinf(t2) * halfPatch * 0.9f;
}

__global__ void normalizeSphericalKeypointsKernel(float4 *keypoints, int nkeypoints, int width, int height,
        SphericalLensParams lens)
{
        const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= nkeypoints)
                return;

        float4 kp = keypoints[idx];

        const float nx = (kp.x - lens.cx) / lens.fx;
        const float ny = (kp.y - lens.cy) / lens.fy;

        const float r2 = nx * nx + ny * ny;
        const float radial = 1.f + lens.k1 * r2 + lens.k2 * r2 * r2 + lens.k3 * r2 * r2 * r2 +
                lens.k4 * r2 * r2 * r2 * r2;

        const float theta = nx * radial;
        const float phi = ny * radial;

        float wrappedTheta = fmodf(theta, CV_2PI_F);
        if (wrappedTheta < 0.f)
                wrappedTheta += CV_2PI_F;

        const float clampedPhi = fminf(CV_PI_F * 0.5f, fmaxf(-CV_PI_F * 0.5f, phi));

        kp.x = wrappedTheta * static_cast<float>(width) / CV_2PI_F;
        kp.y = (clampedPhi + (CV_PI_F * 0.5f)) * static_cast<float>(height) / CV_PI_F;

        keypoints[idx] = kp;
}

__global__ void computeORBKernel(const int *integral, int integralStep, int width, int height, const float4 *keypoints,
        int nkeypoints, unsigned char *descriptors, int descriptorStep, float scaleFactor, int patternSize, int patchSize,
        bool wrapHorizontal)
{
        const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= nkeypoints)
                return;

        const float4 kp = keypoints[idx];
        const float angle = (kp.w >= 0) ? kp.w * CV_DEGREES_TO_RADS : 0.0f;
        const float cs = cosf(angle);
        const float sn = sinf(angle);

        const float scale = scaleFactor * kp.z / static_cast<float>(patchSize);
        const float halfPatch = 0.5f * static_cast<float>(patchSize);

        unsigned char *dst = descriptors + idx * descriptorStep;

        for (int i = 0; i < patternSize; i += 8)
        {
                unsigned char byte = 0;
                for (int bit = 0; bit < 8 && (i + bit) < patternSize; ++bit)
                {
                        float px1, py1, px2, py2;
                        generatePattern(i + bit, halfPatch, px1, py1, px2, py2);

                        const float rx1 = kp.x + scale * (cs * px1 - sn * py1);
                        const float ry1 = kp.y + scale * (sn * px1 + cs * py1);
                        const float rx2 = kp.x + scale * (cs * px2 - sn * py2);
                        const float ry2 = kp.y + scale * (sn * px2 + cs * py2);

                        const float v1 = sampleBilinear(integral, integralStep, width, height, rx1, ry1, wrapHorizontal);
                        const float v2 = sampleBilinear(integral, integralStep, width, height, rx2, ry2, wrapHorizontal);

                        byte |= static_cast<unsigned char>((v1 < v2) ? (1u << bit) : 0u);
                }
                dst[i / 8] = byte;
        }
}

void computeORB(const GpuMat &integral, const GpuMat &keypoints, GpuMat &descriptors, float scaleFactor, int paramSize,
        Size patchSize, bool wrapHorizontal, cudaStream_t stream)
{
        const int descriptorSize = (paramSize + 7) / 8;
        CV_Assert(descriptors.type() == CV_8U && descriptors.cols == descriptorSize);
        CV_Assert(keypoints.type() == CV_32FC4);
        CV_Assert(integral.type() == CV_32S);

        const dim3 block(256);
        const dim3 grid((keypoints.rows + block.x - 1) / block.x);

        computeORBKernel<<<grid, block, 0, stream>>>(integral.ptr<int>(), static_cast<int>(integral.step / sizeof(int)),
                integral.cols - 1, integral.rows - 1, keypoints.ptr<float4>(), keypoints.rows, descriptors.ptr<unsigned char>(),
                static_cast<int>(descriptors.step), scaleFactor, paramSize, patchSize.width, wrapHorizontal);
}

void normalizeSphericalKeypoints(GpuMat &keypoints, Size imageSize, const SphericalLensParams &lensParams, cudaStream_t stream)
{
        const dim3 block(256);
        const dim3 grid((keypoints.rows + block.x - 1) / block.x);
        normalizeSphericalKeypointsKernel<<<grid, block, 0, stream>>>(keypoints.ptr<float4>(), keypoints.rows, imageSize.width,
                imageSize.height, lensParams);
}

void calcIntegralImage(const GpuMat &src, GpuMat &dst, Stream &stream)
{
        cv::cuda::integral(src, dst, stream);
}

} // namespace gpu
} // namespace cuda
} // namespace cv
