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

#include "cuda_macro.h"

namespace cv
{
namespace cuda
{
namespace gpu
{

static constexpr float CV_DEGREES_TO_RADS = 0.017453292519943295f;
static constexpr float CV_PI_F = 3.14159265358979323846f;
static constexpr float CV_2PI_F = 6.2831853071795864769f;

static __device__ __constant__ int ORB_PATTERN[256 * 4] = {
         8, -3,  9,  5,  4,  2,  7, -12, -11,  9, -8,  2,  7, -12, 12, -13,
         2, -13,  2, 12,  1, -7,  1,  6, -2, -10, -2, -4, -13, -13, -11, -8,
        -13, -3, -12, -9, 10,  4, 11,  9, -13, -8, -8, -9, -11,  7, -9, 12,
         7,  7, 12,  6, -4, -5, -3,  0, -13,  2, -12, -3, -9,  0, -7,  5,
        12, -6, 12, -1, -3,  6, -2, 12, -6, -13, -4, -8, 11, -13, 12, -8,
         4,  7,  5,  1,  5, -3, 10, -3,  3, -7,  6, 12, -8, -7, -6, -2,
        -2, 11, -1, -10, -13, 12, -8, 10, -7,  3, -5, -3, -4,  2, -3,  7,
        -10, -12, -6, 11,  5, -12,  6, -7,  5, -6,  7, -1,  1,  0,  4, -5,
         9, 11, 11, -13,  4,  7,  4, 12,  2, -1,  4,  4, -4, -12, -2,  7,
        -8, -5, -7, -10,  4, 11,  9, 12,  0, -8,  1, -13, -13, -2, -8,  2,
        -3, -2, -2,  3, -6,  9, -4, -9,  8, 12, 10,  7,  0,  9,  1,  3,
         7, -5, 11, -10, -13, -6, -11,  0, 10,  7, 12,  1, -6, -3, -6, 12,
        10, -9, 12, -4, -13,  8, -8, -12, -13,  0, -8, -4,  3,  3,  7,  8,
         5,  7, 10, -7, -1,  7,  1, -12,  3, -10,  5,  6,  2, -4,  3, -10,
        -13,  0, -13,  5, -13, -7, -12, 12, -13,  3, -11,  8, -7, 12, -4,  7,
         6, -10, 12,  8, -9, -1, -7, -6, -2, -5,  0, 12, -12,  5, -7,  5,
         3, -10,  8, -13, -7, -7, -4,  5, -3, -2, -1, -7,  2,  9,  5, -11,
        -11, -13, -5, -13, -1,  6,  0, -1,  5, -3,  5,  2, -4, -13, -4, 12,
        -9, -6, -9,  6, -12, -10, -8, -4, 10,  2, 12, -3,  7, 12, 12, 12,
        -7, -13, -6,  5, -4,  9, -3,  4,  7, -1, 12,  2, -7,  6, -5,  1,
        -13, 11, -12,  5, -3,  7, -2, -6,  7, -8, 12, -7, -13, -7, -11, -12,
         1, -3, 12, 12,  2, -6,  3,  0, -4,  3, -2, -13, -1, -13,  1,  9,
         7,  1,  8, -6,  1, -1,  3, 12,  9,  1, 12,  6, -1, -9, -1,  3,
        -13, -13, -10,  5,  7,  7, 10, 12, 12, -5, 12,  9,  6,  3,  7, 11,
         5, -13,  6, 10,  2, -12,  2,  3,  3,  8,  4, -6,  2,  6, 12, -13,
         9, -12, 10,  3, -8,  4, -7,  9, -11, 12, -4, -6,  1, 12,  2, -8,
         6, -9,  7, -4,  2,  3,  3, -2,  6,  3, 11,  0,  3, -3,  8, -8,
         7,  8,  9,  3, -11, -5, -6, -4, -10, 11, -5, 10, -5, -8, -3, 12,
        -10,  5, -9,  0,  8, -1, 12, -6,  4, -6,  6, -11, -10, 12, -8,  7,
         4, -2,  6,  7, -2,  0, -2, 12, -5, -8, -5,  2,  7, -6, 10, 12,
        -9, -13, -8, -8, -5, -13, -5, -2,  8, -8,  9, -13, -9, -11, -9,  0,
         1, -8,  1, -2,  7, -4,  9,  1, -2,  1, -1, -4, 11, -6, 12, -11,
        -12, -9, -6,  4,  3,  7,  7, 12,  5,  5, 10,  8,  0, -4,  2,  8,
        -9, 12, -5, -13,  0,  7,  2, 12, -1,  2,  1,  7,  5, 11,  7, -9,
         3,  5,  6, -8, -13, -4, -8,  9, -5,  9, -3, -3, -4, -7, -3, -12,
         6,  5,  8,  0, -7,  6, -6, 12, -13,  6, -5, -2,  1, -10,  3, 10,
         4,  1,  8, -4, -2, -2,  2, -13,  2, -12, 12, 12, -2, -13,  0, -6,
         4,  1,  9,  3, -6, -10, -3, -5, -3, -13, -1,  1,  7,  5, 12, -11,
         4, -2,  5, -7, -13,  9, -9, -5,  7,  1,  8,  6,  7, -8,  7,  6,
        -7, -4, -7,  1, -8, 11, -7, -8, -13,  6, -12, -8,  2,  4,  3,  9,
        10, -5, 12,  3, -6, -5, -6,  7,  8, -3,  9, -8,  2, -12,  2,  8,
        -11, -2, -10,  3, -12, -13, -7, -9, -11,  0, -10, -5,  5, -3, 11,  8,
        -2, -13, -1, 12, -1, -8,  0,  9, -13, -11, -12, -5, -10, -2, -10, 11,
        -3,  9, -2, -13,  2, -3,  3,  2, -9, -13, -4,  0, -4,  6, -3, -10,
        -4, 12, -2, -7, -6, -11, -4,  9,  6, -3,  6, 11, -13, 11, -5,  5,
        11, 11, 12,  6,  7, -5, 12, -2, -1, 12,  0,  7, -4, -8, -3, -2,
        -7,  1, -6,  7, -13, -12, -8, -13, -7, -2, -6, -8, -8,  5, -6, -9,
        -5, -1, -4,  5, -13,  7, -8, 10,  1,  5,  5, -13,  1,  0, 10, -13,
         9, 12, 10, -1,  5, -8, 10, -9, -1, 11,  1, -13, -9, -3, -6,  2,
        -1, -10,  1, 12, -13,  1, -8, -10,  8, -11, 10, -6,  2, -13,  3, -6,
         7, -13, 12, -9, -10, -10, -5, -7, -10, -8, -8, -13,  4, -6,  8,  5,
         3, 12,  8, -13, -4,  2, -3, -3,  5, -13, 10, -12,  4, -13,  5, -1,
        -9,  9, -4,  3,  0,  3,  3, -9, -12,  1, -6,  1,  3,  2,  4, -8,
        -10, -10, -10,  9,  8, -13, 12, 12, -8, -12, -6, -5,  2,  2,  3,  7,
        10,  6, 11, -8,  6,  8,  8, -12, -7, 10, -6,  5, -3, -9, -3,  9,
        -1, -13, -1,  5, -3, -7, -3,  4, -8, -2, -8,  3,  4,  2, 12, 12,
         2, -5,  3, 11,  6, -9, 11, -13,  3, -1,  7, 12, 11, -1, 12,  4,
        -3,  0, -3,  6,  4, -11,  4, 12,  2, -4,  2,  1, -10, -6, -8,  1,
        -13,  7, -11,  1, -13, 12, -11, -13,  6,  0, 11, -13,  0, -1,  1,  4,
        -13,  3, -9, -2, -9,  8, -6, -3, -13, -6, -8, -2,  5, -9,  8, 10,
         2,  7,  3, -9, -1, -6, -1, -1,  9,  5, 11, -2, 11, -3, 12, -8,
         3,  0,  3,  5, -1,  4,  0, 10,  3, -6,  4,  5, -13,  0, -10,  5,
         5,  8, 12, 11,  8,  9,  9, -6,  7, -4,  8, -12, -10,  4, -10,  9,
         7,  3, 12,  4,  9, -7, 10, -2,  7,  0, 12, -2, -1, -6,  0, -11,
};

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

static __device__ inline void loadPattern(int idx, float &x1, float &y1, float &x2, float &y2)
{
        const int baseIdx = idx * 4;
        x1 = static_cast<float>(ORB_PATTERN[baseIdx]);
        y1 = static_cast<float>(ORB_PATTERN[baseIdx + 1]);
        x2 = static_cast<float>(ORB_PATTERN[baseIdx + 2]);
        y2 = static_cast<float>(ORB_PATTERN[baseIdx + 3]);
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

        unsigned char *dst = descriptors + idx * descriptorStep;

        for (int i = 0; i < patternSize; i += 8)
        {
                unsigned char byte = 0;
                for (int bit = 0; bit < 8 && (i + bit) < patternSize; ++bit)
                {
                        float px1, py1, px2, py2;
                        loadPattern(i + bit, px1, py1, px2, py2);

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
        CV_Assert(paramSize <= 256);

        const int descriptorSize = (paramSize + 7) / 8;
        CV_Assert(descriptors.type() == CV_8U && descriptors.cols == descriptorSize);
        CV_Assert(keypoints.type() == CV_32FC4);
        CV_Assert(integral.type() == CV_32S);

        const dim3 block(256);
        const dim3 grid((keypoints.rows + block.x - 1) / block.x);

        computeORBKernel<<<grid, block, 0, stream>>>(integral.ptr<int>(), static_cast<int>(integral.step / sizeof(int)),
                integral.cols - 1, integral.rows - 1, keypoints.ptr<float4>(), keypoints.rows, descriptors.ptr<unsigned char>(),
                static_cast<int>(descriptors.step), scaleFactor, paramSize, patchSize.width, wrapHorizontal);
        CUDA_CHECK(cudaGetLastError());

        // 同期 API 呼び出しではデフォルトストリームの完了を待つが、非同期ストリームでは呼び出し元に委ねる
        if (stream == nullptr)
                CUDA_CHECK(cudaStreamSynchronize(nullptr));
}

void normalizeSphericalKeypoints(GpuMat &keypoints, Size imageSize, const SphericalLensParams &lensParams, cudaStream_t stream)
{
        const dim3 block(256);
        const dim3 grid((keypoints.rows + block.x - 1) / block.x);
        normalizeSphericalKeypointsKernel<<<grid, block, 0, stream>>>(keypoints.ptr<float4>(), keypoints.rows, imageSize.width,
                imageSize.height, lensParams);
}

} // namespace gpu
} // namespace cuda
} // namespace cv
