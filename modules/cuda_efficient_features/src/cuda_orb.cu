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

#include <opencv2/cudaimgproc.hpp>

namespace cv
{
namespace cuda
{
namespace gpu
{

static constexpr float CV_DEGREES_TO_RADS = 0.017453292519943295f;

static __device__ inline float getPixel(const int *integral, int integralStep, int width, int height, int x, int y)
{
        x = max(0, min(x, width - 1));
        y = max(0, min(y, height - 1));

        const int idx = (y + 1) * integralStep + (x + 1);
        const int idxL = (y + 1) * integralStep + x;
        const int idxT = y * integralStep + (x + 1);
        const int idxTL = y * integralStep + x;

        return static_cast<float>(integral[idx] - integral[idxL] - integral[idxT] + integral[idxTL]);
}

static __device__ inline float sampleBilinear(const int *integral, int integralStep, int width, int height, float x, float y)
{
        const int x0 = static_cast<int>(floorf(x));
        const int y0 = static_cast<int>(floorf(y));

        const float dx = x - x0;
        const float dy = y - y0;

        const float i00 = getPixel(integral, integralStep, width, height, x0, y0);
        const float i10 = getPixel(integral, integralStep, width, height, x0 + 1, y0);
        const float i01 = getPixel(integral, integralStep, width, height, x0, y0 + 1);
        const float i11 = getPixel(integral, integralStep, width, height, x0 + 1, y0 + 1);

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

__global__ void computeORBKernel(const int *integral, int integralStep, int width, int height, const float4 *keypoints,
        int nkeypoints, unsigned char *descriptors, int descriptorStep, float scaleFactor, int patternSize, int patchSize)
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

                        const float v1 = sampleBilinear(integral, integralStep, width, height, rx1, ry1);
                        const float v2 = sampleBilinear(integral, integralStep, width, height, rx2, ry2);

                        byte |= static_cast<unsigned char>((v1 < v2) ? (1u << bit) : 0u);
                }
                dst[i / 8] = byte;
        }
}

void computeORB(const GpuMat &integral, const GpuMat &keypoints, GpuMat &descriptors, float scaleFactor, int paramSize,
        Size patchSize, cudaStream_t stream)
{
        const int descriptorSize = (paramSize + 7) / 8;
        CV_Assert(descriptors.type() == CV_8U && descriptors.cols == descriptorSize);
        CV_Assert(keypoints.type() == CV_32FC4);
        CV_Assert(integral.type() == CV_32S);

        const dim3 block(256);
        const dim3 grid((keypoints.rows + block.x - 1) / block.x);

        computeORBKernel<<<grid, block, 0, stream>>>(integral.ptr<int>(), static_cast<int>(integral.step / sizeof(int)),
                integral.cols - 1, integral.rows - 1, keypoints.ptr<float4>(), keypoints.rows, descriptors.ptr<unsigned char>(),
                static_cast<int>(descriptors.step), scaleFactor, paramSize, patchSize.width);
}

void calcIntegralImage(const GpuMat &src, GpuMat &dst, Stream &stream)
{
        cv::cuda::integral(src, dst, stream);
}

} // namespace gpu
} // namespace cuda
} // namespace cv
