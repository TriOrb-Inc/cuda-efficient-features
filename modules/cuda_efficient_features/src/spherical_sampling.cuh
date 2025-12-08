/*
Device helpers for spherical sampling across descriptors.
*/

#pragma once

#include <cuda_runtime.h>

#include "spherical_projection.hpp"

namespace cv
{
namespace cuda
{
namespace gpu
{

__device__ __forceinline__ float2 mapToLens(float x, float y, const SphericalSamplingParams &params)
{
        const float paddedWidth = static_cast<float>(params.imageWidth + 2 * params.padding);
        const float paddedHeight = static_cast<float>(params.imageHeight + 2 * params.padding);

        float equiX = x - static_cast<float>(params.padding);
        float equiY = y - static_cast<float>(params.padding);

        if (params.wrapHorizontal)
        {
                const float mod = static_cast<float>(params.imageWidth);
                equiX = fmodf(equiX, mod);
                if (equiX < 0.f)
                        equiX += mod;
        }
        else
        {
                equiX = fminf(fmaxf(equiX, 0.f), static_cast<float>(params.imageWidth - 1));
        }

        equiY = fminf(fmaxf(equiY, 0.f), static_cast<float>(params.imageHeight - 1));

        const float theta = (equiX / static_cast<float>(params.imageWidth)) * params.projection.thetaSpan +
                params.projection.thetaMin;
        const float phi = (equiY / static_cast<float>(params.imageHeight)) * params.projection.phiSpan +
                params.projection.phiMin;

        const float thetaPhi2 = theta * theta + phi * phi;

        float radial = 1.f;
        if (thetaPhi2 > 1e-8f)
        {
                for (int i = 0; i < 5; ++i)
                {
                        const float invRadial = 1.f / (radial + 1e-6f);
                        const float r2 = thetaPhi2 * invRadial * invRadial;
                        radial = 1.f + params.lens.k1 * r2 + params.lens.k2 * r2 * r2 + params.lens.k3 * r2 * r2 * r2 +
                                params.lens.k4 * r2 * r2 * r2 * r2;
                }
        }

        const float invRadial = 1.f / (radial + 1e-6f);
        const float nx = theta * invRadial;
        const float ny = phi * invRadial;

        float px = nx * params.lens.fx + params.lens.cx + static_cast<float>(params.padding);
        float py = ny * params.lens.fy + params.lens.cy + static_cast<float>(params.padding);

        px = fminf(fmaxf(px, 0.f), paddedWidth - 1.f);
        py = fminf(fmaxf(py, 0.f), paddedHeight - 1.f);

        return make_float2(px, py);
}

} // namespace gpu
} // namespace cuda
} // namespace cv
