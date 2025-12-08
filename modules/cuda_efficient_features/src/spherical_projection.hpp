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

#pragma once

#include <algorithm>
#include <cmath>

#include <opencv2/core.hpp>

#include "../include/cuda_efficient_descriptors.h"

namespace cv
{
namespace cuda
{

struct SphericalProjection
{
        float thetaMin = 0.f;
        float thetaSpan = static_cast<float>(CV_2PI);
        float phiMin = -static_cast<float>(CV_PI) * 0.5f;
        float phiSpan = static_cast<float>(CV_PI);
        bool wrapHorizontal = true;
};

struct SphericalSamplingParams
{
        int enabled = 0;
        int wrapHorizontal = 0;
        int padding = 0;
        int imageWidth = 0;
        int imageHeight = 0;
        SphericalLensParams lens{};
        SphericalProjection projection{};
};

inline bool hasValidProjectionLens(const SphericalLensParams& lens)
{
        return lens.fx > 0.f && lens.fy > 0.f;
}

inline float radialScale(const SphericalLensParams& lens, float nx, float ny)
{
        const float r2 = nx * nx + ny * ny;
        const float r4 = r2 * r2;
        const float r6 = r4 * r2;
        const float r8 = r4 * r4;
        return 1.f + lens.k1 * r2 + lens.k2 * r4 + lens.k3 * r6 + lens.k4 * r8;
}

inline SphericalProjection buildSphericalProjection(const SphericalLensParams& lens, const Size& imageSize)
{
        SphericalProjection projection;
        if (!hasValidProjectionLens(lens))
                return projection;

        const auto evalTheta = [&](float x) {
                const float nx = (x - lens.cx) / lens.fx;
                return nx * radialScale(lens, nx, 0.f);
        };

        const auto evalPhi = [&](float y) {
                const float ny = (y - lens.cy) / lens.fy;
                return ny * radialScale(lens, 0.f, ny);
        };

        float thetaMin = evalTheta(0.f);
        float thetaMax = evalTheta(static_cast<float>(imageSize.width));
        if (thetaMin > thetaMax)
                std::swap(thetaMin, thetaMax);

        float thetaSpan = thetaMax - thetaMin;
        if (thetaSpan <= 1e-5f)
        {
                thetaMin = 0.f;
                thetaSpan = static_cast<float>(CV_2PI);
        }

        const float fullSpan = static_cast<float>(CV_2PI);
        const bool wraps = thetaSpan >= fullSpan * 0.95f;
        if (wraps)
        {
                thetaMin = 0.f;
                thetaSpan = fullSpan;
        }
        else
        {
                const float margin = 0.01f * thetaSpan;
                thetaMin -= margin;
                thetaSpan += margin * 2.f;
        }

        float phiMin = evalPhi(0.f);
        float phiMax = evalPhi(static_cast<float>(imageSize.height));
        if (phiMin > phiMax)
                std::swap(phiMin, phiMax);

        float phiSpan = phiMax - phiMin;
        if (phiSpan <= 1e-5f)
        {
                phiMin = -static_cast<float>(CV_PI) * 0.5f;
                phiSpan = static_cast<float>(CV_PI);
        }
        else
        {
                const float targetSpan = static_cast<float>(CV_PI);
                if (phiSpan > targetSpan)
                        phiSpan = targetSpan;
        }

        projection.thetaMin = thetaMin;
        projection.thetaSpan = thetaSpan;
        projection.phiMin = phiMin;
        projection.phiSpan = phiSpan;
        projection.wrapHorizontal = wraps;
        return projection;
}

} // namespace cuda
} // namespace cv
