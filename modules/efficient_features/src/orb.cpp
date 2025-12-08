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

#include "efficient_descriptors.h"

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include "spherical_projection.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cv
{
        namespace
        {
                static constexpr int HALF_PATCH = 15;

                void applyScale(std::vector<KeyPoint>& keypoints, float scaleFactor)
                {
                        if (scaleFactor == 1.f)
                                return;

                        for (auto& kpt : keypoints)
                                kpt.size *= scaleFactor;
                }

                inline void shiftKeypoints(std::vector<KeyPoint>& keypoints, cv::Point2f offset)
                {
                        for (auto& kpt : keypoints)
                                kpt.pt += offset;
                }

        inline bool hasValidLens(const SphericalLensParams& lens)
        {
                return lens.fx > 0.f && lens.fy > 0.f;
        }

        inline SphericalLensParams resolveLens(const SphericalLensParams& lens, const Size& imageSize)
        {
                if (hasValidLens(lens))
                        return lens;

                SphericalLensParams fallback;
                fallback.fx = static_cast<float>(imageSize.width) / 6.2831853071795864769f;
                fallback.fy = static_cast<float>(imageSize.height) / 3.14159265358979323846f;
                fallback.cx = 0.f;
                fallback.cy = static_cast<float>(imageSize.height) * 0.5f;
                fallback.k1 = 0.f;
                fallback.k2 = 0.f;
                fallback.k3 = 0.f;
                fallback.k4 = 0.f;
                return fallback;
        }

        inline SphericalLensParams scaleLensForImage(const SphericalLensParams& lens, const Size& imageSize, Size& baseSize)
        {
                SphericalLensParams resolved = resolveLens(lens, imageSize);

                if (!hasValidLens(lens))
                        return resolved;

                if (baseSize.area() == 0)
                        baseSize = imageSize;

                if (baseSize == imageSize)
                        return resolved;

                const float sx = static_cast<float>(imageSize.width) / static_cast<float>(baseSize.width);
                const float sy = static_cast<float>(imageSize.height) / static_cast<float>(baseSize.height);

                resolved.fx *= sx;
                resolved.fy *= sy;
                resolved.cx *= sx;
                resolved.cy *= sy;

                return resolved;
        }

                inline Point2f toEquirectangular(const Point2f& pt, const Size& imageSize, const SphericalLensParams& lens,
                        const SphericalProjection& projection)
                {
                        const float nx = (pt.x - lens.cx) / lens.fx;
                        const float ny = (pt.y - lens.cy) / lens.fy;

                        const float r2 = nx * nx + ny * ny;
                        const float radial = 1.f + lens.k1 * r2 + lens.k2 * r2 * r2 + lens.k3 * r2 * r2 * r2 +
                                lens.k4 * r2 * r2 * r2 * r2;

                        const float theta = nx * radial;
                        const float phi = ny * radial;

                        float normalizedTheta = theta - projection.thetaMin;
                        if (projection.wrapHorizontal)
                        {
                                normalizedTheta = std::fmod(normalizedTheta, projection.thetaSpan);
                                if (normalizedTheta < 0.f)
                                        normalizedTheta += projection.thetaSpan;
                        }
                        else
                        {
                                normalizedTheta = std::min(projection.thetaSpan, std::max(0.f, normalizedTheta));
                        }

                        float normalizedPhi = phi - projection.phiMin;
                        normalizedPhi = std::min(projection.phiSpan, std::max(0.f, normalizedPhi));

                        const float thetaScale = static_cast<float>(imageSize.width) / projection.thetaSpan;
                        const float phiScale = static_cast<float>(imageSize.height) / projection.phiSpan;
                        return Point2f(normalizedTheta * thetaScale, normalizedPhi * phiScale);
                }
        } // namespace

        class EORB_Impl : public EORB
        {
        public:
                explicit EORB_Impl(float scale_factor) : scale_factor_(scale_factor)
                {
                        orb_ = cv::ORB::create(1);
                }

                void compute(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors) override
                {
                        if (image.empty())
                                return;

                        if (keypoints.empty())
                        {
                                descriptors.release();
                                return;
                        }

                        CV_Assert(image.type() == CV_8U);

                        std::vector<KeyPoint> scaled = keypoints;
                        applyScale(scaled, scale_factor_);
                        orb_->compute(image, scaled, descriptors);
                }

                int descriptorSize() const override { return orb_->descriptorSize(); }
                int descriptorType() const override { return orb_->descriptorType(); }
                int defaultNorm() const override { return orb_->defaultNorm(); }

        private:
                float scale_factor_;
                Ptr<cv::ORB> orb_;
        };

        class SphericalORB_Impl : public SphericalORB
        {
        public:
                explicit SphericalORB_Impl(float scale_factor, SphericalLensParams lens_params)
                        : scale_factor_(scale_factor), lens_params_(lens_params)
                {
                        orb_ = cv::ORB::create(1);
                }

                void compute(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors) override
                {
                        if (image.empty())
                                return;

                        if (keypoints.empty())
                        {
                                descriptors.release();
                                return;
                        }

                        CV_Assert(image.type() == CV_8U);

                        Mat padded;
                        const Mat imageMat = image.getMat();
                        const SphericalLensParams lens = scaleLensForImage(lens_params_, imageMat.size(), lens_base_size_);
                        const SphericalProjection projection = buildSphericalProjection(lens, imageMat.size());
                        if (projection.wrapHorizontal)
                        {
                                cv::copyMakeBorder(imageMat, padded, 0, 0, HALF_PATCH, HALF_PATCH, BORDER_WRAP);
                                cv::copyMakeBorder(padded, padded, HALF_PATCH, HALF_PATCH, 0, 0, BORDER_REFLECT_101);
                        }
                        else
                        {
                                cv::copyMakeBorder(imageMat, padded, HALF_PATCH, HALF_PATCH, HALF_PATCH, HALF_PATCH,
                                        BORDER_REFLECT_101);
                        }

                        std::vector<KeyPoint> scaled = keypoints;
                        applyScale(scaled, scale_factor_);

                        for (auto& kpt : scaled)
                                kpt.pt = toEquirectangular(kpt.pt, imageMat.size(), lens, projection);

                        shiftKeypoints(scaled, Point2f(static_cast<float>(HALF_PATCH), static_cast<float>(HALF_PATCH)));

                        orb_->compute(padded, scaled, descriptors);
                }

                int descriptorSize() const override { return orb_->descriptorSize(); }
                int descriptorType() const override { return orb_->descriptorType(); }
                int defaultNorm() const override { return orb_->defaultNorm(); }

        private:
                float scale_factor_;
                SphericalLensParams lens_params_;
                Size lens_base_size_;
                Ptr<cv::ORB> orb_;
        };

        Ptr<EORB> EORB::create(float scale_factor)
        {
                return makePtr<EORB_Impl>(scale_factor);
        }

        Ptr<SphericalORB> SphericalORB::create(float scale_factor, SphericalLensParams lens_params)
        {
                return makePtr<SphericalORB_Impl>(scale_factor, lens_params);
        }
} // namespace cv
