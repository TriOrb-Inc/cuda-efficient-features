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

                inline Point2f toEquirectangular(const Point2f& pt, const Size& imageSize, const SphericalLensParams& lens)
                {
                        constexpr float TWO_PI = 6.2831853071795864769f;
                        constexpr float HALF_PI = 1.5707963267948966192f;

                        const float theta = (pt.x - lens.cx) / lens.fx;
                        const float phi = (pt.y - lens.cy) / lens.fy;

                        float wrappedTheta = std::fmod(theta, TWO_PI);
                        if (wrappedTheta < 0.f)
                                wrappedTheta += TWO_PI;

                        const float clampedPhi = std::max(-HALF_PI, std::min(HALF_PI, phi));

                        const float x = wrappedTheta * static_cast<float>(imageSize.width) / TWO_PI;
                        const float y = (clampedPhi + HALF_PI) * static_cast<float>(imageSize.height) / (2.f * HALF_PI);
                        return Point2f(x, y);
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
                        cv::copyMakeBorder(imageMat, padded, 0, 0, HALF_PATCH, HALF_PATCH, BORDER_WRAP);
                        cv::copyMakeBorder(padded, padded, HALF_PATCH, HALF_PATCH, 0, 0, BORDER_REFLECT_101);

                        std::vector<KeyPoint> scaled = keypoints;
                        applyScale(scaled, scale_factor_);

                        SphericalLensParams lens = lens_params_;
                        if (!hasValidLens(lens))
                        {
                                lens.fx = static_cast<float>(imageMat.cols) / 6.2831853071795864769f;
                                lens.fy = static_cast<float>(imageMat.rows) / 3.14159265358979323846f;
                                lens.cx = 0.f;
                                lens.cy = static_cast<float>(imageMat.rows) * 0.5f;
                        }

                        for (auto& kpt : scaled)
                                kpt.pt = toEquirectangular(kpt.pt, imageMat.size(), lens);

                        shiftKeypoints(scaled, Point2f(static_cast<float>(HALF_PATCH), static_cast<float>(HALF_PATCH)));

                        orb_->compute(padded, scaled, descriptors);
                }

                int descriptorSize() const override { return orb_->descriptorSize(); }
                int descriptorType() const override { return orb_->descriptorType(); }
                int defaultNorm() const override { return orb_->defaultNorm(); }

        private:
                float scale_factor_;
                SphericalLensParams lens_params_;
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
