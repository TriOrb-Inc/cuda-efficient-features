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

#include "efficient_descriptors.h"

#include <opencv2/features2d.hpp>
#include <vector>

namespace cv
{
        namespace
        {
                void applyScale(std::vector<KeyPoint>& keypoints, float scaleFactor)
                {
                        if (scaleFactor == 1.f)
                                return;

                        for (auto& kpt : keypoints)
                                kpt.size *= scaleFactor;
                }
        }

        class EAKAZE_Impl : public EAKAZE
        {
        public:
                EAKAZE_Impl(float scale_factor, int descriptor_bits) : scale_factor_(scale_factor)
                {
                        CV_Assert(descriptor_bits == 256 || descriptor_bits == 512);
                        akaze_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, descriptor_bits);
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
                        akaze_->compute(image, scaled, descriptors);
                }

                int descriptorSize() const override { return akaze_->descriptorSize(); }
                int descriptorType() const override { return akaze_->descriptorType(); }
                int defaultNorm() const override { return akaze_->defaultNorm(); }

        private:
                float scale_factor_;
                Ptr<cv::AKAZE> akaze_;
        };

        Ptr<EAKAZE> EAKAZE::create(float scale_factor, int descriptor_bits)
        {
                return makePtr<EAKAZE_Impl>(scale_factor, descriptor_bits);
        }
} // namespace cv
