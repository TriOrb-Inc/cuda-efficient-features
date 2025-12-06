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

#include "cuda_efficient_descriptors.h"

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

#include "cuda_efficient_features.h"

namespace cv
{
namespace cuda
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
        } // namespace

        namespace akaze_internal
        {
                std::vector<KeyPoint> downloadKeypointsScaled(InputArray keypoints, float scaleFactor, Stream& stream);
        }

        class AKAZEImpl : public AKAZE
        {
        public:
                AKAZEImpl(float scaleFactor, int descriptorBits) : scaleFactor_(scaleFactor)
                {
                        CV_Assert(descriptorBits == 256 || descriptorBits == 512);
                        akaze_ = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, descriptorBits);
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
                        applyScale(scaled, scaleFactor_);
                        akaze_->compute(image, scaled, descriptors);
                }

                void computeAsync(InputArray image, InputArray keypoints, OutputArray descriptors, Stream& stream) override
                {
                        if (image.empty())
                                return;

                        std::vector<KeyPoint> hostKeypoints = akaze_internal::downloadKeypointsScaled(keypoints, scaleFactor_, stream);
                        if (hostKeypoints.empty())
                        {
                                descriptors.release();
                                return;
                        }

                        CV_Assert(image.type() == CV_8U);

                        Mat hostImage;
                        if (image.kind() == _InputArray::KindFlag::MAT)
                                hostImage = image.getMat();
                        else if (image.kind() == _InputArray::KindFlag::CUDA_GPU_MAT)
                                image.getGpuMat().download(hostImage, stream);
                        else
                                CV_Error(Error::StsBadArg, "Unsupported image type for AKAZE");

                        Mat hostDescriptors;
                        akaze_->compute(hostImage, hostKeypoints, hostDescriptors);

                        descriptors.create(hostDescriptors.rows, hostDescriptors.cols, hostDescriptors.type());
                        if (descriptors.kind() == _InputArray::KindFlag::MAT)
                        {
                                hostDescriptors.copyTo(descriptors.getMat());
                        }
                        else if (descriptors.kind() == _InputArray::KindFlag::CUDA_GPU_MAT)
                        {
                                descriptors.getGpuMatRef().upload(hostDescriptors, stream);
                        }
                        else
                        {
                                CV_Error(Error::StsBadArg, "Unsupported descriptor output for AKAZE");
                        }
                }

                int descriptorSize() const override { return akaze_->descriptorSize(); }
                int descriptorType() const override { return akaze_->descriptorType(); }
                int defaultNorm() const override { return akaze_->defaultNorm(); }

        private:
                float scaleFactor_;
                Ptr<cv::AKAZE> akaze_;
        };

        Ptr<AKAZE> AKAZE::create(float scaleFactor, int descriptorBits)
        {
                        return makePtr<AKAZEImpl>(scaleFactor, descriptorBits);
        }
} // namespace cuda
} // namespace cv
