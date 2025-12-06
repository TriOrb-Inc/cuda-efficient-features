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

#include <algorithm>
#include <cmath>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

#include "cuda_efficient_features.h"

namespace cv
{
namespace cuda
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

                inline SphericalLensParams scaleLensForImage(const SphericalLensParams& lens, const Size& imageSize,
                        Size& baseSize)
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

                inline Point2f toEquirectangular(const Point2f& pt, const Size& imageSize, const SphericalLensParams& lens)
                {
                        constexpr float TWO_PI = 6.2831853071795864769f;
                        constexpr float HALF_PI = 1.5707963267948966192f;

                        const float nx = (pt.x - lens.cx) / lens.fx;
                        const float ny = (pt.y - lens.cy) / lens.fy;

                        const float r2 = nx * nx + ny * ny;
                        const float radial = 1.f + lens.k1 * r2 + lens.k2 * r2 * r2 + lens.k3 * r2 * r2 * r2 +
                                lens.k4 * r2 * r2 * r2 * r2;

                        const float theta = nx * radial;
                        const float phi = ny * radial;

                        float wrappedTheta = std::fmod(theta, TWO_PI);
                        if (wrappedTheta < 0.f)
                                wrappedTheta += TWO_PI;

                        const float clampedPhi = std::max(-HALF_PI, std::min(HALF_PI, phi));

                        const float x = wrappedTheta * static_cast<float>(imageSize.width) / TWO_PI;
                        const float y = (clampedPhi + HALF_PI) * static_cast<float>(imageSize.height) / (2.f * HALF_PI);
                        return Point2f(x, y);
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

                        const std::vector<KeyPoint> hostKeypoints = akaze_internal::downloadKeypointsScaled(keypoints, scaleFactor_, stream);
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

        class SphericalAKAZEImpl : public SphericalAKAZE
        {
        public:
                SphericalAKAZEImpl(float scaleFactor, int descriptorBits, SphericalLensParams lensParams)
                        : scaleFactor_(scaleFactor), lensParams_(lensParams)
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

                        const Mat imageMat = image.getMat();
                        const SphericalLensParams lens = scaleLensForImage(lensParams_, imageMat.size(), lensBaseSize_);

                        Mat padded;
                        copyMakeBorder(imageMat, padded, 0, 0, HALF_PATCH, HALF_PATCH, BORDER_WRAP);
                        copyMakeBorder(padded, padded, HALF_PATCH, HALF_PATCH, 0, 0, BORDER_REFLECT_101);

                        std::vector<KeyPoint> transformed = keypoints;
                        applyScale(transformed, scaleFactor_);
                        for (auto& kpt : transformed)
                                kpt.pt = toEquirectangular(kpt.pt, imageMat.size(), lens);

                        shiftKeypoints(transformed, Point2f(static_cast<float>(HALF_PATCH), static_cast<float>(HALF_PATCH)));

                        akaze_->compute(padded, transformed, descriptors);
                }

                void computeAsync(InputArray image, InputArray keypoints, OutputArray descriptors, Stream& stream) override
                {
                        if (image.empty())
                                return;

                        const std::vector<KeyPoint> hostKeypoints = akaze_internal::downloadKeypointsScaled(keypoints,
                                scaleFactor_, stream);
                        if (hostKeypoints.empty())
                        {
                                descriptors.release();
                                return;
                        }

                        Mat hostImage;
                        if (image.kind() == _InputArray::KindFlag::MAT)
                                hostImage = image.getMat();
                        else if (image.kind() == _InputArray::KindFlag::CUDA_GPU_MAT)
                                image.getGpuMat().download(hostImage, stream);
                        else
                                CV_Error(Error::StsBadArg, "Unsupported image type for AKAZE");

                        CV_Assert(hostImage.type() == CV_8U);

                        const SphericalLensParams lens = scaleLensForImage(lensParams_, hostImage.size(), lensBaseSize_);

                        Mat padded;
                        copyMakeBorder(hostImage, padded, 0, 0, HALF_PATCH, HALF_PATCH, BORDER_WRAP);
                        copyMakeBorder(padded, padded, HALF_PATCH, HALF_PATCH, 0, 0, BORDER_REFLECT_101);

                        std::vector<KeyPoint> transformed = hostKeypoints;
                        for (auto& kpt : transformed)
                                kpt.pt = toEquirectangular(kpt.pt, hostImage.size(), lens);

                        shiftKeypoints(transformed, Point2f(static_cast<float>(HALF_PATCH), static_cast<float>(HALF_PATCH)));

                        Mat hostDescriptors;
                        akaze_->compute(padded, transformed, hostDescriptors);

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
                SphericalLensParams lensParams_;
                Size lensBaseSize_;
                Ptr<cv::AKAZE> akaze_;
        };

                Ptr<AKAZE> AKAZE::create(float scaleFactor, int descriptorBits)
        {
                        return makePtr<AKAZEImpl>(scaleFactor, descriptorBits);
        }

        Ptr<SphericalAKAZE> SphericalAKAZE::create(float scaleFactor, int descriptorBits, SphericalLensParams lensParams)
        {
                        return makePtr<SphericalAKAZEImpl>(scaleFactor, descriptorBits, lensParams);
        }
} // namespace cuda
} // namespace cv
