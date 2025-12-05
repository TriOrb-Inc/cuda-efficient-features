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

#include "cuda_efficient_descriptors.h"
#include "cuda_efficient_features.h"

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <vector>

#include "cuda_efficient_features_internal.h"
#include "cuda_orb_internal.h"

namespace cv
{
	namespace cuda
	{

                namespace
                {
                        static constexpr int PARAM_SIZE = 256;
                        static constexpr Size PATCH_SIZE = Size(31, 31);
                        static constexpr int HALF_PATCH = PATCH_SIZE.width / 2;

                        struct ORBBuffers
                        {
                                GpuMat image;
                                GpuMat integral;
                                GpuMat keypoints;
                                GpuMat descriptors;
                        };

                        template <bool WrapHorizontal>
                        void computeDescriptors(InputArray _image, const std::variant<_InputArray, KeyPoints> &_keypoints,
                                OutputArray _descriptors, Stream &stream, float scaleFactor, ORBBuffers &buffers)
                        {
                                if (_image.empty())
                                        return;

                                if (isEmpty(_keypoints))
                                {
                                        _descriptors.release();
                                        return;
                                }

                                CV_Assert(_image.type() == CV_8U);

                                getInputMat(_image, buffers.image, stream);

                                if constexpr (WrapHorizontal)
                                {
                                        GpuMat horizontalWrapped;
                                        cv::cuda::copyMakeBorder(buffers.image, horizontalWrapped, 0, 0, HALF_PATCH, HALF_PATCH,
                                                BORDER_WRAP, Scalar(), stream);

                                        cv::cuda::copyMakeBorder(horizontalWrapped, buffers.image, HALF_PATCH, HALF_PATCH, 0, 0,
                                                BORDER_REFLECT_101, Scalar(), stream);
                                }

                                gpu::calcIntegralImage(buffers.image, buffers.integral, stream);

                                getKeypointsMat(_keypoints, buffers.keypoints, stream);

                                if constexpr (WrapHorizontal)
                                {
                                        cv::cuda::add(buffers.keypoints, Scalar(HALF_PATCH, HALF_PATCH, 0, 0), buffers.keypoints,
                                                noArray(), -1, stream);
                                }

                                getOutputMat(_descriptors, buffers.descriptors, buffers.keypoints.rows, PARAM_SIZE / 8, CV_8U);

                                gpu::computeORB(buffers.integral, buffers.keypoints, buffers.descriptors, scaleFactor, PARAM_SIZE,
                                        PATCH_SIZE, WrapHorizontal, StreamAccessor::getStream(stream));

                                if (_descriptors.kind() == _InputArray::KindFlag::MAT)
                                        buffers.descriptors.download(_descriptors, stream);
                        }
                } // namespace

                class EORB_Impl : public EORB
                {
                public:
                        explicit EORB_Impl(float scaleFactor) : scaleFactor_(scaleFactor) {}

                        void compute(InputArray _image, KeyPoints &_keypoints, OutputArray _descriptors) override
                        {
                                const std::variant<_InputArray, KeyPoints> keypoints = _keypoints;
                                computeDescriptors<false>(_image, keypoints, _descriptors, Stream::Null(), scaleFactor_, buffers_);
                        }

                        void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream &stream) override
                        {
                                const std::variant<_InputArray, KeyPoints> keypoints = _keypoints;
                                computeDescriptors<false>(_image, keypoints, _descriptors, stream, scaleFactor_, buffers_);
                        }

                        int descriptorSize() const override { return PARAM_SIZE / 8; }
                        int descriptorType() const override { return CV_8U; }
                        int defaultNorm() const override { return NORM_HAMMING; }

                private:
                        float scaleFactor_;
                        ORBBuffers buffers_;
                };

                class SphericalORB_Impl : public SphericalORB
                {
                public:
                        explicit SphericalORB_Impl(float scaleFactor) : scaleFactor_(scaleFactor) {}

                        void compute(InputArray _image, KeyPoints &_keypoints, OutputArray _descriptors) override
                        {
                                const std::variant<_InputArray, KeyPoints> keypoints = _keypoints;
                                computeDescriptors<true>(_image, keypoints, _descriptors, Stream::Null(), scaleFactor_, buffers_);
                        }

                        void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream &stream) override
                        {
                                const std::variant<_InputArray, KeyPoints> keypoints = _keypoints;
                                computeDescriptors<true>(_image, keypoints, _descriptors, stream, scaleFactor_, buffers_);
                        }

                        int descriptorSize() const override { return PARAM_SIZE / 8; }
                        int descriptorType() const override { return CV_8U; }
                        int defaultNorm() const override { return NORM_HAMMING; }

                private:
                        float scaleFactor_;
                        ORBBuffers buffers_;
                };

                Ptr<EORB> EORB::create(float scaleFactor)
                {
                        return makePtr<EORB_Impl>(scaleFactor);
                }

                Ptr<SphericalORB> SphericalORB::create(float scaleFactor)
                {
                        return makePtr<SphericalORB_Impl>(scaleFactor);
                }

        } // namespace cuda
} // namespace cv
