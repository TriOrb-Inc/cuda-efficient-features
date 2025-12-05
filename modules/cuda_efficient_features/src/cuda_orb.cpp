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
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <vector>

#include "cuda_efficient_features_internal.h"
#include "cuda_orb_internal.h"

namespace cv
{
	namespace cuda
	{

                class EORB_Impl : public EORB
                {

                public:
                        static const int LOCATION_ROW = 0;
                        static const int RESPONSE_ROW = 1;
                        static const int ANGLE_ROW = 2;
                        static const int OCTAVE_ROW = 3;
                        static const int SIZE_ROW = 4;

                        explicit EORB_Impl(float scaleFactor) : scaleFactor_(scaleFactor) {}

                        void compute(InputArray _image, KeyPoints &_keypoints, OutputArray _descriptors) override
                        {
                                const std::variant<_InputArray, KeyPoints> keypoints = _keypoints;
                                computeImpl(_image, keypoints, _descriptors, Stream::Null());
                        }

                        void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream &stream) override
                        {
                                const std::variant<_InputArray, KeyPoints> keypoints = _keypoints;
                                computeImpl(_image, keypoints, _descriptors, stream);
                        }

                        int descriptorSize() const override { return descriptorSize_; }
                        int descriptorType() const override { return CV_8U; }
                        int defaultNorm() const override { return NORM_HAMMING; }

                private:
                        void computeImpl(InputArray _image, const std::variant<_InputArray, KeyPoints> &_keypoints,
                                                         OutputArray _descriptors, Stream &stream)
                        {
                                if (_image.empty())
                                        return;

                                if (isEmpty(_keypoints))
                                {
                                        _descriptors.release();
                                        return;
                                }

                                CV_Assert(_image.type() == CV_8U);

                                getInputMat(_image, image_, stream);
                                gpu::calcIntegralImage(image_, integral_, stream);

                                getKeypointsMat(_keypoints, keypoints_, stream);
                                getOutputMat(_descriptors, descriptors_, keypoints_.rows, descriptorSize(), descriptorType());

                                gpu::computeORB(integral_, keypoints_, descriptors_, scaleFactor_, PARAM_SIZE, PATCH_SIZE,
                                                                StreamAccessor::getStream(stream));

                                if (_descriptors.kind() == _InputArray::KindFlag::MAT)
                                        descriptors_.download(_descriptors, stream);
                        }

                        static constexpr int PARAM_SIZE = 256;
                        static constexpr Size PATCH_SIZE = Size(31, 31);
                        float scaleFactor_;
                        int descriptorSize_ = 32;

                        GpuMat image_;
                        GpuMat integral_;
                        GpuMat keypoints_;
                        GpuMat descriptors_;
                };

                Ptr<EORB> EORB::create(float scaleFactor)
                {
                        return makePtr<EORB_Impl>(scaleFactor);
                }

        } // namespace cuda
} // namespace cv
