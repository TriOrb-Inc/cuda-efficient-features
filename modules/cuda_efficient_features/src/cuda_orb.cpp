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
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>

#include <vector>

#include "cuda_efficient_features_internal.h"

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

                        EORB_Impl(float scaleFactor) : scaleFactor_(scaleFactor)
                        {
                                orb_gpu_ = cv::cuda::ORB::create(1);
                                descriptorSize_ = orb_gpu_->getDescriptorSize();
                        }

                        void compute(InputArray _image, KeyPoints &_keypoints, OutputArray _descriptors) override
                        {
                                if (_image.empty())
                                        return;

                                if (_keypoints.empty())
                                {
                                        _descriptors.release();
                                        return;
                                }

                                CV_Assert(_image.type() == CV_8U);

                                GpuMat image;
                                getInputMat(_image, image);

                                std::vector<KeyPoint> keypoints = _keypoints;
                                applyScale(keypoints);

                                if (_descriptors.kind() == _InputArray::KindFlag::MAT)
                                {
                                        GpuMat descriptorsGpu;
                                        descriptorsGpu.create(static_cast<int>(keypoints.size()), descriptorSize(), descriptorType());
                                        orb_gpu_->computeAsync(image, keypoints, descriptorsGpu, Stream::Null());
                                        descriptorsGpu.download(_descriptors);
                                }
                                else
                                {
                                        GpuMat descriptors = _descriptors.getGpuMat();
                                        orb_gpu_->computeAsync(image, keypoints, descriptors, Stream::Null());
                                }
                        }

                        void convert(InputArray src, CV_OUT std::vector<KeyPoint> &dst)
                        {
                                Mat tmp;
                                if (src.kind() == _InputArray::KindFlag::MAT)
                                        tmp = src.getMat();
                                else if (src.kind() == _InputArray::KindFlag::CUDA_GPU_MAT)
                                        src.getGpuMat().download(tmp);

                                if (tmp.empty())
                                {
                                        dst.clear();
                                        return;
                                }

                                CV_Assert(tmp.rows >= 5);

                                const Vec2s *points = tmp.ptr<Vec2s>(LOCATION_ROW);
                                const float *responses = tmp.ptr<float>(RESPONSE_ROW);
                                const float *angles = tmp.ptr<float>(ANGLE_ROW);
                                const int *octaves = tmp.ptr<int>(OCTAVE_ROW);
                                const float *sizes = tmp.ptr<float>(SIZE_ROW);

                                const int nkeypoints = tmp.cols;
                                dst.resize(nkeypoints);
                                for (int i = 0; i < nkeypoints; i++)
                                {
                                        KeyPoint kpt;
                                        kpt.pt = Point2f(points[i][0], points[i][1]);
                                        kpt.response = responses[i];
                                        kpt.angle = angles[i];
                                        kpt.octave = octaves[i];
                                        kpt.size = sizes[i];
                                        dst[i] = kpt;
                                }
                        }

                        void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream &stream) override
                        {
                                if (_image.empty())
                                        return;

                                if (_keypoints.empty())
                                {
                                        _descriptors.release();
                                        return;
                                }

                                CV_Assert(_image.type() == CV_8U);

                                GpuMat image;
                                getInputMat(_image, image, stream);

                                std::vector<KeyPoint> keypoints;
                                convert(_keypoints, keypoints);
                                if (keypoints.empty())
                                {
                                        _descriptors.release();
                                        return;
                                }

                                applyScale(keypoints);

                                if (_descriptors.kind() == _InputArray::KindFlag::MAT)
                                {
                                        GpuMat descriptorsGpu;
                                        descriptorsGpu.create(static_cast<int>(keypoints.size()), descriptorSize(), descriptorType());
                                        orb_gpu_->computeAsync(image, keypoints, descriptorsGpu, stream);
                                        descriptorsGpu.download(_descriptors, stream);
                                }
                                else
                                {
                                        GpuMat descriptors = _descriptors.getGpuMat();
                                        orb_gpu_->computeAsync(image, keypoints, descriptors, stream);
                                }
                        }

                        int descriptorSize() const override { return descriptorSize_; }
                        int descriptorType() const override { return CV_8U; }
                        int defaultNorm() const override { return NORM_HAMMING; }

                private:
                        void applyScale(std::vector<KeyPoint> &keypoints) const
                        {
                                if (scaleFactor_ == 1.f)
                                        return;

                                for (auto &kpt : keypoints)
                                        kpt.size *= scaleFactor_;
                        }

                        float scaleFactor_;
                        int descriptorSize_;
                        cv::Ptr<cv::cuda::ORB> orb_gpu_;
                };

		Ptr<EORB> EORB::create(float scaleFactor)
		{
			return makePtr<EORB_Impl>(scaleFactor);
		}

	} // namespace cuda
} // namespace cv
