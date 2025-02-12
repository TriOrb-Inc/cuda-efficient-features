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

#include "cuda_orb_internal.h"
#include "cuda_efficient_features_internal.h"
#include "device_buffer.h"

#include <iostream>

namespace cv
{
	namespace cuda
	{

		void convertKeypoints(const GpuMat &src, GpuMat &dst, cudaStream_t stream);

		class EORB_Impl : public EORB
		{

		public:
			static const int LOCATION_ROW = 0;
			static const int RESPONSE_ROW = 1;
			static const int ANGLE_ROW = 2;
			static const int OCTAVE_ROW = 3;
			static const int SIZE_ROW = 4;
			static const int ROWS_COUNT = 5;

			EORB_Impl(float scaleFactor) : scaleFactor_(scaleFactor), paramSize_(256), patchSize_(32, 32)
			{
				orb_cpu_ = cv::ORB::create(1);
				orb_gpu_ = cv::cuda::ORB::create(1);
			}

			void compute(InputArray _image, KeyPoints &_keypoints, OutputArray _descriptors) override
			{
				orb_cpu_->compute(_image, _keypoints, _descriptors); // CPU only
			}

			void convert(InputArray src, CV_OUT std::vector<KeyPoint> &dst)
			{
				Mat tmp;
				if (src.kind() == _InputArray::KindFlag::MAT)
					tmp = src.getMat();
				else if (src.kind() == _InputArray::KindFlag::CUDA_GPU_MAT)
					src.getGpuMat().download(tmp);

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
				// 作業中なのでCPU only
				cv::Mat image, descriptors;
				std::vector<KeyPoint> keypoints;
				if (_image.kind() == _InputArray::KindFlag::MAT)
					image = _image.getMat();
				else if (_image.kind() == _InputArray::KindFlag::CUDA_GPU_MAT)
					_image.getGpuMat().download(image);
				convert(_keypoints, keypoints);
				orb_cpu_->compute(image, keypoints, descriptors);
				descriptors_.upload(descriptors, stream);
#if 0
				cv::Mat tmp(6, keypoints.size(), CV_32F);
				for (auto &kpt : keypoints)
				{
					tmp.at<float>(0, 0) = kpt.pt.x;
					tmp.at<float>(1, 0) = kpt.pt.y;
					tmp.at<float>(2, 0) = kpt.response;
					tmp.at<float>(3, 0) = kpt.angle;
					tmp.at<float>(4, 0) = kpt.octave;
					tmp.at<float>(5, 0) = kpt.size;
				}
				keypoints_.upload(tmp, stream);
				orb_gpu_->computeAsync(_image, keypoints_, descriptors_, stream);
				if (_descriptors.kind() == _InputArray::KindFlag::MAT)
					descriptors_.download(_descriptors);
#endif
			}

			int descriptorSize() const override { return 32; }
			int descriptorType() const override { return CV_8U; }
			int defaultNorm() const override { return NORM_HAMMING; }

		private:
			float scaleFactor_;
			int paramSize_;
			Size patchSize_;
			cv::Ptr<cv::ORB> orb_cpu_;
			cv::Ptr<cv::cuda::ORB> orb_gpu_;

			GpuMat image_, keypoints_, descriptors_, integral_;
			DeviceBuffer buf_;
		};

		Ptr<EORB> EORB::create(float scaleFactor)
		{
			return makePtr<EORB_Impl>(scaleFactor);
		}

	} // namespace cuda
} // namespace cv
