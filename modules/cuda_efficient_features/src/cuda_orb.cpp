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

		class EORB_Impl : public EORB
		{

		public:
			EORB_Impl(float scaleFactor) : scaleFactor_(scaleFactor), paramSize_(256), patchSize_(32, 32)
			{
				orb_cpu_ = cv::ORB::create(1);
				orb_gpu_ = cv::cuda::ORB::create(1);
			}

			void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream &stream) override
			{
				getInputMat(_image, image_, stream);
				getKeypointsMat(_keypoints, keypoints_, stream);
				getOutputMat(_descriptors, descriptors_, keypoints_.rows, descriptorSize(), descriptorType());
				orb_gpu_->computeAsync(image_, keypoints_, descriptors_, stream);
				if (_descriptors.kind() == _InputArray::KindFlag::MAT)
					descriptors_.download(_descriptors);
			}

			void compute(InputArray _image, KeyPoints &_keypoints, OutputArray _descriptors) override
			{
				orb_cpu_->compute(_image, _keypoints, _descriptors); // CPU only
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
