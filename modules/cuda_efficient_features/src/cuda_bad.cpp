/*
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

#include <algorithm>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "cuda_bad_internal.h"
#include "cuda_efficient_features_internal.h"
#include "cuda_orb_internal.h"
#include "device_buffer.h"

namespace cv
{
namespace cuda
{

namespace
{
        constexpr int HALF_PATCH = 16;

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

        struct BADBuffers
        {
                GpuMat image, keypoints, descriptors, integral;
                DeviceBuffer buf;
        };

        template <bool WrapHorizontal>
        void computeBADDescriptors(InputArray _image, const std::variant<_InputArray, KeyPoints>& _keypoints,
                OutputArray _descriptors, Stream& stream, float scaleFactor, int paramSize, Size patchSize,
                BADBuffers& buffers, const SphericalLensParams& lens, Size& lensBaseSize)
        {
                if (_image.empty())
                        return;

                if (isEmpty(_keypoints))
                {
                        _descriptors.release();
                        return;
                }

                CV_Assert(_image.type() == CV_8U);

                const Size imageSize = _image.size();
                const SphericalLensParams resolvedLens = scaleLensForImage(lens, imageSize, lensBaseSize);

                getInputMat(_image, buffers.image, stream);

                if constexpr (WrapHorizontal)
                {
                        GpuMat horizontalWrapped;
                        cv::cuda::copyMakeBorder(buffers.image, horizontalWrapped, 0, 0, HALF_PATCH, HALF_PATCH, BORDER_WRAP,
                                Scalar(), stream);
                        cv::cuda::copyMakeBorder(horizontalWrapped, buffers.image, HALF_PATCH, HALF_PATCH, 0, 0,
                                BORDER_REFLECT_101, Scalar(), stream);
                }

                gpu::calcIntegralImage(buffers.image, buffers.integral, stream);

                getKeypointsMat(_keypoints, buffers.keypoints, stream);

                if constexpr (WrapHorizontal)
                {
                        gpu::normalizeSphericalKeypoints(buffers.keypoints, imageSize, resolvedLens,
                                StreamAccessor::getStream(stream));
                        cv::cuda::add(buffers.keypoints, Scalar(HALF_PATCH, HALF_PATCH, 0, 0), buffers.keypoints, noArray(), -1,
                                stream);
                }

                getOutputMat(_descriptors, buffers.descriptors, buffers.keypoints.rows, paramSize / 8, CV_8U);

                gpu::computeBAD(buffers.integral, buffers.keypoints, buffers.descriptors, scaleFactor, paramSize, patchSize,
                        StreamAccessor::getStream(stream));

                if (_descriptors.kind() == _InputArray::KindFlag::MAT)
                        buffers.descriptors.download(_descriptors);
        }
} // namespace

class BADImpl : public BAD
{
public:

	BADImpl(float scaleFactor, int nbits) : scaleFactor_(scaleFactor), nbits_(nbits), patchSize_(32, 32)
	{
		paramSize_ = nbits == SIZE_256_BITS ? 256 : 512;
		gpu::loadBoxPairParams(paramSize_);
	}

	void computeBAD(InputArray _image, InputKeyPoints _keypoints, OutputArray _descriptors, Stream& stream)
	{
		if (_image.empty())
			return;

		if (isEmpty(_keypoints))
		{
			// clean output buffer (it may be reused with "allocated" data)
			_descriptors.release();
			return;
		}

		CV_Assert(_image.type() == CV_8U);

		getInputMat(_image, image_, stream);
		getKeypointsMat(_keypoints, keypoints_, stream);
		getOutputMat(_descriptors, descriptors_, keypoints_.rows, descriptorSize(), descriptorType());

		integral_ = buf_.createMat(image_.rows + 1, image_.cols + 1, CV_32S);
		gpu::calcIntegralImage(image_, integral_, stream);
		gpu::computeBAD(integral_, keypoints_, descriptors_, scaleFactor_, paramSize_, patchSize_, StreamAccessor::getStream(stream));

		if (_descriptors.kind() == _InputArray::KindFlag::MAT)
			descriptors_.download(_descriptors);
	}

	void compute(InputArray _image, KeyPoints& _keypoints, OutputArray _descriptors) override
	{
		computeBAD(_image, _keypoints, _descriptors, Stream::Null());
	}

	void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream& stream) override
	{
		computeBAD(_image, _keypoints, _descriptors, stream);
	}

	int descriptorSize() const override { return paramSize_ / 8; }
	int descriptorType() const override { return CV_8U; }
	int defaultNorm() const override { return NORM_HAMMING; }

private:

	float scaleFactor_;
	int nbits_;
	Size patchSize_;
	int paramSize_;

        GpuMat image_, keypoints_, descriptors_, integral_;
        DeviceBuffer buf_;
};

class SphericalBADImpl : public SphericalBAD
{
public:

        SphericalBADImpl(float scaleFactor, int nbits, SphericalLensParams lensParams)
                : scaleFactor_(scaleFactor), nbits_(nbits), patchSize_(32, 32), lensParams_(lensParams)
        {
                paramSize_ = nbits == BAD::SIZE_256_BITS ? 256 : 512;
                gpu::loadBoxPairParams(paramSize_);
        }

        void compute(InputArray _image, KeyPoints& _keypoints, OutputArray _descriptors) override
        {
                const std::variant<_InputArray, KeyPoints> keypoints = _keypoints;
                const SphericalLensParams scaledLens = scaleLensForImage(lensParams_, _image.size(), lensBaseSize_);
                computeBADDescriptors<true>(_image, keypoints, _descriptors, Stream::Null(), scaleFactor_, paramSize_, patchSize_,
                        buffers_, scaledLens, lensBaseSize_);
        }

        void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream& stream) override
        {
                const std::variant<_InputArray, KeyPoints> keypoints = _keypoints;
                const SphericalLensParams scaledLens = scaleLensForImage(lensParams_, _image.size(), lensBaseSize_);
                computeBADDescriptors<true>(_image, keypoints, _descriptors, stream, scaleFactor_, paramSize_, patchSize_,
                        buffers_, scaledLens, lensBaseSize_);
        }

        int descriptorSize() const override { return paramSize_ / 8; }
        int descriptorType() const override { return CV_8U; }
        int defaultNorm() const override { return NORM_HAMMING; }

private:
        float scaleFactor_;
        int nbits_;
        Size patchSize_;
        int paramSize_;

        SphericalLensParams lensParams_;
        Size lensBaseSize_;

        BADBuffers buffers_;
};

Ptr<BAD> BAD::create(float scaleFactor, int nbits)
{
        return makePtr<BADImpl>(scaleFactor, nbits);
}

Ptr<SphericalBAD> SphericalBAD::create(float scaleFactor, int nbits, SphericalLensParams lensParams)
{
        return makePtr<SphericalBADImpl>(scaleFactor, nbits, lensParams);
}

} // namespace cuda
} // namespace cv
