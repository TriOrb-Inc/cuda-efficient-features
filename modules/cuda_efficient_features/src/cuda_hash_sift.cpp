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
#include <cstdlib>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <cublas_v2.h>

#include "cuda_hash_sift_internal.h"
#include "cuda_efficient_features_internal.h"
#include "cuda_orb_internal.h"
#include "spherical_projection.hpp"
#include "device_buffer.h"

namespace cv
{
namespace cuda
{

#define CUBLAS_CHECK(err) \
do {\
	if (err != CUBLAS_STATUS_SUCCESS) { \
		printf("[CUBLAS Error] (code: %d) at %s:%d\n", err, __FILE__, __LINE__); \
	} \
} while (0)

static void hashSIFTGemm(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const cublasHandle_t& handle)
{
	CV_Assert( src1.type() == CV_32FC1 );
	CV_Assert( src1.cols == src2.cols );

	const float alphaf = 1.0f;
	const float betaf = 0.0f;
	const cublasOperation_t transa = CUBLAS_OP_T;
	const cublasOperation_t transb = CUBLAS_OP_N;

	CUBLAS_CHECK( cublasSgemm_v2(handle, transa, transb, src2.rows, src1.rows, src2.cols,
		&alphaf,
		src2.ptr<float>(), static_cast<int>(src2.step / sizeof(float)),
		src1.ptr<float>(), static_cast<int>(src1.step / sizeof(float)),
		&betaf,
		dst.ptr<float>(), static_cast<int>(dst.step / sizeof(float))) );
}

class MatmulAndSign
{
public:

	MatmulAndSign()
	{
		CUBLAS_CHECK( cublasCreate_v2(&handle_) );
		CUBLAS_CHECK( cublasSetPointerMode_v2(handle_, CUBLAS_POINTER_MODE_HOST) );
		CUBLAS_CHECK( cublasSetAtomicsMode(handle_, CUBLAS_ATOMICS_NOT_ALLOWED) );
		CUBLAS_CHECK( cublasSetMathMode(handle_, CUBLAS_DEFAULT_MATH) );
	}

	~MatmulAndSign()
	{
		CUBLAS_CHECK( cublasDestroy_v2(handle_) );
	}

	void operator()(const GpuMat& responses, const GpuMat& bMatrix, GpuMat& descriptors, Stream& stream)
	{
		CV_Assert(responses.rows == descriptors.rows);
		if (isDeterministicProjectionEnabled())
		{
			gpu::projectAndBinarizeHashSIFTDeterministic(
				responses, bMatrix, descriptors, StreamAccessor::getStream(stream));
			return;
		}

		CUBLAS_CHECK( cublasSetStream_v2(handle_, StreamAccessor::getStream(stream)) );

		GpuMat tmp = bufTmp_.createMat(responses.rows, bMatrix.rows, responses.type());
		hashSIFTGemm(responses, bMatrix, tmp, handle_);
		gpu::binarizeDescriptors(tmp, descriptors, StreamAccessor::getStream(stream));
	}

private:

	static bool isDeterministicHashSIFTEnabled()
	{
		const char* value = std::getenv("TRIORB_CUDA_HASH_SIFT_DETERMINISTIC");
		if (value == nullptr || value[0] == '\0')
			value = std::getenv("TRIORB_CUDA_FEATURES_DETERMINISTIC");
		if (value == nullptr || value[0] == '\0')
			return true;
		if (value[0] == '0')
			return false;
		if ((value[0] == 'f' || value[0] == 'F') && (value[1] == 'a' || value[1] == 'A'))
			return false;
		if ((value[0] == 'n' || value[0] == 'N') && (value[1] == 'o' || value[1] == 'O'))
			return false;
		if ((value[0] == 'o' || value[0] == 'O') && (value[1] == 'f' || value[1] == 'F'))
			return false;
		return true;
	}

	static bool isDeterministicProjectionEnabled()
	{
		const char* value = std::getenv("TRIORB_CUDA_HASH_SIFT_DETERMINISTIC_PROJECT");
		if (value == nullptr || value[0] == '\0')
			return false;
		if (value[0] == '0')
			return false;
		if ((value[0] == 'f' || value[0] == 'F') && (value[1] == 'a' || value[1] == 'A'))
			return false;
		if ((value[0] == 'n' || value[0] == 'N') && (value[1] == 'o' || value[1] == 'O'))
			return false;
		if ((value[0] == 'o' || value[0] == 'O') && (value[1] == 'f' || value[1] == 'F'))
			return false;
		return isDeterministicHashSIFTEnabled();
	}

	DeviceBuffer bufTmp_;
        cublasHandle_t handle_;
};

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
} // namespace

class SIFTImpl : public SIFT
{
public:

        explicit SIFTImpl(float croppingScale) : croppingScale_(croppingScale)
        {
        }

        void computeSIFT(InputArray _image, InputKeyPoints _keypoints, OutputArray _descriptors, Stream& stream)
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
                getKeypointsMat(_keypoints, keypoints_, stream);
                getOutputMat(_descriptors, descriptors_, keypoints_.rows, descriptorSize(), descriptorType());

                GpuMat responses = bufResponses_.createMat(keypoints_.rows, 129, CV_32F);
                SphericalSamplingParams sphericalParams;
                gpu::computePatchSIFTs(image_, keypoints_, responses, croppingScale_, 1./6, 1.6, sphericalParams,
                        StreamAccessor::getStream(stream));

                GpuMat raw = responses.colRange(1, responses.cols);
                raw.copyTo(descriptors_, stream);

                if (_descriptors.kind() == _InputArray::KindFlag::MAT)
                        descriptors_.download(_descriptors);
        }

        void compute(InputArray _image, KeyPoints& _keypoints, OutputArray _descriptors) override
        {
                computeSIFT(_image, _keypoints, _descriptors, Stream::Null());
        }

        void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream& stream) override
        {
                computeSIFT(_image, _keypoints, _descriptors, stream);
        }

        int descriptorSize() const override { return 128; }
        int descriptorType() const override { return CV_32F; }
        int defaultNorm() const override { return NORM_L2; }

private:

        float croppingScale_;
        GpuMat image_, keypoints_, descriptors_;
        DeviceBuffer bufResponses_;
};

class SphericalSIFTImpl : public SphericalSIFT
{
public:

        SphericalSIFTImpl(float croppingScale, SphericalLensParams lensParams)
                : croppingScale_(croppingScale), lensParams_(lensParams)
        {
        }

        void computeSIFT(InputArray _image, InputKeyPoints _keypoints, OutputArray _descriptors, Stream& stream,
                const SphericalLensParams& lens)
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
                const SphericalLensParams resolvedLens = resolveLens(lens, imageSize);
                const SphericalProjection projection = buildSphericalProjection(resolvedLens, imageSize);

                getInputMat(_image, image_, stream);

                if (projection.wrapHorizontal)
                {
                        GpuMat padded;
                        cv::cuda::copyMakeBorder(image_, padded, 0, 0, HALF_PATCH, HALF_PATCH, BORDER_WRAP, Scalar(), stream);
                        cv::cuda::copyMakeBorder(padded, image_, HALF_PATCH, HALF_PATCH, 0, 0, BORDER_REFLECT_101, Scalar(), stream);
                }
                else
                {
                        cv::cuda::copyMakeBorder(image_, image_, HALF_PATCH, HALF_PATCH, HALF_PATCH, HALF_PATCH, BORDER_REFLECT_101,
                                Scalar(), stream);
                }

                getKeypointsMat(_keypoints, keypoints_, stream);
                gpu::normalizeSphericalKeypoints(keypoints_, imageSize, resolvedLens, projection,
                        StreamAccessor::getStream(stream));
                cv::cuda::add(keypoints_, Scalar(HALF_PATCH, HALF_PATCH, 0, 0), keypoints_, noArray(), -1, stream);

                getOutputMat(_descriptors, descriptors_, keypoints_.rows, descriptorSize(), descriptorType());

                SphericalSamplingParams sphericalParams;
                if (hasValidLens(lens))
                {
                        sphericalParams.enabled = 1;
                        sphericalParams.wrapHorizontal = projection.wrapHorizontal ? 1 : 0;
                        sphericalParams.padding = HALF_PATCH;
                        sphericalParams.imageWidth = imageSize.width;
                        sphericalParams.imageHeight = imageSize.height;
                        sphericalParams.lens = resolvedLens;
                        sphericalParams.projection = projection;
                }

                GpuMat responses = bufResponses_.createMat(keypoints_.rows, 129, CV_32F);
                gpu::computePatchSIFTs(image_, keypoints_, responses, croppingScale_, 1./6, 1.6, sphericalParams,
                        StreamAccessor::getStream(stream));

                GpuMat raw = responses.colRange(1, responses.cols);
                raw.copyTo(descriptors_, stream);

                if (_descriptors.kind() == _InputArray::KindFlag::MAT)
                        descriptors_.download(_descriptors);
        }

        void compute(InputArray _image, KeyPoints& _keypoints, OutputArray _descriptors) override
        {
                const std::variant<_InputArray, KeyPoints> keypoints = _keypoints;
                const SphericalLensParams scaledLens = scaleLensForImage(lensParams_, _image.size(), lensBaseSize_);
                computeSIFT(_image, keypoints, _descriptors, Stream::Null(), scaledLens);
        }

        void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream& stream) override
        {
                const std::variant<_InputArray, KeyPoints> keypoints = _keypoints;
                const SphericalLensParams scaledLens = scaleLensForImage(lensParams_, _image.size(), lensBaseSize_);
                computeSIFT(_image, keypoints, _descriptors, stream, scaledLens);
        }

        int descriptorSize() const override { return 128; }
        int descriptorType() const override { return CV_32F; }
        int defaultNorm() const override { return NORM_L2; }

private:

        float croppingScale_;
        SphericalLensParams lensParams_;
        Size lensBaseSize_;

        GpuMat image_, keypoints_, descriptors_;
        DeviceBuffer bufResponses_;
};

class HashSIFTImpl : public HashSIFT
{
public:

	HashSIFTImpl(float croppingScale, int nbits) : croppingScale_(croppingScale)
	{
#include "hash_sift.p512.h"
#include "hash_sift.p256.h"

		if (nbits == SIZE_512_BITS)
			Mat(512, 129, CV_64F, (void*)HASH_SIFT_512_VALS).convertTo(bMatrix_, CV_32F);
		else if (nbits == SIZE_256_BITS)
			Mat(256, 129, CV_64F, (void*)HASH_SIFT_256_VALS).convertTo(bMatrix_, CV_32F);
		else
			CV_Error(Error::StsBadArg, "n_bits should be either SIZE_512_BITS or SIZE_256_BITS");

		nbits_ = bMatrix_.rows;
		d_bMatrix_.upload(bMatrix_);
	}

	void computeHashSIFT(InputArray _image, InputKeyPoints _keypoints, OutputArray _descriptors, Stream& stream)
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

		GpuMat responses = bufResponses_.createMat(keypoints_.rows, 129, CV_32F);
                SphericalSamplingParams sphericalParams;
		gpu::computePatchSIFTs(image_, keypoints_, responses, croppingScale_, 1./6, 1.6, sphericalParams,
                        StreamAccessor::getStream(stream));
		matmulAndSign_(responses, d_bMatrix_, descriptors_, stream);

		if (_descriptors.kind() == _InputArray::KindFlag::MAT)
			descriptors_.download(_descriptors);
	}

	void compute(InputArray _image, KeyPoints& _keypoints, OutputArray _descriptors) override
	{
		computeHashSIFT(_image, _keypoints, _descriptors, Stream::Null());
	}

	void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream& stream) override
	{
		computeHashSIFT(_image, _keypoints, _descriptors, stream);
	}

	int descriptorSize() const override { return nbits_ / 8; }
	int descriptorType() const override { return CV_8U; }
	int defaultNorm() const override { return NORM_HAMMING; }

private:

	float croppingScale_;
	Mat bMatrix_;
	int nbits_;

	GpuMat image_, keypoints_, descriptors_, d_bMatrix_;
	DeviceBuffer bufResponses_;
        MatmulAndSign matmulAndSign_;
};

class SphericalHashSIFTImpl : public SphericalHashSIFT
{
public:

        SphericalHashSIFTImpl(float croppingScale, int nbits, SphericalLensParams lensParams)
                : croppingScale_(croppingScale), lensParams_(lensParams)
        {
#include "hash_sift.p512.h"
#include "hash_sift.p256.h"

                if (nbits == HashSIFT::SIZE_512_BITS)
                        Mat(512, 129, CV_64F, (void*)HASH_SIFT_512_VALS).convertTo(bMatrix_, CV_32F);
                else if (nbits == HashSIFT::SIZE_256_BITS)
                        Mat(256, 129, CV_64F, (void*)HASH_SIFT_256_VALS).convertTo(bMatrix_, CV_32F);
                else
                        CV_Error(Error::StsBadArg, "n_bits should be either SIZE_512_BITS or SIZE_256_BITS");

                nbits_ = bMatrix_.rows;
                d_bMatrix_.upload(bMatrix_);
        }

        void computeHashSIFT(InputArray _image, InputKeyPoints _keypoints, OutputArray _descriptors, Stream& stream,
                const SphericalLensParams& lens)
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
                const SphericalLensParams resolvedLens = resolveLens(lens, imageSize);
                const SphericalProjection projection = buildSphericalProjection(resolvedLens, imageSize);

                getInputMat(_image, image_, stream);

                if (projection.wrapHorizontal)
                {
                        GpuMat padded;
                        cv::cuda::copyMakeBorder(image_, padded, 0, 0, HALF_PATCH, HALF_PATCH, BORDER_WRAP, Scalar(), stream);
                        cv::cuda::copyMakeBorder(padded, image_, HALF_PATCH, HALF_PATCH, 0, 0, BORDER_REFLECT_101, Scalar(), stream);
                }
                else
                {
                        cv::cuda::copyMakeBorder(image_, image_, HALF_PATCH, HALF_PATCH, HALF_PATCH, HALF_PATCH, BORDER_REFLECT_101,
                                Scalar(), stream);
                }

                getKeypointsMat(_keypoints, keypoints_, stream);
                gpu::normalizeSphericalKeypoints(keypoints_, imageSize, resolvedLens, projection, StreamAccessor::getStream(stream));
                cv::cuda::add(keypoints_, Scalar(HALF_PATCH, HALF_PATCH, 0, 0), keypoints_, noArray(), -1, stream);

                getOutputMat(_descriptors, descriptors_, keypoints_.rows, descriptorSize(), descriptorType());

                SphericalSamplingParams sphericalParams;
                if (hasValidLens(lens))
                {
                        sphericalParams.enabled = 1;
                        sphericalParams.wrapHorizontal = projection.wrapHorizontal ? 1 : 0;
                        sphericalParams.padding = HALF_PATCH;
                        sphericalParams.imageWidth = imageSize.width;
                        sphericalParams.imageHeight = imageSize.height;
                        sphericalParams.lens = resolvedLens;
                        sphericalParams.projection = projection;
                }

                GpuMat responses = bufResponses_.createMat(keypoints_.rows, 129, CV_32F);
                gpu::computePatchSIFTs(image_, keypoints_, responses, croppingScale_, 1./6, 1.6, sphericalParams,
                        StreamAccessor::getStream(stream));
                matmulAndSign_(responses, d_bMatrix_, descriptors_, stream);

                if (_descriptors.kind() == _InputArray::KindFlag::MAT)
                        descriptors_.download(_descriptors);
        }

        void compute(InputArray _image, KeyPoints& _keypoints, OutputArray _descriptors) override
        {
                const std::variant<_InputArray, KeyPoints> keypoints = _keypoints;
                const SphericalLensParams scaledLens = scaleLensForImage(lensParams_, _image.size(), lensBaseSize_);
                computeHashSIFT(_image, keypoints, _descriptors, Stream::Null(), scaledLens);
        }

        void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream& stream) override
        {
                const std::variant<_InputArray, KeyPoints> keypoints = _keypoints;
                const SphericalLensParams scaledLens = scaleLensForImage(lensParams_, _image.size(), lensBaseSize_);
                computeHashSIFT(_image, keypoints, _descriptors, stream, scaledLens);
        }

        int descriptorSize() const override { return nbits_ / 8; }
        int descriptorType() const override { return CV_8U; }
        int defaultNorm() const override { return NORM_HAMMING; }

private:

        float croppingScale_;
        Mat bMatrix_;
        int nbits_;

        SphericalLensParams lensParams_;
        Size lensBaseSize_;

        GpuMat image_, keypoints_, descriptors_, d_bMatrix_;
        DeviceBuffer bufResponses_;
        MatmulAndSign matmulAndSign_;
};

Ptr<HashSIFT> HashSIFT::create(float croppingScale, int nbits)
{
        return makePtr<HashSIFTImpl>(croppingScale, nbits);
}

Ptr<SphericalHashSIFT> SphericalHashSIFT::create(float croppingScale, int nbits, SphericalLensParams lensParams)
{
        return makePtr<SphericalHashSIFTImpl>(croppingScale, nbits, lensParams);
}

Ptr<SIFT> SIFT::create(float croppingScale)
{
        return makePtr<SIFTImpl>(croppingScale);
}

Ptr<SphericalSIFT> SphericalSIFT::create(float croppingScale, SphericalLensParams lensParams)
{
        return makePtr<SphericalSIFTImpl>(croppingScale, lensParams);
}

} // namespace cuda
} // namespace cv
