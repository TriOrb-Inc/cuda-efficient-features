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

#ifndef __CUDA_EFFICIENT_FEATURES_H__
#define __CUDA_EFFICIENT_FEATURES_H__

#include <cstdint>

#include <opencv2/core/cuda.hpp>
#include <opencv2/features2d.hpp>

#include "cuda_efficient_descriptors.h"

namespace cv
{
namespace cuda
{

class EfficientFeatures : public Feature2D
{
public:

	static const int LOCATION_ROW = 0;
	static const int RESPONSE_ROW = 1;
	static const int ANGLE_ROW    = 2;
	static const int OCTAVE_ROW   = 3;
	static const int SIZE_ROW     = 4;
	static const int ROWS_COUNT   = 5;

        enum DescriptorType
        {
                BAD_256,
                BAD_512,
                HASH_SIFT_256,
                HASH_SIFT_512,
                SPHERICAL_BAD_256,
                SPHERICAL_BAD_512,
                SPHERICAL_HASH_SIFT_256,
                SPHERICAL_HASH_SIFT_512,
                AKAZE_256,
                AKAZE_512,
                SPHERICAL_AKAZE_256,
                SPHERICAL_AKAZE_512,
                ORB,
                SPHERICAL_ORB,
                SIFT,
                SPHERICAL_SIFT,
        };

	static Ptr<EfficientFeatures> create(int nfeatures = 5000, float scaleFactor = 1.2f, int nlevels = 8,
		int firstLevel = 0, int fastThreshold = 20, int nonmaxRadius = 15, DescriptorType dtype = HASH_SIFT_256);

	virtual ~EfficientFeatures();

	/** @brief Detects keypoints in an image.

	@param image Image.
	@param keypoints The detected keypoints.
	@param mask Mask specifying where to look for keypoints (optional). It must be a 8-bit integer
	matrix with non-zero values in the region of interest.
	@param stream CUDA stream.
	 */
	virtual void detectAsync(InputArray image, OutputArray keypoints, InputArray mask = noArray(), Stream& stream = Stream::Null()) = 0;

	/** @brief Computes the descriptors for a set of keypoints detected in an image.

	@param image Image.
	@param keypoints Input collection of keypoints.
	@param descriptors Computed descriptors. Row j is the descriptor for j-th keypoint.
	@param stream CUDA stream.
	 */
	virtual void computeAsync(InputArray image, InputArray keypoints, OutputArray descriptors, Stream& stream = Stream::Null()) = 0;

	/** Detects keypoints and computes the descriptors. */
	virtual void detectAndComputeAsync(InputArray image, InputArray mask, OutputArray keypoints, OutputArray descriptors,
		bool useProvidedKeypoints = false, Stream& stream = Stream::Null()) = 0;

	/** Detects keypoints and computes descriptors per channel for multi-channel input. */
	virtual void detectAndComputeAsyncPerChannel(InputArray image, InputArray mask,
		std::vector<GpuMat>& keypoints, std::vector<GpuMat>& descriptors,
		bool useProvidedKeypoints = false, Stream& stream = Stream::Null()) = 0;

	/** Converts keypoints array from internal representation to standard vector. */
	virtual void convert(InputArray gpu_keypoints, std::vector<KeyPoint>& keypoints) = 0;

	virtual void setMaxFeatures(int maxFeatures) = 0;
	virtual int getMaxFeatures() const = 0;

	virtual void setScaleFactor(float scaleFactor) = 0;
	virtual float getScaleFactor() const = 0;

	virtual void setNLevels(int nlevels) = 0;
	virtual int getNLevels() const = 0;

	virtual void setFirstLevel(int firstLevel) = 0;
	virtual int getFirstLevel() const = 0;

	virtual void setFastThreshold(int fastThreshold) = 0;
	virtual int getFastThreshold() const = 0;

	virtual void setNonmaxRadius(int nonmaxRadius) = 0;
	virtual int getNonmaxRadius() const = 0;

	virtual void setDescriptorType(DescriptorType descriptorType, const SphericalLensParams& params = SphericalLensParams()) = 0;
	virtual DescriptorType getDescriptorType() const = 0;
	virtual SphericalLensParams getSphericalLensParams() const = 0;

	/**
	 * @brief Enable deterministic keypoint ordering across runs.
	 *
	 * When enabled, the extractor canonicalizes keypoint array ordering after
	 * radius-based non-maximum suppression (which otherwise uses atomicAdd-based
	 * output packing and scrambles order) and uses a stable sort inside
	 * `limitPoints`, so that tied Harris responses are broken by a canonical
	 * (y, x) ordering. This removes the cross-run non-determinism in which
	 * subset of keypoints survive truncation to `nfeatures`.
	 *
	 * Default is `false` to preserve legacy behavior and avoid the small sort
	 * overhead when determinism is not required.
	 */
	virtual void setDeterministic(bool /*deterministic*/) { }
	virtual bool isDeterministic() const { return false; }

	/**
	 * @brief Set optional read-only trace context for the next extraction call.
	 *
	 * The default implementation is a no-op so existing callers are unaffected.
	 * Downstream wrappers use this to attach sensor / timestamp labels to
	 * diagnostic-only stage fingerprints without changing extraction results.
	 */
	virtual void setDiagnosticContext(const char* /*sensorId*/, std::uint64_t /*timestamp*/, int /*slotIndex*/) { }
};

} // namespace cuda
} // namespace cv

#endif // !__CUDA_EFFICIENT_FEATURES_H__
