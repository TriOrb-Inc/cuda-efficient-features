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
#include <iostream>

#include "cuda_efficient_features.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "device_buffer.h"
#include "cuda_efficient_features_internal.h"
#include "cuda_efficient_descriptors.h"

namespace cv
{
namespace cuda
{

static constexpr int PATCH_SIZE = 31;
static constexpr int HALF_PATCH_SIZE = 15;
static constexpr double CORNER_DENSITY = 0.1;

static std::uint64_t mixU64Fingerprint(std::uint64_t hash, const std::uint64_t value)
{
        for (std::size_t shift = 0; shift < 64; shift += 8)
        {
                const auto byte = static_cast<std::uint8_t>((value >> shift) & 0xffU);
                hash ^= static_cast<std::uint64_t>(byte);
                hash *= 0x00000100000001b3ULL;
        }
        return hash;
}

static std::uint64_t coordinateFingerprint(const Vec2s& point)
{
        std::uint64_t hash = 0xcbf29ce484222325ULL;
        hash = mixU64Fingerprint(hash, static_cast<std::uint64_t>(point[0]));
        hash = mixU64Fingerprint(hash, static_cast<std::uint64_t>(point[1]));
        return hash;
}

static std::uint64_t orderedCoordinateFingerprint(const std::vector<Vec2s>& points)
{
        std::uint64_t hash = 0xcbf29ce484222325ULL;
        for (const auto& point : points)
        {
                hash = mixU64Fingerprint(hash, coordinateFingerprint(point));
        }
        return hash;
}

static std::uint64_t unorderedCoordinateSetFingerprint(std::vector<Vec2s> points)
{
        std::sort(points.begin(), points.end(), [](const Vec2s& lhs, const Vec2s& rhs) {
                if (lhs[1] != rhs[1])
                        return lhs[1] < rhs[1];
                return lhs[0] < rhs[0];
        });

        std::uint64_t hash = 0xcbf29ce484222325ULL;
        for (const auto& point : points)
        {
                hash = mixU64Fingerprint(hash, coordinateFingerprint(point));
        }
        return hash;
}

struct FeatureResponseStats
{
        float minValue = 0.0f;
        float p50Value = 0.0f;
        float p90Value = 0.0f;
        float maxValue = 0.0f;
        float meanValue = 0.0f;
};

static FeatureResponseStats computeResponseStats(std::vector<float> responses)
{
        FeatureResponseStats stats;
        if (responses.empty())
                return stats;

        double sum = 0.0;
        for (const auto response : responses)
                sum += static_cast<double>(response);

        std::sort(responses.begin(), responses.end());
        const auto percentileAt = [&responses](const double percentile) {
                const auto lastIndex = static_cast<double>(responses.size() - 1U);
                const auto rawIndex = static_cast<std::size_t>(std::llround(lastIndex * percentile));
                return responses[std::min(rawIndex, responses.size() - 1U)];
        };

        stats.minValue = responses.front();
        stats.p50Value = percentileAt(0.50);
        stats.p90Value = percentileAt(0.90);
        stats.maxValue = responses.back();
        stats.meanValue = static_cast<float>(sum / static_cast<double>(responses.size()));
        return stats;
}

static int featureStageFingerprintFrameLimit()
{
        const char* value = std::getenv("TRIORB_CUDA_FEATURE_STAGE_FINGERPRINT_MAX_FRAMES");
        if (value == nullptr || value[0] == '\0')
                return 0;

        char* end = nullptr;
        const auto parsed = std::strtol(value, &end, 10);
        if (end == value)
                return 0;
        if (parsed < 0)
                return 0;
        if (parsed > 1000)
                return 1000;
        return static_cast<int>(parsed);
}

static bool shouldLogFeatureStageFingerprint(const std::string& sensorId, const std::uint64_t timestamp)
{
        if (sensorId.empty())
                return false;

        const auto frameLimit = featureStageFingerprintFrameLimit();
        if (frameLimit <= 0)
                return false;

        static std::mutex mutex;
        static std::unordered_map<std::string, std::set<std::uint64_t>> loggedTimestampsBySensor;

        std::lock_guard<std::mutex> lock(mutex);
        auto& loggedTimestamps = loggedTimestampsBySensor[sensorId];
        if (loggedTimestamps.find(timestamp) != loggedTimestamps.end())
                return true;
        if (static_cast<int>(loggedTimestamps.size()) >= frameLimit)
                return false;
        loggedTimestamps.insert(timestamp);
        return true;
}

static void logFeatureStageFingerprint(
        const std::string& sensorId,
		const std::uint64_t timestamp,
        const int slotIndex,
        const int level,
        const char* stage,
        const GpuMat& points,
        const bool hasResponse,
        Stream& stream)
{
        if (!shouldLogFeatureStageFingerprint(sensorId, timestamp) || points.empty())
                return;

        Mat downloaded;
        points.download(downloaded, stream);
        stream.waitForCompletion();
        if (downloaded.empty() || downloaded.cols <= 0)
                return;

        const int npoints = downloaded.cols;
        const Vec2s* pointRows = downloaded.ptr<Vec2s>(EfficientFeatures::LOCATION_ROW);

        std::vector<Vec2s> coordinates;
        coordinates.reserve(static_cast<std::size_t>(npoints));
        for (int i = 0; i < npoints; ++i)
                coordinates.push_back(pointRows[i]);

        std::vector<float> responses;
        if (hasResponse && downloaded.rows > EfficientFeatures::RESPONSE_ROW)
        {
                const float* responseRows = downloaded.ptr<float>(EfficientFeatures::RESPONSE_ROW);
                responses.reserve(static_cast<std::size_t>(npoints));
                for (int i = 0; i < npoints; ++i)
                        responses.push_back(responseRows[i]);
        }
        const auto responseStats = computeResponseStats(responses);

        std::ostringstream streamLine;
        streamLine << std::setprecision(6)
                << "[cuda_feature_stage_fingerprint]"
                << " sensor_id=" << sensorId
                << " timestamp=" << timestamp
                << " slot_index=" << slotIndex
                << " level=" << level
                << " stage=" << stage
                << " feature_count=" << npoints
                << " xy_order_hash=" << std::hex << std::setw(16) << std::setfill('0')
                << orderedCoordinateFingerprint(coordinates)
                << " xy_set_hash=" << std::setw(16)
                << unorderedCoordinateSetFingerprint(coordinates)
                << std::dec << std::setfill(' ')
                << " has_response=" << (hasResponse ? 1 : 0)
                << std::scientific
                << " response_min=" << responseStats.minValue
                << " response_p50=" << responseStats.p50Value
                << " response_p90=" << responseStats.p90Value
                << " response_max=" << responseStats.maxValue
                << " response_mean=" << responseStats.meanValue;
        const int sampleCount = std::min(npoints, 3);
        for (int i = 0; i < sampleCount; ++i)
        {
                streamLine << std::defaultfloat
                        << " sample" << i << "_x=" << coordinates[static_cast<std::size_t>(i)][0]
                        << " sample" << i << "_y=" << coordinates[static_cast<std::size_t>(i)][1];
                if (hasResponse && static_cast<std::size_t>(i) < responses.size())
                {
                        streamLine << std::scientific
                                << " sample" << i << "_response=" << responses[static_cast<std::size_t>(i)];
                }
        }
        std::cout << streamLine.str() << std::endl;
}

void calcKeypoints(const GpuMat& image, const GpuMat& mask, GpuMat& keypoints, int nfeatures, int threshold,
	GpuMat& d_buffer, HostMem& h_buffer, cudaStream_t stream, bool deterministic,
	GpuMat& fullCaptureBuffer, HostMem& fullCaptureHostBuffer, const char* sensorId, std::uint64_t timestamp,
	int slotIndex, int level);
int radiusSuppressionBufferSize(Size imgSize, int npoints);
void radiusSuppression(const GpuMat& src, GpuMat& dst, Size imgSize, float radius,
	GpuMat& d_buffer, HostMem& h_buffer, cudaStream_t stream, bool deterministic);
void limitPoints(GpuMat& points, int maxpoints, cudaStream_t stream, bool deterministic);
void calcResponses(const GpuMat& image, GpuMat& points, cudaStream_t stream);
void calcAngles(const GpuMat& image, GpuMat& points, cudaStream_t stream, float quantizationDeg);
void scalePoints(GpuMat& points, float scale, int octave, cudaStream_t stream);
void convertKeypoints(const GpuMat& src, GpuMat& dst, cudaStream_t stream);

float deterministicAngleQuantizationDeg()
{
	const char* value = std::getenv("TRIORB_CUDA_FEATURE_ANGLE_QUANTIZATION_DEG");
	if (value == nullptr || value[0] == '\0')
		return 0.0f;

	char* end = nullptr;
	const float parsed = std::strtof(value, &end);
	if (end == value || parsed < 0.f)
		return 0.0f;
	return parsed;
}

static Ptr<EfficientDescriptorsAsync> createDescriber(
        EfficientFeatures::DescriptorType descriptorType, const SphericalLensParams& lensParams)
{
        switch (descriptorType)
        {
        case EfficientFeatures::BAD_256:
                return cuda::BAD::create(1, cuda::BAD::SIZE_256_BITS);
		break;
	case EfficientFeatures::BAD_512:
		return cuda::BAD::create(1, cuda::BAD::SIZE_512_BITS);
		break;
        case EfficientFeatures::HASH_SIFT_256:
                return cuda::HashSIFT::create(1, cuda::HashSIFT::SIZE_256_BITS);
                break;
        case EfficientFeatures::HASH_SIFT_512:
                return cuda::HashSIFT::create(1, cuda::HashSIFT::SIZE_512_BITS);
                break;
        case EfficientFeatures::SPHERICAL_BAD_256:
                return cuda::SphericalBAD::create(1, cuda::BAD::SIZE_256_BITS, lensParams);
                break;
        case EfficientFeatures::SPHERICAL_BAD_512:
                return cuda::SphericalBAD::create(1, cuda::BAD::SIZE_512_BITS, lensParams);
                break;
        case EfficientFeatures::SPHERICAL_HASH_SIFT_256:
                return cuda::SphericalHashSIFT::create(1, cuda::HashSIFT::SIZE_256_BITS, lensParams);
                break;
        case EfficientFeatures::SPHERICAL_HASH_SIFT_512:
                return cuda::SphericalHashSIFT::create(1, cuda::HashSIFT::SIZE_512_BITS, lensParams);
                break;
        case EfficientFeatures::AKAZE_256:
                return cuda::AKAZE::create(1, 256);
                break;
        case EfficientFeatures::AKAZE_512:
                return cuda::AKAZE::create(1, 512);
                break;
        case EfficientFeatures::SPHERICAL_AKAZE_256:
                return cuda::SphericalAKAZE::create(1, 256, lensParams);
                break;
        case EfficientFeatures::SPHERICAL_AKAZE_512:
                return cuda::SphericalAKAZE::create(1, 512, lensParams);
                break;
        case EfficientFeatures::ORB:
                return cuda::EORB::create(1);
                break;
        case EfficientFeatures::SPHERICAL_ORB:
                return cuda::SphericalORB::create(1, lensParams);
                break;
        case EfficientFeatures::SIFT:
                return cuda::SIFT::create(1);
                break;
        case EfficientFeatures::SPHERICAL_SIFT:
                return cuda::SphericalSIFT::create(1, lensParams);
                break;
        default:
                return nullptr;
        }

	return nullptr;
}

void getInputMat(InputArray src, GpuMat& dst, Stream& stream)
{
	switch (src.kind())
	{
	case _InputArray::KindFlag::MAT:
		dst.upload(src, stream);
		break;
	case _InputArray::KindFlag::CUDA_GPU_MAT:
		dst = src.getGpuMat();
		break;
	default:
		CV_Error(Error::StsBadArg, "Unsupported");
	}
}

void getOutputMat(OutputArray src, GpuMat& dst, int rows, int cols, int type)
{
	switch (src.kind())
	{
	case _InputArray::KindFlag::MAT:
		dst.create(rows, cols, type);
		break;
	case _InputArray::KindFlag::CUDA_GPU_MAT:
		src.create(rows, cols, type);
		dst = src.getGpuMat();
		break;
	default:
		CV_Error(Error::StsBadArg, "Unsupported");
	}
}

void getKeypointsMat(InputKeyPoints _src, GpuMat& dst, Stream& stream)
{
	if (std::holds_alternative<_InputArray>(_src))
	{
		InputArray src = std::get<_InputArray>(_src);

		GpuMat tmp;
		getInputMat(src, tmp, stream);

		CV_Assert(tmp.rows == 5 && tmp.type() == CV_32F);

		dst.create(tmp.cols, 1, CV_32FC4);
		convertKeypoints(tmp, dst, StreamAccessor::getStream(stream));
	}
	else
	{
		const KeyPoints& src = std::get<KeyPoints>(_src);

		const int nkeypoints = static_cast<int>(src.size());
		Mat tmp(nkeypoints, 1, CV_32FC4);
		for (int i = 0; i < nkeypoints; i++)
		{
			const auto& kpt = src[i];
			tmp.at<cv::Vec4f>(i) = cv::Vec4f(kpt.pt.x, kpt.pt.y, kpt.size, kpt.angle);
		}
		dst.upload(tmp, stream);
	}
}

bool isEmpty(InputKeyPoints keypoints)
{
	return std::visit([](const auto& v) { return v.empty(); }, keypoints);
}

static void calcImagePyramid(const GpuMat& image, std::vector<GpuMat>& images, std::vector<float>& scales,
	float scaleFactor, int nlevels, Stream& stream)
{
	CV_Assert(image.type() == CV_8U);

	images.resize(nlevels);
	scales.resize(nlevels);

	float scale = 1.f;
	image.copyTo(images[0], stream);
	scales[0] = scale;

	for (int s = 1; s < nlevels; s++)
	{
		scale *= scaleFactor;
		const float invScale = 1.f / scale;
		const int h = cvRound(invScale * image.rows);
		const int w = cvRound(invScale * image.cols);
		resize(images[s - 1], images[s], Size(w, h), 0, 0, INTER_LINEAR, stream);
		scales[s] = scale;
	}
}

static void calcNumFeaturesPerLevel(int total, float scaleFactor, int nlevels, std::vector<int>& nfeaturesPerLevel)
{
	// compute number of features in each scale
	nfeaturesPerLevel.resize(nlevels);

	const double factor = 1 / scaleFactor;
	double nfeatues = total * (1 - factor) / (1 - std::pow(factor, nlevels));
	int sumfeatures = 0;
	for (int s = 0; s < nlevels - 1; s++)
	{
		nfeaturesPerLevel[s] = cvRound(nfeatues);
		sumfeatures += nfeaturesPerLevel[s];
		nfeatues *= factor;
	}
	nfeaturesPerLevel[nlevels - 1] = std::max(total - sumfeatures, 0);
}

static void createMask(GpuMat& mask, Size imgSize, int border, Stream& stream)
{
	mask.create(imgSize, CV_8U);
	mask.setTo(Scalar::all(0), stream);
	const Rect ROI(border, border, imgSize.width - 2 * border, imgSize.height - 2 * border);
	mask(ROI).setTo(Scalar::all(255), stream);
}

class EfficientFeaturesImpl : public EfficientFeatures
{
public:

        EfficientFeaturesImpl(int nfeatures, float scaleFactor, int nlevels,
                int firstLevel, int fastThreshold, int nonmaxRadius, DescriptorType descriptorType) : nfeatures_(nfeatures), scaleFactor_(scaleFactor),
                nlevels_(nlevels), firstLevel_(firstLevel), fastThreshold_(fastThreshold), nonmaxRadius_(nonmaxRadius), descriptorType_(descriptorType)
        {
                describer_ = createDescriber(descriptorType, lensParams_);
                filter_ = cuda::createGaussianFilter(CV_8UC1, -1, Size(7, 7), 2, 2, BORDER_REFLECT_101);
                h_buffer_.create(1, 16, CV_32S);
        }

	void detect(InputArray image, std::vector<KeyPoint>& keypoints, InputArray mask) override
	{
		detectAsync(image, keypoints_, mask, Stream::Null());
		convert(keypoints_, keypoints);
	}

	void compute(InputArray image, std::vector<KeyPoint>& keypoints, OutputArray descriptors) override
	{
		describer_->compute(image, keypoints, descriptors);
	}

	void detectAndCompute(InputArray image, InputArray mask, std::vector<KeyPoint>& keypoints, OutputArray descriptors,
		bool useProvidedKeypoints) override
	{
		detectAndComputeAsync(image, mask, keypoints_, descriptors, useProvidedKeypoints, Stream::Null());
		convert(keypoints_, keypoints);
	}

	void detectAsync(InputArray image, OutputArray keypoints, InputArray mask, Stream& stream) override
	{
		detectAndComputeAsync(image, mask, keypoints, noArray(), false, stream);
	}

	void computeAsync(InputArray image, InputArray keypoints, OutputArray descriptors, Stream& stream) override
	{
		describer_->computeAsync(image, keypoints, descriptors, stream);
	}

	void detectAndComputeAsync(InputArray _image, InputArray _mask, OutputArray _keypoints, OutputArray _descriptors,
		bool useProvidedKeypoints, Stream& stream) override
	{
		CV_Assert(_image.type() == CV_8U);
		CV_Assert(!useProvidedKeypoints);

		const bool needDescriptors = _descriptors.needed();
		const cudaStream_t cuStream = StreamAccessor::getStream(stream);

        getInputMat(_image, image_, stream);
        calcImagePyramid(image_, imagePyr_, scales_, scaleFactor_, nlevels_, stream);

        if (!_mask.empty()) {
            getInputMat(_mask, mask_, stream);
            std::vector<float> __;
            calcImagePyramid(mask_, maskPyr_, __, scaleFactor_, nlevels_, stream);
        }

        calcNumFeaturesPerLevel(nfeatures_, scaleFactor_, nlevels_, nfeaturesPerLevel_);

        int nkeypoints = 0;
        maskPyr_.resize(nlevels_);
        kptsPyr_.resize(nlevels_);
        kptsBuf_.resize(nlevels_);
		const bool logStageFingerprint = shouldLogStageFingerprint();
        for (int s = firstLevel_; s < nlevels_; s++)
		{
			const GpuMat& image = imagePyr_[s];
			GpuMat& mask = maskPyr_[s];

			if (mask.size() != image.size())
				createMask(mask, image.size(), HALF_PATCH_SIZE, stream);

			const int maxpoints = cvRound(CORNER_DENSITY * image.size().area());
			GpuMat tmppoints = fastBuf_.createMat(4, maxpoints, CV_32F);
			GpuMat keypoints = kptsBuf_[s].createMat(ROWS_COUNT, maxpoints, CV_32F);
			GpuMat fullCaptureBuffer;
			if (deterministic_)
			{
				const auto imageArea = image.size().area();
				fullCaptureBuffer = fastFullCaptureBuf_.createMat(1, imageArea, CV_16SC2);
				if (static_cast<std::size_t>(imageArea) > fastFullCaptureHostCapacity_)
				{
					fastFullCaptureHost_.create(1, imageArea, CV_16SC2);
					fastFullCaptureHostCapacity_ = static_cast<std::size_t>(imageArea);
				}
			}

			const int bufferSize = radiusSuppressionBufferSize(image.size(), maxpoints);
			GpuMat d_buffer = suppBuf_.createMat(bufferSize, 1, CV_32S);

			calcKeypoints(image, mask, tmppoints, maxpoints,
				fastThreshold_, d_buffer, h_buffer_, cuStream, deterministic_, fullCaptureBuffer,
				fastFullCaptureHost_, diagnosticSensorId_.c_str(), diagnosticTimestamp_, diagnosticSlotIndex_, s);
			if (logStageFingerprint)
				logFeatureStageFingerprint(
					diagnosticSensorId_, diagnosticTimestamp_, diagnosticSlotIndex_,
					s, "calcKeypoints", tmppoints, false, stream);

			calcResponses(image, tmppoints, cuStream);
			if (logStageFingerprint)
				logFeatureStageFingerprint(
					diagnosticSensorId_, diagnosticTimestamp_, diagnosticSlotIndex_,
					s, "calcResponses", tmppoints, true, stream);

			radiusSuppression(tmppoints, keypoints, image.size(), nonmaxRadius_,
				d_buffer, h_buffer_, cuStream, deterministic_);
			if (logStageFingerprint)
				logFeatureStageFingerprint(
					diagnosticSensorId_, diagnosticTimestamp_, diagnosticSlotIndex_,
					s, "radiusSuppression", keypoints, true, stream);

			limitPoints(keypoints, nfeaturesPerLevel_[s], cuStream, deterministic_);
			if (logStageFingerprint)
				logFeatureStageFingerprint(
					diagnosticSensorId_, diagnosticTimestamp_, diagnosticSlotIndex_,
					s, "limitPoints", keypoints, true, stream);

			calcAngles(image, keypoints, cuStream,
				deterministic_ ? deterministicAngleQuantizationDeg() : 0.f);

			kptsPyr_[s] = keypoints;
			nkeypoints += keypoints.cols;
		}

		if (nkeypoints == 0)
		{
			_keypoints.release();
			if (needDescriptors)
				descriptors_.release();
			return;
		}

		getOutputMat(_keypoints, keypoints_, ROWS_COUNT, nkeypoints, CV_32F);

		if (needDescriptors)
		{
			getOutputMat(_descriptors, descriptors_, nkeypoints, descriptorSize(), descriptorType());
			blurPyr_.resize(nlevels_);
			descPyr_.resize(nlevels_);
		}

		int offset = 0;
		for (int s = firstLevel_; s < nlevels_; s++)
		{
			GpuMat& keypoints = kptsPyr_[s];
			const int npoints = keypoints.cols;
			if (npoints <= 0)
				continue;

			const Range dstRange(offset, offset + npoints);
			if (needDescriptors)
			{
				GpuMat descriptors = descriptors_.rowRange(dstRange);
				filter_->apply(imagePyr_[s], blurPyr_[s], stream);
				describer_->computeAsync(blurPyr_[s], keypoints, descriptors, stream);
			}

			// insert keypoints
			scalePoints(keypoints, scales_[s], s, cuStream);
			keypoints.copyTo(keypoints_.colRange(dstRange), stream);

			offset += npoints;
		}

		if (_keypoints.kind() == _InputArray::KindFlag::MAT)
			keypoints_.download(_keypoints, stream);

		if (needDescriptors && _descriptors.kind() == _InputArray::KindFlag::MAT)
			descriptors_.download(_descriptors, stream);
	}

	void detectAndComputeAsyncPerChannel(InputArray _image, InputArray _mask,
		std::vector<GpuMat>& keypoints, std::vector<GpuMat>& descriptors,
		bool useProvidedKeypoints, Stream& stream) override
	{
		CV_Assert(_image.depth() == CV_8U);
		CV_Assert(!useProvidedKeypoints);

		if (!_mask.empty())
		{
			CV_Assert(_mask.type() == CV_8U);
			CV_Assert(_mask.channels() == 1);
		}

		GpuMat image;
		getInputMat(_image, image, stream);
		const int channels = image.channels();
		CV_Assert(channels >= 1);

		std::vector<GpuMat> planes;
		if (channels == 1)
		{
			planes.emplace_back(image);
		}
		else
		{
			planes.resize(channels);
			cuda::split(image, planes, stream);
		}

		keypoints.resize(channels);
		descriptors.resize(channels);
		for (int c = 0; c < channels; ++c)
			detectAndComputeAsync(planes[c], _mask, keypoints[c], descriptors[c], false, stream);
	}

	void convert(InputArray src, CV_OUT std::vector<KeyPoint>& dst) override
	{
		Mat tmp;
		if (src.kind() == _InputArray::KindFlag::MAT)
			tmp = src.getMat();
		else if (src.kind() == _InputArray::KindFlag::CUDA_GPU_MAT)
			src.getGpuMat().download(tmp);

		const Vec2s* points = tmp.ptr<Vec2s>(LOCATION_ROW);
		const float* responses = tmp.ptr<float>(RESPONSE_ROW);
		const float* angles = tmp.ptr<float>(ANGLE_ROW);
		const int* octaves = tmp.ptr<int>(OCTAVE_ROW);
		const float* sizes = tmp.ptr<float>(SIZE_ROW);

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

	int descriptorSize() const { return describer_->descriptorSize(); }
	int descriptorType() const { return describer_->descriptorType(); }
	int defaultNorm() const { return describer_->defaultNorm(); }

	void setMaxFeatures(int maxFeatures) { nfeatures_ = maxFeatures; }
	int getMaxFeatures() const { return nfeatures_; }

	void setScaleFactor(float scaleFactor) { scaleFactor_ = scaleFactor; }
	float getScaleFactor() const { return scaleFactor_; }

	void setNLevels(int nlevels) { nlevels_ = nlevels; }
	int getNLevels() const { return nlevels_; }

	void setFirstLevel(int firstLevel) { firstLevel_ = firstLevel; }
	int getFirstLevel() const { return firstLevel_; }

	void setFastThreshold(int fastThreshold) { fastThreshold_ = fastThreshold; }
	int getFastThreshold() const { return fastThreshold_; }

	void setNonmaxRadius(int nonmaxRadius) { nonmaxRadius_ = nonmaxRadius; }
	int getNonmaxRadius() const { return nonmaxRadius_; }

	void setDescriptorType(DescriptorType descriptorType, const SphericalLensParams& params)
	{
			descriptorType_ = descriptorType;
			lensParams_ = params;
			describer_ = createDescriber(descriptorType, lensParams_);
	}

	SphericalLensParams getSphericalLensParams() const { return lensParams_; }
	DescriptorType getDescriptorType() const { return descriptorType_; }

	void setDeterministic(bool deterministic) override { deterministic_ = deterministic; }
	bool isDeterministic() const override { return deterministic_; }

	void setDiagnosticContext(const char* sensorId, std::uint64_t timestamp, int slotIndex) override
	{
		diagnosticSensorId_ = sensorId != nullptr ? sensorId : "";
		diagnosticTimestamp_ = timestamp;
		diagnosticSlotIndex_ = slotIndex;
	}

private:
	bool shouldLogStageFingerprint()
	{
		if (diagnosticSensorId_.empty())
			return false;

		if (diagnosticTimestamp_ != diagnosticLastTimestamp_)
		{
			diagnosticLastTimestamp_ = diagnosticTimestamp_;
			if (diagnosticLoggedFrameCount_ >= featureStageFingerprintFrameLimit())
				return false;
			diagnosticLoggedFrameCount_ += 1;
		}
		return true;
	}

	int nfeatures_;
	float scaleFactor_;
	int nlevels_;
	int firstLevel_;
	int fastThreshold_;
	int nonmaxRadius_;
        DescriptorType descriptorType_;
        SphericalLensParams lensParams_{};
	bool deterministic_ = false;
	std::string diagnosticSensorId_;
	std::uint64_t diagnosticTimestamp_ = 0;
	std::uint64_t diagnosticLastTimestamp_ = 0;
	int diagnosticSlotIndex_ = -1;
	int diagnosticLoggedFrameCount_ = 0;

    GpuMat image_, mask_, keypoints_, descriptors_;
    std::vector<GpuMat> imagePyr_, maskPyr_, kptsPyr_, blurPyr_, descPyr_;

	DeviceBuffer fastBuf_, fastFullCaptureBuf_, suppBuf_;
    std::vector<DeviceBuffer> kptsBuf_;

    HostMem h_buffer_;
	HostMem fastFullCaptureHost_;
	std::size_t fastFullCaptureHostCapacity_ = 0U;
    int* h_count_;

    std::vector<float> scales_;
    std::vector<int> nfeaturesPerLevel_;
	Ptr<EfficientDescriptorsAsync> describer_;
	Ptr<cuda::Filter> filter_;
};

Ptr<EfficientFeatures> EfficientFeatures::create(int nfeatures, float scaleFactor, int nlevels,
	int firstLevel, int fastThreshold, int nonmaxRadius, DescriptorType dtype)
{
	return makePtr<EfficientFeaturesImpl>(nfeatures, scaleFactor, nlevels,
		firstLevel, fastThreshold, nonmaxRadius, dtype);
}

EfficientFeatures::~EfficientFeatures()
{
}

} // namespace cuda
} // namespace cv
