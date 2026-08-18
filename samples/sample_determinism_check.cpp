/*
Copyright 2026 TriOrb Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Unit-level determinism verification for cv::cuda::EfficientFeatures.
//
// The CUDA feature extractor contains two non-deterministic stages by default:
//   1. radiusSuppressionKernel packs surviving points into the output array
//      using atomicAdd(count, 1), so the output order depends on GPU thread
//      scheduling.
//   2. limitPoints uses thrust::sort_by_key (not stable), so when multiple
//      keypoints tie on Harris response the subset kept after truncation to
//      nfeatures is not reproducible across runs.
//
// This sample runs the extractor N times on the same input image and reports
// whether the resulting keypoint / descriptor arrays are bit-identical across
// all runs. It is intended as a quick smoke check when toggling the
// setDeterministic flag (or the TRIORB_CUDA_FEATURES_DETERMINISTIC env var
// upstream).
//
// Usage:
//   sample_determinism_check --input-image=PATH
//                            [--iterations=5]
//                            [--deterministic=1]
//                            [--descriptor-type=2]
//                            [--max-keypoints=10000]
//                            [--fast-threshold=20]
//
// Descriptor type integer mirrors cv::cuda::EfficientFeatures::DescriptorType.
// Exit codes: 0 = deterministic, 1 = differed, 2 = usage / I/O error.

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cuda_efficient_features.h>

namespace
{

struct KeypointSnapshot
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

bool extractOnce(
    const cv::Mat & gray,
    int maxKeypoints,
    int fastThreshold,
    int nonmaxRadius,
    cv::cuda::EfficientFeatures::DescriptorType descType,
    bool deterministic,
    KeypointSnapshot & out)
{
    auto feature = cv::cuda::EfficientFeatures::create(maxKeypoints);
    if (!feature)
    {
        std::cerr << "Failed to create EfficientFeatures extractor." << std::endl;
        return false;
    }
    feature->setFastThreshold(fastThreshold);
    feature->setNonmaxRadius(nonmaxRadius);
    feature->setDescriptorType(descType);
    feature->setDeterministic(deterministic);

    feature->detectAndCompute(gray, cv::noArray(), out.keypoints, out.descriptors);
    return true;
}

bool keypointsEqual(const std::vector<cv::KeyPoint> & a, const std::vector<cv::KeyPoint> & b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const auto & ka = a[i];
        const auto & kb = b[i];
        if (ka.pt.x != kb.pt.x) return false;
        if (ka.pt.y != kb.pt.y) return false;
        if (ka.response != kb.response) return false;
        if (ka.angle != kb.angle) return false;
        if (ka.octave != kb.octave) return false;
        if (ka.size != kb.size) return false;
    }
    return true;
}

bool descriptorsEqual(const cv::Mat & a, const cv::Mat & b)
{
    if (a.rows != b.rows || a.cols != b.cols || a.type() != b.type())
    {
        return false;
    }
    if (a.empty())
    {
        return true;
    }
    for (int row = 0; row < a.rows; ++row)
    {
        const std::size_t row_bytes = static_cast<std::size_t>(a.cols) * a.elemSize();
        if (std::memcmp(a.ptr(row), b.ptr(row), row_bytes) != 0)
        {
            return false;
        }
    }
    return true;
}

std::size_t countKeypointSetDifference(
    const std::vector<cv::KeyPoint> & a,
    const std::vector<cv::KeyPoint> & b)
{
    std::size_t mismatch = 0;
    for (const auto & ka : a)
    {
        bool found = false;
        for (const auto & kb : b)
        {
            if (ka.pt.x == kb.pt.x && ka.pt.y == kb.pt.y)
            {
                found = true;
                break;
            }
        }
        if (!found)
        {
            ++mismatch;
        }
    }
    return mismatch;
}

}  // namespace

static const std::string keys =
    "{ input-image     |       | input image path. }"
    "{ iterations      |     5 | number of extraction iterations. }"
    "{ deterministic   |     1 | 1 = enable deterministic ordering, 0 = default. }"
    "{ max-keypoints   | 10000 | maximum number of keypoints. }"
    "{ fast-threshold  |    20 | FAST threshold. }"
    "{ nonmax-radius   |    15 | radius of non-maximum suppression. }"
    "{ descriptor-type |     2 | DescriptorType enum value (default HASH_SIFT_256). }"
    "{ help h          |       | print help message. }";

int main(int argc, char * argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    const std::string filename = parser.get<std::string>("input-image");
    const int iterations = std::max(2, parser.get<int>("iterations"));
    const bool deterministic = parser.get<int>("deterministic") != 0;
    const int maxKeypoints = parser.get<int>("max-keypoints");
    const int fastThreshold = parser.get<int>("fast-threshold");
    const int nonmaxRadius = parser.get<int>("nonmax-radius");
    const int descTypeInt = parser.get<int>("descriptor-type");

    if (!parser.check())
    {
        parser.printErrors();
        parser.printMessage();
        return 2;
    }
    if (filename.empty())
    {
        std::cerr << "--input-image is required." << std::endl;
        parser.printMessage();
        return 2;
    }

    cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    if (image.empty())
    {
        std::cerr << "imread failed: " << filename << std::endl;
        return 2;
    }
    cv::Mat gray;
    if (image.channels() == 1)
    {
        gray = image;
    }
    else if (image.channels() == 3)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels() == 4)
    {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        std::cerr << "Unsupported channel count: " << image.channels() << std::endl;
        return 2;
    }

    const auto resolvedType = static_cast<cv::cuda::EfficientFeatures::DescriptorType>(descTypeInt);

    std::cout << "=== determinism check ===" << std::endl;
    std::cout << "image           : " << filename << " (" << image.size() << ")" << std::endl;
    std::cout << "iterations      : " << iterations << std::endl;
    std::cout << "deterministic   : " << (deterministic ? "ON" : "OFF") << std::endl;
    std::cout << "descriptor type : " << descTypeInt << std::endl;
    std::cout << "max keypoints   : " << maxKeypoints << std::endl;
    std::cout << std::endl;

    KeypointSnapshot reference;
    if (!extractOnce(gray, maxKeypoints, fastThreshold, nonmaxRadius, resolvedType, deterministic, reference))
    {
        return 2;
    }
    std::cout << "iter 0: " << reference.keypoints.size() << " keypoints (reference)" << std::endl;

    bool all_equal = true;
    for (int iter = 1; iter < iterations; ++iter)
    {
        KeypointSnapshot cur;
        if (!extractOnce(gray, maxKeypoints, fastThreshold, nonmaxRadius, resolvedType, deterministic, cur))
        {
            return 2;
        }
        const bool kpEq = keypointsEqual(reference.keypoints, cur.keypoints);
        const bool dscEq = descriptorsEqual(reference.descriptors, cur.descriptors);
        const std::size_t set_diff = kpEq ? 0 : countKeypointSetDifference(reference.keypoints, cur.keypoints);
        std::cout << "iter " << iter
                  << ": " << cur.keypoints.size() << " keypoints"
                  << ", kp_equal=" << (kpEq ? "true" : "false")
                  << ", desc_equal=" << (dscEq ? "true" : "false")
                  << ", set_diff=" << set_diff
                  << std::endl;
        if (!kpEq || !dscEq)
        {
            all_equal = false;
        }
    }

    std::cout << std::endl;
    if (all_equal)
    {
        std::cout << "RESULT: DETERMINISTIC (all " << iterations << " iterations identical)" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "RESULT: NON-DETERMINISTIC" << std::endl;
        return 1;
    }
}
