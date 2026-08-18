#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/features2d.hpp>

#include "../modules/cuda_efficient_features/src/deterministic_fast_precap.h"
#include <cuda_efficient_features.h>

namespace {

using cv::Vec2s;
using cv::cuda::fast_precap_detail::SelectionAudit;
using cv::cuda::fast_precap_detail::SelectionStatus;

/**
 * @brief production と独立した SplitMix64 CPU reference を計算する。
 * @param point `(x,y)` の候補座標。
 * @return fixed policy の 64 bit rank。
 * @example `(2,1)` は `0x9a9f5e0655f6a5b3` を返す。
 */
std::uint64_t referenceSplitMix64(const Vec2s &point) {
  const auto x = static_cast<std::uint16_t>(point[0]);
  const auto y = static_cast<std::uint16_t>(point[1]);
  std::uint64_t value = (static_cast<std::uint64_t>(y) << 16U) | x;
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

/**
 * @brief CPU reference で全候補を rank sort し先頭 cap を返す。
 * @param points 順序を任意にした unique 候補列。
 * @param cap 選択数。
 * @return fixed rank 順の先頭 cap 座標。
 * @example 3784候補とcap3608から3608候補を返す。
 */
std::vector<Vec2s> referenceSelection(std::vector<Vec2s> points,
                                      const std::size_t cap) {
  std::sort(points.begin(), points.end(),
            [](const Vec2s &lhs, const Vec2s &rhs) {
              const auto lhsMix = referenceSplitMix64(lhs);
              const auto rhsMix = referenceSplitMix64(rhs);
              if (lhsMix != rhsMix)
                return lhsMix < rhsMix;
              if (lhs[1] != rhs[1])
                return lhs[1] < rhs[1];
              return lhs[0] < rhs[0];
            });
  points.resize(cap);
  return points;
}

/**
 * @brief unit fixture 用の重複しない row-major 座標を作る。
 * @param count 生成点数。
 * @param width fixture の行幅。
 * @return `count` 個の unique `(x,y)`。
 * @example count=10,width=4なら `(0,0)..(1,2)` を返す。
 */
std::vector<Vec2s> makeUniqueGrid(const std::size_t count, const int width) {
  std::vector<Vec2s> points;
  points.reserve(count);
  for (std::size_t index = 0; index < count; ++index)
    points.emplace_back(static_cast<short>(index % width),
                        static_cast<short>(index / width));
  return points;
}

/**
 * @brief production canonicalizer を固定256x256 fixtureへ適用する。
 * @param points 入出力候補列。
 * @param initialCount 初回候補数。
 * @param recaptureCount 再取得候補数。
 * @param storedCount buffer書込み数。
 * @param capacity full capture容量。
 * @param cap 既存area cap。
 * @param audit 任意の保存則出力。
 * @return canonicalizer status。
 * @example 9点/cap8ならSELECTEDを返し先頭8点を固定する。
 */
SelectionStatus select(std::vector<Vec2s> &points,
                       const std::size_t initialCount,
                       const std::size_t recaptureCount,
                       const std::size_t storedCount,
                       const std::size_t capacity, const std::size_t cap,
                       SelectionAudit *audit = nullptr) {
  return cv::cuda::fast_precap_detail::canonicalizeFullFastCapture(
      points.data(), initialCount, recaptureCount, storedCount, capacity, cap,
      256, 256, audit);
}

/**
 * @brief 候補列を8x8 occupancy fixtureへ集計する。
 * @param points 集計する候補列。
 * @param width 画像幅。
 * @param height 画像高さ。
 * @return row-major 64 cell count。
 * @example 64x64画像では各cellが8x8 pixelを覆う。
 */
std::array<int, 64> occupancy8x8(const std::vector<Vec2s> &points,
                                 const int width, const int height) {
  std::array<int, 64> counts{};
  for (const auto &point : points) {
    const auto cellX = std::min(static_cast<int>(point[0]) * 8 / width, 7);
    const auto cellY = std::min(static_cast<int>(point[1]) * 8 / height, 7);
    counts[static_cast<std::size_t>(cellY * 8 + cellX)] += 1;
  }
  return counts;
}

TEST(FastPrecap, SplitMix64KnownAnswersAndPackingAreFixed) {
  using namespace cv::cuda::fast_precap_detail;
  EXPECT_EQ(splitMix64(0ULL), 0xe220a8397b1dcdafULL);
  EXPECT_EQ(splitMix64(1ULL), 0x910a2dec89025cc1ULL);
  EXPECT_EQ(packU16Yx(Vec2s(2, 1)), 0x0000000000010002ULL);
  EXPECT_EQ(splitMix64(packU16Yx(Vec2s(2, 1))), 0x9a9f5e0655f6a5b3ULL);
  EXPECT_STREQ(SPLITMIX64_PRECAP_POLICY, "splitmix64_pack_u16_yx_v1");
  EXPECT_EQ(SPLITMIX64_PRECAP_SEED, 0ULL);
}

TEST(FastPrecap, ExistingCvRoundAreaCapAndOverflowBoundaryStayExact) {
  EXPECT_EQ(cvRound(0.1 * 11), 1);
  EXPECT_EQ(cvRound(0.1 * 15), 2);
  EXPECT_EQ(cvRound(0.1 * 25), 2);
  EXPECT_FALSE(
      cv::cuda::fast_precap_detail::requiresCanonicalPrecap(3607, 3608));
  EXPECT_FALSE(
      cv::cuda::fast_precap_detail::requiresCanonicalPrecap(3608, 3608));
  EXPECT_TRUE(
      cv::cuda::fast_precap_detail::requiresCanonicalPrecap(3609, 3608));
}

TEST(FastPrecap, CapMinusOneAndCapAreNotCanonicalizedOrReordered) {
  for (const std::size_t count : {7U, 8U}) {
    auto points = makeUniqueGrid(count, 16);
    std::reverse(points.begin(), points.end());
    const auto before = points;
    EXPECT_EQ(select(points, count, count, count, count, 8),
              SelectionStatus::NOT_REQUIRED);
    EXPECT_EQ(points, before);
  }
}

TEST(FastPrecap, CapPlusOneMatchesIndependentReferenceForPermutations) {
  auto original = std::vector<Vec2s>{
      Vec2s(3, 1), Vec2s(9, 7), Vec2s(2, 8), Vec2s(6, 4), Vec2s(1, 5),
      Vec2s(8, 2), Vec2s(4, 9), Vec2s(7, 3), Vec2s(5, 6),
  };
  const auto expected = referenceSelection(original, 8);
  for (int permutation = 0; permutation < 6; ++permutation) {
    auto points = original;
    std::mt19937 generator(static_cast<std::uint32_t>(permutation + 1));
    std::shuffle(points.begin(), points.end(), generator);
    SelectionAudit audit;
    ASSERT_EQ(select(points, 9, 9, 9, 9, 8, &audit), SelectionStatus::SELECTED);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), points.begin()));
    EXPECT_EQ(audit.selectedCount, 8U);
    EXPECT_EQ(audit.droppedCount, 1U);
  }
}

TEST(FastPrecap, Overflow3784To3608IsPermutationInvariantAndConservesCounts) {
  const auto original = makeUniqueGrid(3784, 64);
  const auto expected = referenceSelection(original, 3608);
  std::uint64_t stableDigest = 0U;
  for (int permutation = 0; permutation < 4; ++permutation) {
    auto points = original;
    std::mt19937 generator(static_cast<std::uint32_t>(0x713U + permutation));
    std::shuffle(points.begin(), points.end(), generator);
    SelectionAudit audit;
    ASSERT_EQ(select(points, 3784, 3784, 3784, 4096, 3608, &audit),
              SelectionStatus::SELECTED);
    EXPECT_TRUE(std::equal(expected.begin(), expected.end(), points.begin()));
    EXPECT_EQ(audit.initialCount, audit.selectedCount + audit.droppedCount);
    EXPECT_EQ(audit.selectedCount, 3608U);
    EXPECT_EQ(audit.droppedCount, 176U);
    if (permutation == 0)
      stableDigest = audit.selectedOrderDigest;
    else
      EXPECT_EQ(audit.selectedOrderDigest, stableDigest);
  }
}

TEST(FastPrecap, FullCaptureContractViolationsFailClosedWithoutFallback) {
  auto points = makeUniqueGrid(10, 16);
  EXPECT_EQ(select(points, 10, 9, 9, 10, 8),
            SelectionStatus::RECAPTURE_COUNT_MISMATCH);
  EXPECT_EQ(select(points, 10, 10, 9, 10, 8),
            SelectionStatus::STORED_COUNT_MISMATCH);
  EXPECT_EQ(select(points, 10, 10, 10, 9, 8),
            SelectionStatus::RECAPTURE_CAPACITY_EXCEEDED);
  points[3] = Vec2s(-1, 2);
  EXPECT_EQ(select(points, 10, 10, 10, 10, 8),
            SelectionStatus::INVALID_COORDINATE);
  points = makeUniqueGrid(10, 16);
  points[7] = points[2];
  EXPECT_EQ(select(points, 10, 10, 10, 10, 8),
            SelectionStatus::DUPLICATE_COORDINATE);
}

TEST(FastPrecap, RawAndSelectedOccupancyCanBeComparedAtFourAndEightCells) {
  auto points = makeUniqueGrid(3784, 64);
  auto rowMajorPrefix = points;
  rowMajorPrefix.resize(3608);
  const auto rawOccupancy = occupancy8x8(points, 64, 64);
  const auto prefixOccupancy = occupancy8x8(rowMajorPrefix, 64, 64);
  ASSERT_EQ(select(points, 3784, 3784, 3784, 4096, 3608),
            SelectionStatus::SELECTED);
  points.resize(3608);
  const auto selectedOccupancy = occupancy8x8(points, 64, 64);
  EXPECT_EQ(std::accumulate(rawOccupancy.begin(), rawOccupancy.end(), 0), 3784);
  EXPECT_EQ(
      std::accumulate(selectedOccupancy.begin(), selectedOccupancy.end(), 0),
      3608);
  EXPECT_GT(std::count_if(selectedOccupancy.begin(), selectedOccupancy.end(),
                          [](const int count) { return count > 0; }),
            std::count_if(prefixOccupancy.begin(), prefixOccupancy.end(),
                          [](const int count) { return count > 0; }));
  const auto occupancyRange = [](const auto &occupancy) {
    const auto bounds = std::minmax_element(occupancy.begin(), occupancy.end());
    return *bounds.second - *bounds.first;
  };
  EXPECT_LT(occupancyRange(selectedOccupancy), occupancyRange(prefixOccupancy));

  std::array<int, 16> selectedOccupancy4x4{};
  std::array<int, 16> prefixOccupancy4x4{};
  for (int blockY = 0; blockY < 4; ++blockY) {
    for (int blockX = 0; blockX < 4; ++blockX) {
      int raw4x4 = 0;
      int selected4x4 = 0;
      int prefix4x4 = 0;
      for (int offsetY = 0; offsetY < 2; ++offsetY)
        for (int offsetX = 0; offsetX < 2; ++offsetX) {
          const auto index = static_cast<std::size_t>(
              (blockY * 2 + offsetY) * 8 + blockX * 2 + offsetX);
          raw4x4 += rawOccupancy[index];
          selected4x4 += selectedOccupancy[index];
          prefix4x4 += prefixOccupancy[index];
        }
      EXPECT_LE(selected4x4, raw4x4);
      const auto blockIndex = static_cast<std::size_t>(blockY * 4 + blockX);
      selectedOccupancy4x4[blockIndex] = selected4x4;
      prefixOccupancy4x4[blockIndex] = prefix4x4;
    }
  }
  EXPECT_LT(occupancyRange(selectedOccupancy4x4),
            occupancyRange(prefixOccupancy4x4));
}

TEST(FastPrecap,
     FourCudaStreamsKeepPayloadsIsolatedAcrossMultiLevelBufferReuse) {
  if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
    GTEST_SKIP() << "CUDA device is required";

  // 6 levels目でも既存15px border ROIが残る最小側の画像を使う。
  cv::Mat image(384, 384, CV_8U);
  cv::RNG random(0x51f7a23U);
  random.fill(image, cv::RNG::UNIFORM, 0, 256);
  std::vector<cv::KeyPoint> cpuFast;
  cv::FAST(image, cpuFast, 7, false);
  ASSERT_GT(cpuFast.size(),
            static_cast<std::size_t>(cvRound(0.1 * image.size().area())));

  std::array<cv::Ptr<cv::cuda::EfficientFeatures>, 4> detectors;
  std::array<cv::cuda::Stream, 4> streams;
  std::array<cv::cuda::GpuMat, 4> images;
  std::array<cv::cuda::GpuMat, 4> keypoints;
  std::array<cv::cuda::GpuMat, 4> descriptors;
  std::array<cv::Mat, 4> firstKeypoints;
  std::array<cv::Mat, 4> firstDescriptors;
  for (std::size_t slot = 0; slot < detectors.size(); ++slot) {
    detectors[slot] = cv::cuda::EfficientFeatures::create(
        1600, 1.5f, 6, 0, 7, 8,
        cv::cuda::EfficientFeatures::DescriptorType::ORB);
    detectors[slot]->setDeterministic(true);
    images[slot].upload(image, streams[slot]);
    detectors[slot]->detectAndComputeAsync(images[slot], cv::noArray(),
                                           keypoints[slot], descriptors[slot],
                                           false, streams[slot]);
  }
  for (auto &stream : streams)
    stream.waitForCompletion();
  for (std::size_t slot = 0; slot < detectors.size(); ++slot) {
    keypoints[slot].download(firstKeypoints[slot]);
    descriptors[slot].download(firstDescriptors[slot]);
    ASSERT_FALSE(firstKeypoints[slot].empty());
    ASSERT_EQ(firstKeypoints[slot].cols, firstDescriptors[slot].rows);
    if (slot > 0U) {
      EXPECT_EQ(cv::norm(firstKeypoints[0], firstKeypoints[slot], cv::NORM_INF),
                0.0);
      EXPECT_EQ(
          cv::norm(firstDescriptors[0], firstDescriptors[slot], cv::NORM_INF),
          0.0);
    }
  }

  for (std::size_t reverse = detectors.size(); reverse > 0U; --reverse) {
    const auto slot = reverse - 1U;
    detectors[slot]->detectAndComputeAsync(images[slot], cv::noArray(),
                                           keypoints[slot], descriptors[slot],
                                           false, streams[slot]);
  }
  for (auto &stream : streams)
    stream.waitForCompletion();
  for (std::size_t slot = 0; slot < detectors.size(); ++slot) {
    cv::Mat repeatedKeypoints;
    cv::Mat repeatedDescriptors;
    keypoints[slot].download(repeatedKeypoints);
    descriptors[slot].download(repeatedDescriptors);
    EXPECT_EQ(cv::norm(firstKeypoints[slot], repeatedKeypoints, cv::NORM_INF),
              0.0);
    EXPECT_EQ(
        cv::norm(firstDescriptors[slot], repeatedDescriptors, cv::NORM_INF),
        0.0);
  }
}

TEST(FastPrecap, DeterministicFalseKeepsLegacyCappedPathAndSkipsPrecap) {
  if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
    GTEST_SKIP() << "CUDA device is required";

  cv::Mat image(384, 384, CV_8U);
  cv::RNG random(0x771eU);
  random.fill(image, cv::RNG::UNIFORM, 0, 256);
  auto detector = cv::cuda::EfficientFeatures::create(1600, 1.5f, 1, 0, 7, 8);
  detector->setDeterministic(false);
  detector->setDiagnosticContext("legacy_pre_cap_test", 1U, 0);
  ASSERT_EQ(
      setenv("TRIORB_CUDA_FEATURE_KEYPOINT_FINGERPRINT_MAX_FRAMES", "1", 1), 0);
  testing::internal::CaptureStdout();
  std::vector<cv::KeyPoint> keypoints;
  detector->detect(image, keypoints);
  const auto audit = testing::internal::GetCapturedStdout();
  unsetenv("TRIORB_CUDA_FEATURE_KEYPOINT_FINGERPRINT_MAX_FRAMES");
  EXPECT_NE(audit.find("deterministic=0"), std::string::npos);
  EXPECT_NE(audit.find("precap_policy=legacy_atomic_prefix"),
            std::string::npos);
  EXPECT_NE(audit.find("precap_full_capture=0"), std::string::npos);
  EXPECT_NE(audit.find("precap_applied=0"), std::string::npos);
}

} // namespace
