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

#ifndef __DETERMINISTIC_FAST_PRECAP_H__
#define __DETERMINISTIC_FAST_PRECAP_H__

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include <opencv2/core.hpp>

namespace cv {
namespace cuda {
namespace fast_precap_detail {

inline constexpr const char *SPLITMIX64_PRECAP_POLICY =
    "splitmix64_pack_u16_yx_v1";
inline constexpr std::uint64_t SPLITMIX64_PRECAP_SEED = 0ULL;

/**
 * @brief SplitMix64 v1 の固定 mixing を 64 bit 値へ適用する。
 * @param value `(y << 16) | x` で pack した座標。例: `(1,2)` は `0x00010002`。
 * @return wrapping 演算後の 64 bit rank。例: 同じ座標には常に同じ値を返す。
 * @example 入力: `value=0`。出力: `0xe220a8397b1dcdaf`。
 */
inline std::uint64_t splitMix64(const std::uint64_t value) {
  std::uint64_t mixed = value + 0x9e3779b97f4a7c15ULL;
  mixed = (mixed ^ (mixed >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  mixed = (mixed ^ (mixed >> 27U)) * 0x94d049bb133111ebULL;
  return mixed ^ (mixed >> 31U);
}

/**
 * @brief FAST 座標を policy 固定の `u16(y),u16(x)` 順で pack する。
 * @param point `point[0]=x`, `point[1]=y` の FAST 座標。例: `(2,1)`。
 * @return `(u64(y)<<16)|u64(x)`。例: `(2,1)` は `0x00010002`。
 * @example 入力: `Vec2s(2,1)`。出力: `65538`。
 */
inline std::uint64_t packU16Yx(const Vec2s &point) {
  const auto x = static_cast<std::uint16_t>(point[0]);
  const auto y = static_cast<std::uint16_t>(point[1]);
  return (static_cast<std::uint64_t>(y) << 16U) | static_cast<std::uint64_t>(x);
}

/**
 * @brief FAST pre-cap の total rank `(SplitMix64(pack),y,x)` を比較する。
 * @param lhs 左辺の `(x,y)` 座標。
 * @param rhs 右辺の `(x,y)` 座標。
 * @return 左辺が rank 昇順で先なら true。
 * @example 入力順を反転しても、この比較で sort した結果は同一になる。
 */
inline bool splitMix64RankLess(const Vec2s &lhs, const Vec2s &rhs) {
  const auto lhsMix = splitMix64(packU16Yx(lhs));
  const auto rhsMix = splitMix64(packU16Yx(rhs));
  if (lhsMix != rhsMix)
    return lhsMix < rhsMix;
  if (lhs[1] != rhs[1])
    return lhs[1] < rhs[1];
  return lhs[0] < rhs[0];
}

enum class SelectionStatus {
  SELECTED,
  NOT_REQUIRED,
  INVALID_ARGUMENT,
  RECAPTURE_COUNT_MISMATCH,
  STORED_COUNT_MISMATCH,
  RECAPTURE_CAPACITY_EXCEEDED,
  INVALID_COORDINATE,
  DUPLICATE_COORDINATE,
};

struct SelectionAudit {
  std::size_t initialCount = 0;
  std::size_t recaptureCapacity = 0;
  std::size_t recaptureCount = 0;
  std::size_t storedCount = 0;
  std::size_t selectedCount = 0;
  std::size_t droppedCount = 0;
  std::uint64_t selectedOrderDigest = 0xcbf29ce484222325ULL;
};

/**
 * @brief pre-cap canonicalizer を呼ぶ必要がある overflow かを判定する。
 * @param rawCount 初回 capped kernel の全候補 counter。例: 3609。
 * @param areaCap 呼出側が既存 `cvRound(0.1*image.area())` で得た cap。例:
 * 3608。
 * @return `rawCount > areaCap` のときだけ true。
 * @example `3607/3608` と `3608/3608` は false、`3609/3608` は true。
 */
inline bool requiresCanonicalPrecap(const std::size_t rawCount,
                                    const std::size_t areaCap) {
  return rawCount > areaCap;
}

/**
 * @brief full recapture 候補を固定 SplitMix64 rank で in-place canonicalize
 * する。
 * @param points recapture 済み `(x,y)` 配列。成功時は全件が total rank
 * 順になる。
 * @param initialCount 初回 capped kernel が数えた全候補数。
 * @param recaptureCount full-capacity kernel が数えた全候補数。
 * @param storedCount full-capacity buffer へ実際に書けた候補数。
 * @param recaptureCapacity full-capacity device/host buffer の要素数。
 * @param areaCap 既存 `cvRound(0.1*image.area())` の値。
 * @param imageRows 入力 pyramid image の高さ。
 * @param imageCols 入力 pyramid image の幅。
 * @param audit 成功・失敗時の count 保存則を受け取る任意 pointer。
 * @return overflow 選択成功または固定した fail-close status。
 * @example 3609 候補、cap 3608 なら rank 先頭 3608 点を `points[0..3608)`
 * に置く。
 */
inline SelectionStatus canonicalizeFullFastCapture(
    Vec2s *points, const std::size_t initialCount,
    const std::size_t recaptureCount, const std::size_t storedCount,
    const std::size_t recaptureCapacity, const std::size_t areaCap,
    const int imageRows, const int imageCols, SelectionAudit *audit) {
  SelectionAudit localAudit;
  localAudit.initialCount = initialCount;
  localAudit.recaptureCapacity = recaptureCapacity;
  localAudit.recaptureCount = recaptureCount;
  localAudit.storedCount = storedCount;
  if (audit != nullptr)
    *audit = localAudit;

  if (!requiresCanonicalPrecap(initialCount, areaCap))
    return SelectionStatus::NOT_REQUIRED;
  if (points == nullptr || areaCap == 0U || imageRows <= 0 || imageCols <= 0)
    return SelectionStatus::INVALID_ARGUMENT;
  if (recaptureCount != initialCount)
    return SelectionStatus::RECAPTURE_COUNT_MISMATCH;
  if (storedCount != recaptureCount)
    return SelectionStatus::STORED_COUNT_MISMATCH;
  if (recaptureCount > recaptureCapacity)
    return SelectionStatus::RECAPTURE_CAPACITY_EXCEEDED;

  for (std::size_t index = 0; index < recaptureCount; ++index) {
    const auto x = static_cast<int>(points[index][0]);
    const auto y = static_cast<int>(points[index][1]);
    if (x < 0 || y < 0 || x >= imageCols || y >= imageRows)
      return SelectionStatus::INVALID_COORDINATE;
  }

  std::sort(points, points + recaptureCount, splitMix64RankLess);
  for (std::size_t index = 1; index < recaptureCount; ++index) {
    if (points[index] == points[index - 1U])
      return SelectionStatus::DUPLICATE_COORDINATE;
  }

  localAudit.selectedCount = std::min(initialCount, areaCap);
  localAudit.droppedCount = initialCount - localAudit.selectedCount;
  for (std::size_t index = 0; index < localAudit.selectedCount; ++index) {
    localAudit.selectedOrderDigest ^= packU16Yx(points[index]);
    localAudit.selectedOrderDigest *= 0x00000100000001b3ULL;
  }
  if (audit != nullptr)
    *audit = localAudit;
  return SelectionStatus::SELECTED;
}

/**
 * @brief fail-close status を audit 用の固定文字列へ変換する。
 * @param status canonicalizer の戻り値。
 * @return schema 安定な ASCII reason。
 * @example `STORED_COUNT_MISMATCH` は `stored_count_mismatch`。
 */
inline const char *selectionStatusName(const SelectionStatus status) {
  switch (status) {
  case SelectionStatus::SELECTED:
    return "selected";
  case SelectionStatus::NOT_REQUIRED:
    return "not_required";
  case SelectionStatus::INVALID_ARGUMENT:
    return "invalid_argument";
  case SelectionStatus::RECAPTURE_COUNT_MISMATCH:
    return "recapture_count_mismatch";
  case SelectionStatus::STORED_COUNT_MISMATCH:
    return "stored_count_mismatch";
  case SelectionStatus::RECAPTURE_CAPACITY_EXCEEDED:
    return "recapture_capacity_exceeded";
  case SelectionStatus::INVALID_COORDINATE:
    return "invalid_coordinate";
  case SelectionStatus::DUPLICATE_COORDINATE:
    return "duplicate_coordinate";
  }
  return "unknown";
}

} // namespace fast_precap_detail
} // namespace cuda
} // namespace cv

#endif // !__DETERMINISTIC_FAST_PRECAP_H__
