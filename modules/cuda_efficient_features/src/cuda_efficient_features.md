# cuda_efficient_features.cpp

## 目的

`cuda_efficient_features` は GPU feature extraction の実装を提供する module です。同梱コードまたは参照実装の役割を、このリポジトリの文脈で素早く把握できるようにします。

## 対象範囲

- 対象 source: `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_efficient_features.cpp`
- 判定条件: 505 行のため sidecar 文書を維持対象にする
- 主な定義: `EfficientFeaturesImpl`、`calcKeypoints`、`radiusSuppressionBufferSize`、`radiusSuppression`、`limitPoints`、`calcResponses`、`calcAngles`、`scalePoints`
- 主な依存: `opencv2/cudaarithm.hpp`、`opencv2/cudafilters.hpp`、`opencv2/cudaimgproc.hpp`、`opencv2/cudawarping.hpp`、`opencv2/core/cuda_stream_accessor.hpp`、`iostream`、`cuda_efficient_features.h`

## 現状

### 主な構成
- `EfficientFeaturesImpl` がこの module の主要な構成要素になっている
- `calcKeypoints` がこの module の主要な構成要素になっている
- `radiusSuppressionBufferSize` がこの module の主要な構成要素になっている
- `radiusSuppression` がこの module の主要な構成要素になっている
- `limitPoints` がこの module の主要な構成要素になっている
- `calcResponses` がこの module の主要な構成要素になっている

### 連携境界
- `opencv2/cudafilters.hpp` と連携しながら責務を完結させる
- `opencv2/cudaarithm.hpp` と連携しながら責務を完結させる
- `opencv2/cudaimgproc.hpp` と連携しながら責務を完結させる
- `opencv2/cudawarping.hpp` と連携しながら責務を完結させる
- `opencv2/core/cuda_stream_accessor.hpp` と連携しながら責務を完結させる
- `iostream` と連携しながら責務を完結させる
- `cuda_efficient_features.h` と連携しながら責務を完結させる

## 実装上の判断

- 同梱コードは upstream や参照実装としての責務を尊重し、この文書ではこのリポジトリから見た役割に絞って説明する。
- project 固有の判断は wrapper や利用側へ寄せ、この file 自体の変更理由を追いやすくする。
- 長大 file でも source 本体は read-only 前提で扱い、補足説明は sidecar 文書へ追加する。
- OpenCV 4.10 系では multi-channel 分解に使う `cv::cuda::split()` が `cudaarithm` component へ属するため、この module も `cudaarithm` を build 依存へ含める。
- Cycle 27 の `[cuda_feature_stage_fingerprint]` は read-only 診断であり、各 stage 直後の `GpuMat` を
  CPU へ download して count / 座標 fingerprint / response 分布を記録する。同期と download は診断 run の
  観測コストとして許容し、抽出結果の並び替え、しきい値、feature cap は変更しない。
- stage fingerprint は stage ごとの download / stream 同期を伴うため既定無効である。
  `TRIORB_CUDA_FEATURE_STAGE_FINGERPRINT_MAX_FRAMES=N` を明示した診断 run だけで有効化し、
  process-wide に `sensor_id` ごとの先頭 N timestamp に制限する。
- Cycle 28 の `[cuda_feature_calc_keypoints_fingerprint]` は `calcKeypoints` 内部に限定した read-only 診断である。
  `TRIORB_CUDA_FEATURE_KEYPOINT_FINGERPRINT_MAX_FRAMES=N` を明示した場合だけ有効化し、
  pyramid level の入力画像 hash、mask hash、candidate count、候補座標 hash、block count 分布を記録する。
  通常 run では無効のままにし、feature 抽出のしきい値、候補 cap、並び順は変更しない。
- Loop1 deterministic では `calcImagePyramid()` の level 0 copy を `image.copyTo(images[0], stream)` に変更し、
  caller が指定した CUDA stream 上で pyramid 入力の copy と後続 kernel の順序を固定する。従来の stream なし
  copy は default stream と per-camera stream の依存が曖昧になり、同一 HDF5 の 2 run で descriptor payload が
  まれに分岐する起点になっていた。
- `radiusSuppression` の同一 response tie-break は候補座標 `(y, x)` の辞書順に固定する。低 texture 領域では
  response 同値が多く、block-local emission 順に任せると survivor が run ごとに変わるためである。
- `TRIORB_CUDA_FEATURE_ANGLE_QUANTIZATION_DEG` は angle payload を指定 degree 単位へ丸める診断 knob である。
  BISON-01 では差分縮小に寄与したが、最終的な bit-exact trajectory には stream copy 修正で到達したため、
  既定は `0` のままにする。
- deterministic FAST の 10% area cap は既存の `cvRound(CORNER_DENSITY * image.area())` を変更しない。
  初回 counter が cap 以下なら従来の候補 buffer をそのまま使い、overflow 時だけ instance 専用の再利用 buffer へ
  全候補を再取得して固定 SplitMix64 座標 rank で pre-cap する。これにより current capped subset の並べ替えではなく、
  full raw set から platform 非依存の subset を選ぶ。
- full recapture 用 device / pinned host buffer は level 0 の最大容量を保持して縮小 level と次 frame で再利用する。
  4 camera slot はそれぞれの `EfficientFeaturesImpl` instance に buffer を持つため、stream 間で候補列を共有しない。

## 目標

- upstream 更新や参照比較時に、この file を導入している理由と利用位置を短時間で確認できる状態を保つ。
- project 側の差分が必要になった場合でも、変更理由を wrapper 側文書と合わせて追えるようにする。
- x86 / DGX で feature set が分岐する stage を、`calcKeypoints`、`calcResponses`、
  `radiusSuppression`、`limitPoints` の順に特定できるようにする。
- `calcKeypoints` 直後に分岐が見えた場合、入力画像 / mask / FAST candidate emission のどこで分岐が始まるかを
  `[cuda_feature_calc_keypoints_fingerprint]` で切り分けられる状態にする。
- deterministic overflow で選ばれた座標、response、descriptor の対応と multi-level buffer reuse を 4 stream test で固定する。

## 関連

- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_efficient_features.cpp`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_efficient_features.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_fast.cu`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_fast.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/deterministic_fast_precap.h`
- `slam-core/3rd/cuda-efficient-features/tests/fast_precap_test.cpp`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/bad.p512.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_akaze.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_hash_sift.md`
