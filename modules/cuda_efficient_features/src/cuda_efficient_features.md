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

## 目標

- upstream 更新や参照比較時に、この file を導入している理由と利用位置を短時間で確認できる状態を保つ。
- project 側の差分が必要になった場合でも、変更理由を wrapper 側文書と合わせて追えるようにする。
- x86 / DGX で feature set が分岐する stage を、`calcKeypoints`、`calcResponses`、
  `radiusSuppression`、`limitPoints` の順に特定できるようにする。

## 関連

- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_efficient_features.cpp`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_efficient_features.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/bad.p512.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_akaze.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_hash_sift.md`
