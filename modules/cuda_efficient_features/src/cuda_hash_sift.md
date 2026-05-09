# cuda_hash_sift.cpp

## 目的

`cuda_hash_sift` は SIFT 系 descriptor / hashing 処理を提供する module です。同梱コードまたは参照実装の役割を、このリポジトリの文脈で素早く把握できるようにします。

## 対象範囲

- 対象 source: `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_hash_sift.cpp`
- 判定条件: 512 行のため sidecar 文書を維持対象にする
- 主な定義: `MatmulAndSign`、`SIFTImpl`、`SphericalSIFTImpl`、`HashSIFTImpl`、`SphericalHashSIFTImpl`、`printf`、`hashSIFTGemm`、`operator`
- 主な依存: `opencv2/cudaarithm.hpp`、`opencv2/core/cuda_stream_accessor.hpp`、`cuda_efficient_descriptors.h`、`algorithm`、`cublas_v2.h`、`cuda_hash_sift_internal.h`

## 現状

### 主な構成
- `MatmulAndSign` がこの module の主要な構成要素になっている
- `SIFTImpl` がこの module の主要な構成要素になっている
- `SphericalSIFTImpl` がこの module の主要な構成要素になっている
- `HashSIFTImpl` がこの module の主要な構成要素になっている
- `SphericalHashSIFTImpl` がこの module の主要な構成要素になっている
- `printf` がこの module の主要な構成要素になっている

### 連携境界
- `opencv2/cudaarithm.hpp` と連携しながら責務を完結させる
- `opencv2/core/cuda_stream_accessor.hpp` と連携しながら責務を完結させる
- `cuda_efficient_descriptors.h` と連携しながら責務を完結させる
- `algorithm` と連携しながら責務を完結させる
- `cublas_v2.h` と連携しながら責務を完結させる
- `cuda_hash_sift_internal.h` と連携しながら責務を完結させる

## 実装上の判断

- 同梱コードは upstream や参照実装としての責務を尊重し、この文書ではこのリポジトリから見た役割に絞って説明する。
- project 固有の判断は wrapper や利用側へ寄せ、この file 自体の変更理由を追いやすくする。
- 長大 file でも source 本体は read-only 前提で扱い、補足説明は sidecar 文書へ追加する。
- Loop1 deterministic では patch SIFT histogram の生成を deterministic kernel へ切り替えられるようにした。
  `TRIORB_CUDA_HASH_SIFT_DETERMINISTIC` が未指定または truthy の場合は、feature ごとに 1 thread が histogram
  生成、正規化、clip を固定順で実行する。従来の並列 kernel は `atomicAdd` による加算順が run ごとに揺れ、
  同一 keypoint 座標でも response / descriptor payload が数 bit 分岐していた。
- `MatmulAndSign` は cuBLAS handle に `CUBLAS_ATOMICS_NOT_ALLOWED` と `CUBLAS_DEFAULT_MATH` を設定する。
  HashSIFT の射影行列積でライブラリ側の atomic reduction や TF32 近似が混ざらないようにするためである。
- `TRIORB_CUDA_HASH_SIFT_DETERMINISTIC_PROJECT=1` は cuBLAS GEMM を使わず feature ごとの固定順 dot product で
  descriptor を二値化する診断 knob である。BISON-01 A/B では closure が悪化したため、既定は無効にする。

## 目標

- upstream 更新や参照比較時に、この file を導入している理由と利用位置を短時間で確認できる状態を保つ。
- project 側の差分が必要になった場合でも、変更理由を wrapper 側文書と合わせて追えるようにする。
- HDF5 replay では HashSIFT descriptor payload の run-to-run 差分を消し、同一 sensor frame 列から同一
  Rust trajectory を得られる状態を保つ。

## 関連

- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_hash_sift.cpp`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_hash_sift.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/bad.p512.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_akaze.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_efficient_features.md`
