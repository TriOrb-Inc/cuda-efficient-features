# bad.cpp

## 目的

`bad` は BAD descriptor 系の feature 計算を提供する module です。同梱コードまたは参照実装の役割を、このリポジトリの文脈で素早く把握できるようにします。

## 対象範囲

- 対象 source: `slam-core/3rd/cuda-efficient-features/modules/efficient_features/src/bad.cpp`
- 判定条件: 557 行のため sidecar 文書を維持対象にする
- 主な定義: `BoxPairParams`、`BAD_Impl`、`SphericalBAD_Impl`、`hasValidLens`、`resolveLens`、`scaleLensForImage`、`Point2f`、`computeBAD`
- 主な依存: `opencv2/imgproc.hpp`、`efficient_descriptors.h`、`spherical_projection.hpp`、`algorithm`、`cmath`、`bad.p512.h`

## 現状

### 主な構成
- `BoxPairParams` がこの module の主要な構成要素になっている
- `BAD_Impl` がこの module の主要な構成要素になっている
- `SphericalBAD_Impl` がこの module の主要な構成要素になっている
- `hasValidLens` がこの module の主要な構成要素になっている
- `resolveLens` がこの module の主要な構成要素になっている
- `scaleLensForImage` がこの module の主要な構成要素になっている

### 連携境界
- `opencv2/imgproc.hpp` と連携しながら責務を完結させる
- `efficient_descriptors.h` と連携しながら責務を完結させる
- `spherical_projection.hpp` と連携しながら責務を完結させる
- `algorithm` と連携しながら責務を完結させる
- `cmath` と連携しながら責務を完結させる
- `bad.p512.h` と連携しながら責務を完結させる

## 実装上の判断

- 同梱コードは upstream や参照実装としての責務を尊重し、この文書ではこのリポジトリから見た役割に絞って説明する。
- project 固有の判断は wrapper や利用側へ寄せ、この file 自体の変更理由を追いやすくする。
- 長大 file でも source 本体は read-only 前提で扱い、補足説明は sidecar 文書へ追加する。

## 目標

- upstream 更新や参照比較時に、この file を導入している理由と利用位置を短時間で確認できる状態を保つ。
- project 側の差分が必要になった場合でも、変更理由を wrapper 側文書と合わせて追えるようにする。

## 関連

- `slam-core/3rd/cuda-efficient-features/modules/efficient_features/src/bad.cpp`
- `slam-core/3rd/cuda-efficient-features/modules/efficient_features/src/bad.md`
- `slam-core/3rd/cuda-efficient-features/modules/efficient_features/src/bad.p512.md`
- `slam-core/3rd/cuda-efficient-features/modules/efficient_features/src/hash_sift.md`
