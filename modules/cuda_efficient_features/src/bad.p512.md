# bad.p512.h

## 目的

`bad.p512` は BAD descriptor 系の feature 計算を提供する module です。同梱コードまたは参照実装の役割を、このリポジトリの文脈で素早く把握できるようにします。

## 対象範囲

- 対象 source: `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/bad.p512.h`
- 判定条件: 394 行のため sidecar 文書を維持対象にする

## 現状

### 主な構成
- file 内の処理量が大きいため、隣接 module との境界を追える補助資料として扱う

### 連携境界
- 同一 package 内の class / utility と連携して役割を分担する

## 実装上の判断

- 同梱コードは upstream や参照実装としての責務を尊重し、この文書ではこのリポジトリから見た役割に絞って説明する。
- project 固有の判断は wrapper や利用側へ寄せ、この file 自体の変更理由を追いやすくする。
- 長大 file でも source 本体は read-only 前提で扱い、補足説明は sidecar 文書へ追加する。

## 目標

- upstream 更新や参照比較時に、この file を導入している理由と利用位置を短時間で確認できる状態を保つ。
- project 側の差分が必要になった場合でも、変更理由を wrapper 側文書と合わせて追えるようにする。

## 関連

- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/bad.p512.h`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/bad.p512.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_akaze.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_efficient_features.md`
- `slam-core/3rd/cuda-efficient-features/modules/cuda_efficient_features/src/cuda_hash_sift.md`
