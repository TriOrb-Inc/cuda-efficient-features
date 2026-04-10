# sample_feature_matching.cpp

## 目的

`sample_feature_matching` は このリポジトリが同梱している外部実装を利用可能な形で保持する module です。同梱コードまたは参照実装の役割を、このリポジトリの文脈で素早く把握できるようにします。

## 対象範囲

- 対象 source: `slam-core/3rd/cuda-efficient-features/samples/sample_feature_matching.cpp`
- 判定条件: 580 行のため sidecar 文書を維持対象にする
- 主な定義: `Dataset`、`findInputRoot`、`resolveDatasets`、`findResultsPath`、`updateResults`、`ifs`、`ensureTrailingBlank`、`ofs`
- 主な依存: `opencv2/calib3d.hpp`、`opencv2/core.hpp`、`opencv2/imgproc.hpp`、`opencv2/highgui.hpp`、`algorithm`、`array`

## 現状

### 主な構成
- `Dataset` がこの module の主要な構成要素になっている
- `findInputRoot` がこの module の主要な構成要素になっている
- `resolveDatasets` がこの module の主要な構成要素になっている
- `findResultsPath` がこの module の主要な構成要素になっている
- `updateResults` がこの module の主要な構成要素になっている
- `ifs` がこの module の主要な構成要素になっている

### 連携境界
- `opencv2/calib3d.hpp` と連携しながら責務を完結させる
- `opencv2/core.hpp` と連携しながら責務を完結させる
- `opencv2/imgproc.hpp` と連携しながら責務を完結させる
- `opencv2/highgui.hpp` と連携しながら責務を完結させる
- `algorithm` と連携しながら責務を完結させる
- `array` と連携しながら責務を完結させる

## 実装上の判断

- 同梱コードは upstream や参照実装としての責務を尊重し、この文書ではこのリポジトリから見た役割に絞って説明する。
- project 固有の判断は wrapper や利用側へ寄せ、この file 自体の変更理由を追いやすくする。
- 長大 file でも source 本体は read-only 前提で扱い、補足説明は sidecar 文書へ追加する。

## 目標

- upstream 更新や参照比較時に、この file を導入している理由と利用位置を短時間で確認できる状態を保つ。
- project 側の差分が必要になった場合でも、変更理由を wrapper 側文書と合わせて追えるようにする。

## 関連

- `slam-core/3rd/cuda-efficient-features/samples/sample_feature_matching.cpp`
- `slam-core/3rd/cuda-efficient-features/samples/sample_feature_matching.md`
