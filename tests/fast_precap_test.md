# fast_precap_test.cpp

## 目的

deterministic FAST overflow pre-cap が、GPUのatomic emission順ではなく固定座標hashから同じ候補集合を選ぶ契約を固定します。

## 対象範囲

- SplitMix64 `pack_u16(y,x)` v1 のknown-answer
- 既存 `cvRound(0.1 * image.area())` のcap境界
- full recaptureのcount、容量、座標、重複fail-close
- 4 CUDA stream、6 pyramid level、response / descriptor対応、buffer再利用
- raw / selectedの4x4・8x8 occupancy比較

## 現状

`fast_precap_tests` は既存descriptor fixtureのhighgui依存から分離したtargetです。pure helper 7件と実GPU経路2件を単独実行できます。

## 実装上の判断

cap+1と3784→3608の期待列はproduction helperを流用せず、test内の独立CPU referenceで算出します。複数入力順、4 slot、逆順再実行を比較し、current capped subsetの並べ替えだけでは通らない契約にしています。

## 目標

fixed hash policy、legacy default-off経路、multi-camera buffer分離の回帰を短いtargetで検出できる状態を保ちます。

## 関連

- `modules/cuda_efficient_features/src/deterministic_fast_precap.h`
- `modules/cuda_efficient_features/src/cuda_fast.cu`
- `modules/cuda_efficient_features/src/cuda_fast.md`
