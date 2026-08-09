# cuda_fast.cu

## 目的

`cuda_fast.cu` は `cuda-efficient-features` の FAST keypoint 候補検出 kernel を持つ source です。Cycle 28 では x86 / DGX の feature set 分岐が `calcKeypoints` 直後から発生しているため、この file に read-only の fingerprint 診断を追加しています。

## 対象範囲

- 対象 source: `cuda_fast.cu`
- 主な処理: `calcKeypointsKernel`、`calcKeypoints`
- 診断 log: `[cuda_feature_calc_keypoints_fingerprint]`
- 環境変数: `TRIORB_CUDA_FEATURE_KEYPOINT_FINGERPRINT_MAX_FRAMES`

## 現状

`calcKeypointsKernel` は FAST 判定に通った pixel を `atomicInc` で global counter へ積み、`keypoints` の row 0 へ `(x, y)` を書き込みます。Cycle 27 の stage fingerprint では、この `calcKeypoints` 出力時点ですでに x86 / DGX の座標集合が一致しないことが分かりました。

Cycle 28 の追加診断は、`calcKeypoints` 直後に以下を標準出力へ記録します。

- pyramid level の入力画像 hash / mean / min / max
- mask hash / valid pixel 数
- raw candidate count / cap 後 candidate count
- candidate 座標の ordered hash / set hash
- CUDA block grid へ候補点を再 binning した block count hash / 分布
- 先頭 3 点の `(x, y)` sample

deterministic mode では、初回 kernel の全候補 counter が既存の
`cvRound(0.1 * image.area())` cap を超えた場合だけ image-area buffer へ全候補を再取得します。
初回 count、再取得 count、実書込み count が一致したことを確認してから、
`SplitMix64((u64(y) << 16) | u64(x))`、tie `(y, x)` の固定 rank で cap 分を選びます。
timestamp、sensor、pyramid level、atomic index は rank に使いません。

## 実装上の判断

- 既定では完全に無効です。`TRIORB_CUDA_FEATURE_KEYPOINT_FINGERPRINT_MAX_FRAMES=N` を明示した診断 run だけで有効化します。
- 診断は CPU download を伴いますが、`calcKeypoints` は既に counter 回収のため `cudaStreamSynchronize` しているため、追加同期の影響は stage 間の挙動変更ではなく診断 run の観測コストとして扱います。
- block 分布は kernel 内で別 counter を増やさず、cap 後の候補点を CPU 側で block grid へ再配置して算出します。これにより algorithm と GPU memory write path を変えずに候補 emission の偏りを観測できます。
- hash pre-cap は deterministic mode の overflow 時だけ実行します。nonoverflow は初回 buffer の候補集合と順序をそのまま後段へ渡し、deterministic=false は従来の capped atomic write path を維持します。
- full recapture の device / pinned host buffer は extractor instance ごとに最大 image area まで拡張して再利用し、pyramid level や次 frame ごとの再確保を避けます。
- count 不一致、容量不足、画像外座標、重複座標は候補の一部採用や legacy fallback をせず fail-close します。
- fingerprint は policy version / seed、raw=selected+dropped 保存則、初回・再取得・実書込み count、raw / selected の座標 digest と 4x4 / 8x8 occupancy を記録します。

## 目標

- x86 / DGX の分岐が入力画像 / mask / FAST candidate emission のどこで始まるかを 1 run で切り分ける。
- image / mask hash が一致して candidate hash だけが分岐する場合、次の調査対象を FAST kernel 内の threshold 比較、atomic emission order、block-local 分布へ絞る。
- image hash が分岐する場合、pyramid 生成または input upload 経路へ調査対象を戻す。
- raw FAST が 10% cap を超える platform でも、atomic emission order ではなく固定座標 hash により同じ候補集合を後段へ渡す。

## 関連

- `cuda_fast.cu`
- `cuda_efficient_features.cpp`
- `cuda_efficient_features.md`
- `deterministic_fast_precap.h`
- `tests/fast_precap_test.cpp`
- `slam-core/docs/cycle28-calc-keypoints-fingerprint.md`
