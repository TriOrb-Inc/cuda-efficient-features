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

## 実装上の判断

- 既定では完全に無効です。`TRIORB_CUDA_FEATURE_KEYPOINT_FINGERPRINT_MAX_FRAMES=N` を明示した診断 run だけで有効化します。
- 診断は CPU download を伴いますが、`calcKeypoints` は既に counter 回収のため `cudaStreamSynchronize` しているため、追加同期の影響は stage 間の挙動変更ではなく診断 run の観測コストとして扱います。
- block 分布は kernel 内で別 counter を増やさず、cap 後の候補点を CPU 側で block grid へ再配置して算出します。これにより algorithm と GPU memory write path を変えずに候補 emission の偏りを観測できます。

## 目標

- x86 / DGX の分岐が入力画像 / mask / FAST candidate emission のどこで始まるかを 1 run で切り分ける。
- image / mask hash が一致して candidate hash だけが分岐する場合、次の調査対象を FAST kernel 内の threshold 比較、atomic emission order、block-local 分布へ絞る。
- image hash が分岐する場合、pyramid 生成または input upload 経路へ調査対象を戻す。

## 関連

- `cuda_fast.cu`
- `cuda_efficient_features.cpp`
- `cuda_efficient_features.md`
- `slam-core/docs/cycle28-calc-keypoints-fingerprint.md`
