# Gemma Decision Kit

**Eider＋最適化済みvLLM＋NVFP4 Gemma 4**で、テキスト・画像・動画・文字起こしした音声を判定します。出力は `choice`・`noul`・`score` と未校正の確率です。[English](README.md)

v0.7はEiderに一本化しました。実際のEiderが質問の準備と判定結果の生成を担当し、vLLMがモデルを計算します。音声はMOSS、動画分割と音声との時間合わせは本キットが担当します。

## 導入と使い方

[固定ランタイム・モデルの導入](docs/INSTALL.md) → [必須のEiderブリッジをビルド](docs/EIDER.md)して `GEMMA_EIDER_LIBRARY` を設定。音声を使う場合は[MOSS・FFmpeg](docs/AUDIO.md)も導入してください。

```sh
gemma-decision predict --model-path /models/nvfp4 --input examples/eider-request.json
gemma-decision analyze --model-path /models/nvfp4 \
  --source /input/recording.mp4 --input examples/request.json \
  --audio-model-path /models/moss --audio-python /state/moss-env/bin/python
```

- `predict` / `serve`：JSON入力。`serve` はGemmaを常駐。画像・短い動画を直接渡す場合は起動時に `--media`。
- `analyze` / `serve-input`：ファイル形式を自動判定。音声認識が終わってからGemmaをロード。HTTP版もリクエストごとにロードします。
- `transcribe` / `download-audio`：任意の文字起こし・音声モデル取得ツール。

GPU設定は通常 `auto` のまま使えます。GB10搭載のDGX Spark／Edge Xpert等は `spark`、対応条件を満たす他GPUは `standard` を選択します。他GPU実機は未検証です。[条件](docs/HARDWARE.md)

`legacy`、`--semantics`、`--profile`、`gb10` 別名、`predict --audio` は削除しました。Python引数・出力形式も変わるため、[移行手順](docs/MIGRATION.md)を確認してください。

## 現在の検証範囲

直近のEider実機評価（v0.6）は既存64問の回答・確率が一致し、暫定ラベルとの一致は61/64。短文1問（121〜188トークン）のウォーム中央値は76.36msでした。画像9/9、短い動画3/3、音声1/1は少数の合成素材による確認です。

**映像＋音声の複合判断はEider1/2、旧方式2/2という未解決の差があります。** 問題文に曖昧さがありますが、成功扱いにはしていません。Eider一本化は配布方針の整理であり、この誤答を修正したという意味ではありません。v0.7ではCPU・パッケージを検証し、新たなGPU性能測定は行っていません。[詳細](docs/EIDER_RELEASE_VALIDATION.md)

音声は文字起こし経由、動画はフレーム抽出と区間判定です。区間をまたぐ推論には制約があり、複数区間のscore/noul集約は未対応。音声・ファイル処理にはロード時間が含まれ、短文の約76msとは異なります。[入力仕様](docs/UNIFIED_INPUT.md)

テキストは既定64K・最大256K、画像／動画は展開後8192入力トークン。長文での正答率低下は今後の課題です。[実測グラフ](docs/CONTEXT.md)。Eiderのメディア試験は約23.5GiB allocated／25.4GiB reservedでしたが、24GB GPUに収まる保証ではありません。

## 過去の構成で測定した日本語判定の精度と速度

**GB10で一致率93.3%（112/120）、短文1問はwarm中央値約80ms。** 下記6構成の比較では最も高い一致率でした。同一機でHTTP計測したDiffusionGemmaとの比較では、短文1問は**1.62倍高速**、v1一致率は**5.0ポイント高い**結果です。

### 検証ハードウェア

| 項目 | 実測環境 |
|---|---|
| マシン・GPU | Edge Xpert、NVIDIA GB10（Blackwell・SM121）を1基使用 |
| メモリ | CPU/GPU共有の統合メモリ。OS認識約121.6GiB。**モデル専用VRAMの容量ではありません** |
| 実行環境 | Linux ARM64。本キットはPyTorch2.11.0+cu130。各構成のランタイムを固定 |
| A試験のCPU使用上限 | コンテナ8CPU相当、`OMP_NUM_THREADS=4` |

より強力なGPUでは、特に長文のprefillを高速化できる可能性があります。ただし他GPUは未測定で、**「数倍速くなる」という倍率は未検証**です。CPU前処理・メモリ帯域・対応カーネルにも左右されるため、公称TOPS等の比率をそのまま速度に掛けることはできません。配布手順のARM64/GB10用イメージは、x86/RTX環境でそのまま動作確認したものではありません。

### モデルのメモリ使用量

| NVFP4の構成 | GPUテンソル使用量のピーク | allocator予約領域のピーク |
|---|---:|---:|
| テキスト高速構成 |21.99GiB|23.91GiB|
| 画像・動画対応（`--media`） |23.11GiB|24.10GiB|

v0.2.0のGB10実機試験での値です。画像・動画側は起動時のプロファイルも含みます。予約領域は使用中の領域を含むため、**2列を足しません**。ドライバー・CPU側メモリをすべて含む値でも、最小必要VRAMの保証でもありません。GB10ではCPU/GPUがメモリを共有するため、OS・ランタイム・ロード用の余裕も必要です。**モデルに121.6GiB必要という意味ではありませんが、24GBの別GPUで動くことも未確認です。** [定義と測定根拠](docs/MEMORY_AND_LONG_INPUT.md)。

日本語の証拠から「支持・矛盾・情報不足」を判定する固定タスクです。精度欄は**AI暫定ラベルとの一致率**で、人間監査済みの正答率ではありません。品質は120ケース、速度は本文54文字の1問を20回反復した別の測定です。

| 構成 | 一致率（v1・120例） | 短文1問・中央値 | 測定 |
|---|---:|---:|---|
| Gemma Decision Kit · Gemma 4 26B-A4B NVFP4 | **93.3% (112/120)** | **80.4ms** | A |
| DiffusionGemma 26B-A4B NVFP4 | 88.3% (106/120) | 130.6ms | A |
| Eider · Qwen3.6-35B-A3B NVFP4 | 85.8% (103/120) | 250.3ms | B |
| SemIf · Qwen3.5-4B BF16 | 64.2% (77/120) | 119.6ms | B |
| Laya multilingual · 0.322B † | 39.2% (47/120) | 8.8ms | B |
| NanoJev · 0.6B navigation checkpoint | 38.3% (46/120) | 40.6ms | B |

**A:** 2026-09-23、同一GB10機でloopback HTTP計測。**B:** 2026-09-20、別のGB10機で計測。QwenはHTTP、ほかはPython APIです。モデルロード後のwarm中央値で、起動時間は除外。モデルサイズ・量子化・テンプレート・API境界が異なるため、完全同条件の速度ランキングではありません。本キットの行は固定最適化レシピを評価用HTTP経由で測定した値です。v0.2.0で判定の一致は確認済みですが、公開サーバーで常に同じレイテンシになる保証ではありません。

**† Layaの全文読込みについて：** 表の8.8msは本文54文字の短文で、**切断なし**、同じ1問の20反復は20/20一致です。一方、2,099文字・8,108文字では各5/5件で全文を保持できず、21.6ms・23.1msという速さでも各0/5一致でした。これは**長文を全文読んだ速度ではありません**。v2品質試験も100/100件で対象主張を含む指示部分が切断されています。表の品質欄に使ったv1の120例には切断がなく、39.2%という一致率を切断だけの結果とはしていません。[入力coverageの内訳](docs/MEMORY_AND_LONG_INPUT.md)。

Laya・NanoJevはこの短文ではさらに高速ですが、本タスクの一致率は低い結果でした。また8問workflowではDiffusionGemmaが優位です（377.6ms対732.8ms、一致40/40対35/40。同じ8問を5回反復）。NanoJevはnavigation向けcheckpointの結果であり、日本語汎用判定向けモデル全般の評価ではありません。

[比較条件・指示文による変化・固定版](docs/COMPARISON.md)・[集計JSON](docs/comparison-results.json)・[テキスト/画像/動画の速度と初回起動の注意](docs/BENCHMARKS.md)

### 長文の最適化前後

**研究用nativeランナーの過去の実測です。** 当時の内部上限は65,536トークンです。v0.3.0では公開APIを最大262,143入力トークンまで拡張しました。[新しいHTTP実測と長さ別の精度](docs/CONTEXT.md)は別試験であり、下表の速度と混同しないでください。

| 本文文字数 / prompt tokens | 安定化済み基準 | 最終構成 | 時間短縮 |
|---|---:|---:|---:|
|10,000 / 7,138|1.563s|1.319s|15.6%|
|30,000 / 21,197|6.569s|4.570s|30.4%|
|50,000 / 35,257|15.248s|8.686s|43.0%|

同一GB10・固定Gemma4 NVFP4重み・同じ全文入力、1質問、モデルと形状を準備済み、prefix再利用なし、各3反復の中央値。入力準備＋native推論を含み、ロード・HTTP・計装時間は除外しています。各長さ1文書の結果で、任意の長文で同じ速度を保証するものではありません。主な改善はAttentionの実行設定、MoEのバッファ・コピー削減、ABC判定用の出力投影です。

基準は**安定化済みnative CUTLASS構成**です。それ以前のFlashInfer構成との比較では、5万文字13.714秒→8.686秒（36.7%短縮）ですが、1万文字は1.242秒→1.319秒で遅くなっています。安定性・品質も異なるため、すべての旧構成・入力に対して高速化したとはしていません。[段階別結果と制約](docs/MEMORY_AND_LONG_INPUT.md)。

## 比較結果・ライセンス

[三択以外の出力比較](docs/TYPED_OUTPUTS.md)・[他モデル比較](docs/COMPARISON.md)・[長文速度改善](docs/MEMORY_AND_LONG_INPUT.md)・[過去の研究資料](docs/history/README.md)。以下を含む過去の数値は元の版の実測で、v0.7の再測定ではありません。

コードはApache-2.0。Eiderのライセンス・NOTICE・出典情報を保持しています。モデルとランタイムは別ライセンスで、重みは同梱しません。[詳細](docs/LICENSES.md)。研究記録はGitに保存し、通常のインストール物から分離しています。
