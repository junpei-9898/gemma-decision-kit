# Gemma Decision Kit

**NVFP4版Gemma 4で、テキスト・画像・短い動画を3択判定するローカル実行キット**です。文章の回答を生成せず、選択結果と未校正の確率分布を返します。

## 日本語判定の精度と速度

**GB10で一致率93.3%（112/120）、短文1問はwarm中央値約80ms。** 下記6構成の比較では最も高い一致率でした。同一機でHTTP計測したDiffusionGemmaとの比較では、短文1問は**1.62倍高速**、v1一致率は**5.0ポイント高い**結果です。

日本語の証拠から「支持・矛盾・情報不足」を判定する固定タスクです。精度欄は**AI暫定ラベルとの一致率**で、人間監査済みの正答率ではありません。品質は120ケース、速度は本文54文字の1問を20回反復した別の測定です。

| 構成 | 一致率（v1・120例） | 短文1問・中央値 | 測定 |
|---|---:|---:|---|
| Gemma Decision Kit · Gemma 4 26B-A4B NVFP4 | **93.3% (112/120)** | **80.4ms** | A |
| DiffusionGemma 26B-A4B NVFP4 | 88.3% (106/120) | 130.6ms | A |
| Eider · Qwen3.6-35B-A3B NVFP4 | 85.8% (103/120) | 250.3ms | B |
| SemIf · Qwen3.5-4B BF16 | 64.2% (77/120) | 119.6ms | B |
| Laya multilingual · 0.322B | 39.2% (47/120) | 8.8ms | B |
| NanoJev · 0.6B navigation checkpoint | 38.3% (46/120) | 40.6ms | B |

**A:** 2026-09-23、同一GB10機でloopback HTTP計測。**B:** 2026-09-20、別のGB10機で計測。QwenはHTTP、ほかはPython APIです。モデルロード後のwarm中央値で、起動時間は除外。モデルサイズ・量子化・テンプレート・API境界が異なるため、完全同条件の速度ランキングではありません。本キットの行は固定最適化レシピを評価用HTTP経由で測定した値です。v0.2.0で判定の一致は確認済みですが、公開サーバーで常に同じレイテンシになる保証ではありません。

Laya・NanoJevはこの短文ではさらに高速ですが、本タスクの一致率は低い結果でした。また8問workflowではDiffusionGemmaが優位です（377.6ms対732.8ms、一致40/40対35/40。同じ8問を5回反復）。NanoJevはnavigation向けcheckpointの結果であり、日本語汎用判定向けモデル全般の評価ではありません。

[比較条件・指示文による変化・固定版](docs/COMPARISON.md)・[集計JSON](docs/comparison-results.json)・[テキスト/画像/動画の速度と初回起動の注意](docs/BENCHMARKS.md)

## 機能と使い方

- v0.2.0はNVFP4に一本化。EXL3は現行配布から外し、旧v0.1.0の履歴に保持。
- 通常起動は従来のテキスト高速構成。`--media`で画像・動画に対応する構成を起動。
- 同じNVFP4モデルを使いますが、テキスト専用と画像・動画対応ではKV設定や最適化が異なります。
- 画像はPNG/JPEGを1枚、動画はMP4を1本。動画はフレームを抽出して視覚情報を判定し、音声は処理しません。
- 3択・複数質問の逐次処理。真偽値型、score型、自由文生成、音声は未対応。
- 実機確認環境はEdge Xpert / GB10、Linux ARM64。別GPUでの動作や性能は未認定。

```sh
gemma-decision predict --model-path /models/nvfp4 --input examples/request.json
gemma-decision predict --media --model-path /models/nvfp4 --input examples/image-request.json
gemma-decision predict --media --model-path /models/nvfp4 --input examples/video-request.json
gemma-decision serve --media --model-path /models/nvfp4 --port 8765
```

[導入手順](docs/INSTALL.md)・[画像/動画のAPIと制限](docs/MEDIA.md)・[検証結果](docs/BENCHMARKS.md)。APIは127.0.0.1限定です。

独立した実験版であり、公式Jev・Eiderの互換品を認定するものではありません。確率は正答率の保証ではなく、テストの正解はAI暫定ラベルです。小規模合成素材での検証を一般的な精度保証に読み替えないでください。

コードと自作サンプルはApache-2.0。モデル重みは別途ダウンロードします。[ライセンス詳細](docs/LICENSES.md)。
