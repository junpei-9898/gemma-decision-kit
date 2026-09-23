# Gemma Decision Kit

**NVFP4版Gemma 4で、テキスト・画像・短い動画を3択判定するローカル実行キット**です。文章の回答を生成せず、選択結果と未校正の確率分布を返します。

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
