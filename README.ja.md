# Gemma Decision Kit

Gemma4を使い、文章から3択の判定と確率を返す実験的なCLI/APIです。自由文の回答は返しません。新しいモデルを学習したものではなく、既存モデルの判定処理を最適化した独立実装です。公式JevやEiderの互換製品ではありません。

- `speed`: NVFP4＋vLLM。速度を優先。
- `memory`: EXL3 4.10bpw＋ExLlamaV3。メモリを優先。
- 1プロセスで片方だけロードします。切り替えには停止・再起動が必要です。
- 初期版はテキスト・3択のみ。複数質問も逐次処理します。画像・動画・音声、真偽値型、スコア型は未対応です。
- 確率は校正されていません。高確率でも正解とは限りません。バックエンドによって判定が変わる場合があります。

[固定環境の導入手順](docs/INSTALL.md)に従ってください。実機検証はLinux ARM64のGB10/Edge Xpertで行っています。ほかのGPUで同じ性能・動作を保証するものではありません。モデル重みは同梱せず、利用者が固定revisionを取得します。

```sh
gemma-decision predict --profile speed --model-path /models/nvfp4 --input examples/request.json
gemma-decision serve --profile memory --model-path /models/exl3 --port 8765
curl http://127.0.0.1:8765/v1/decisions -H 'Content-Type: application/json' --data-binary @examples/request.json
```

入力は[例](examples/request.json)、実機の応答は[NVFP4](examples/response-speed.json)・[EXL3](examples/response-memory.json)を参照。APIはlocalhost限定です。上限超過入力は切り捨てずエラーにします。

[速度・メモリ・精度・既知の制約](docs/BENCHMARKS.md)。過去のベンチマークは固定AI暫定ラベルを使用したもので、未知データでの品質保証ではありません。配布パッケージへの移植時は両構成それぞれ231判定で、元の判定と確率の完全一致を確認しました。

新規コードはApache-2.0。ExLlama由来のMIT部分の表示も保持しています。[NOTICE](NOTICE)と[ライセンスの区分](docs/LICENSES.md)を確認してください。モデル重み・実行イメージのライセンスは配布コードとは別に扱います。
