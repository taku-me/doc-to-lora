# レビュー結果: D2L Qwen3-4B B1ベンチマークスクリプトの評価

## 良い点
- base/rag/d2lの3モード比較を同一のin-process環境で公平に実装している
- モデルのresetとpatch_lora_forwardを利用し、LoRA化の切り替えを正しく行っている
- Listing 8プロンプトフォーマットを忠実に再現している

## 問題点
- **System Roleの欠落による分布外入力**: Qwenはシステムプロンプトに強く依存する。学習データ加工処理 (`convert_ctx_prompt_response_to_messages`) では必ず空のシステムロール (`{"role": "system", "content": ""}`) が先頭に付与されていたが、`build_chat_inputs` ではUserロールのみでトークナイズしている。これが原因でベースモデルの挙動が不安定化し、事実の取りこぼしやハルシネーションが発生している。
  - **代替案**: `build_chat_inputs` 内の `apply_chat_template` に渡すリストを `[{"role": "system", "content": ""}, {"role": "user", "content": content}]` に修正する。
- **Context Chunkingの未実装によるキャパシティ超過**: 論文の「Chunkingによる長文合成」が実装されていない。約1900トークンのpassageを単一チャンク (`shape: [1, 1900]`) として処理しているが、学習時の `max_ctx_chunk_len` は512である。Perceiverの固定Latent (r=8) に訓練時の4倍近い情報を一度に圧縮しようとすると、情報が確実に欠落する。
  - **代替案**: `tokenize_passage` 内でトークン列を `max_ctx_chunk_len` (512) 単位の複数チャンク（次元: `[n_chunks, 512]`）に分割・パディングし、`n_ctx_chunks` に正しい分割数（`[n_chunks]`）を渡すよう実装する。
- **LoRAのTarget Module不足とRank不足**: `target_modules` が `down_proj` のみかつ `lora_r: 8` では、4Bモデルのパラメータにおいてアダプタのキャパシティが小さすぎる。単一の射影層だけで数千トークン分の知識を完全に内部化するのは困難である。
  - **代替案**: 次回の学習設定で `target_modules` を `["gate_proj", "up_proj", "down_proj", "q_proj", "v_proj"]` などへ拡大し、`lora_r` を 16 もしくは 32 に引き上げる。
- **評価指標 (ROUGE-L) の過度なペナルティ**: 短答形式のタスクに対してROUGE-Lを使用すると、冠詞や句読点の微小な差異でスコアが不当に下落する。論文内の評価基準である QA F1 を用いないと、完全成立の有無を正確に測れない。
  - **代替案**: D2L論文で採用されているSQuAD相当のテキスト正規化（小文字化、a/an/the等の冠詞除去、句読点除去）を前処理として適用した Token F1 スコアを採用する。

## 次に何をすべきか
`scripts/bench_b1_qwen.py` のチャンク分割処理（512トークン単位での `ctx_ids` 分割）と `build_chat_inputs` への System Role の追加を直ちに実装する。これらは推論時の入力を学習時の分布に引き戻すための必須要件であり、このズレを解消しない限り D2L 本来の内部化性能は正当に評価できない。
