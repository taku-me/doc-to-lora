# Audrey Tang × Sakana AI Doc-to-LoRA 報告書 検証タスク

## 背景
ユーザーが提示した長文の調査報告書に、AI が幻覚で生成したと思われる固有名詞が混在している可能性がある。Web 検索で 1 つずつ検証するタスク。

---

## 【優先度高】実在性が怪しい固有名詞

### 1. **jdd-kami** エージェント名
- 検索キーワード案:
  - "jdd-kami" Audrey Tang
  - "jdd-kami" AI agent
  - site:twitter.com jdd-kami
  - site:github.com jdd-kami
  - "jdd-kami" Taiwan
- 期待: Twitter/X, GitHub, ブログでの言及、実装コード等

### 2. **jdd-kami.eth** ENS アドレス
- 検索キーワード案:
  - jdd-kami.eth ethereum
  - site:etherscan.io jdd-kami.eth
  - "jdd-kami.eth"
- 期待: etherscan.io での登録確認、resolver 設定、オーナー情報

### 3. **ERC-8004 Agent ID #22714**
- 検索キーワード案:
  - ERC-8004 specification
  - "ERC-8004" ethereum
  - site:eips.ethereum.org ERC-8004
  - "Agent ID" #22714
- 期待: ERC-8004 の定義、#22714 がオンチェーン登録されているか

### 4. **Habermolt** 分散型熟議プラットフォーム
- 検索キーワード案:
  - Habermolt platform
  - "Habermolt" deliberation
  - site:github.com Habermolt
  - Habermolt Taiwan
- 期待: GitHub, 公式サイト、ドキュメント等での存在確認

### 5. **OpenClaw** オープンソース AI エージェントプラットフォーム
- 検索キーワード案:
  - OpenClaw github
  - "OpenClaw" AI agent
  - site:github.com OpenClaw
  - OpenClaw npm
- 期待: GitHub リポジトリ、npm パッケージ、README、ドキュメント

### 6. **NemoClaw** および **OpenShell** ランタイム
- 検索キーワード案:
  - NemoClaw NVIDIA GTC 2026
  - "NemoClaw" runtime
  - "OpenShell" runtime NVIDIA
  - site:nvidia.com NemoClaw OpenShell
  - NVIDIA GTC 2026 announcements
- 期待: NVIDIA 公式発表、GTC 2026 セッション資料

### 7. **Sakana The Conductor (7B)** モデル
- 検索キーワード案:
  - "The Conductor" Sakana AI 7B
  - site:huggingface.co Sakana Conductor
  - site:github.com SakanaAI Conductor
  - Sakana "The Conductor"
- 期待: HuggingFace での公開モデル、GitHub リリース、ブログ記事

### 8. **Gisele Chou**
- 検索キーワード案:
  - Gisele Chou Audrey Tang
  - "Gisele Chou" Taiwan
  - "Gisele Chou" AI researcher
  - site:linkedin.com Gisele Chou
- 期待: LinkedIn, 学術論文, 実務経歴

### 9. **「6-Pack of Care」フレームワーク
- 背景: Joan Tronto / Berenice Fisher のケア倫理は実在。原型は 5 phases (「5 phases of caring」)。
- 検索キーワード案:
  - "6-Pack of Care" Joan Tronto
  - "6 Pack of Care" ethics
  - Joan Tronto care phases
  - Berenice Fisher 5 phases care
- 期待: Tronto の著作、論文での言及、「6-Pack」の正式名称

### 10. **「Digital Citizenship as Freedom of Movement」文書**
- 検索キーワード案:
  - "Digital Citizenship as Freedom of Movement" Audrey Tang
  - Oxford Institute Ethics AI Audrey Tang
  - "Caroline Green" Audrey Tang
  - jdd-kami coauthor fellowship
- 期待: Oxford 公式サイト, 文書へのリンク, Caroline Green の実在確認

### 11. **「A Gentle Bridge」(軽柔之橋) 共著文書**
- 検索キーワード案:
  - "A Gentle Bridge" Audrey Tang "Tenzin Yangtso"
  - "軽柔之橋" Audrey Tang
  - Tenzin Yangtso Audrey Tang
- 期待: 文書本体、出版社、arXiv, Medium 等での掲載

### 12. **「仁工智慧 馬躍鳳騰」中国語表現**
- 検索キーワード案:
  - "仁工智慧 馬躍鳳騰" Audrey Tang
  - jdd-kami "仁工智慧"
  - "馬躍鳳騰" AI
- 期待: ソース文書 (blog, Twitter, インタビュー等)

---

## 【優先度高】性能数値の検証

### 13. **「2000 件トランスクリプト → 約 20 分で LoRA 化」**
- 検索キーワード案:
  - Audrey Tang "2000" "20 minutes" transcript LoRA
  - site:github.com/SakanaAI doc-to-lora "2000"
  - Sakana "Doc-to-LoRA" benchmark
  - arXiv:2602.15902 MacBook performance
- 期待: Audrey Tang 本人のツイート/ブログ, 論文での言及, Sakana 公式ブログ

### 14. **「1 件長文トランスクリプト → 約 1 秒で LoRA 化」**
- 検索キーワード案:
  - Audrey Tang "1 second" LoRA transcript
  - "Doc-to-LoRA" "1 second" performance
  - site:github.com SakanaAI doc-to-lora benchmark
- 期待: ソース, 実測データ, 論文での言及

---

## 【優先度中】実在性確認

### 15. **Audrey Tang × Sakana AI 協業**
- 検索キーワード案:
  - Audrey Tang Sakana AI
  - "Audrey Tang" "Sakana AI"
  - site:blog.sakana.ai Audrey Tang
  - site:audreyt.xyz Sakana
- 期待: ブログ, インタビュー, GitHub, 共同プレスリリース

### 16. **Audrey Tang transcript アーカイブ**
- 検索キーワード案:
  - site:github.com audreyt/transcript
  - site:sayit.archive.tw
  - audreyt transcript GitHub
  - sayit.tw Taiwan
- 期待: リポジトリの存在, データセット規模, 更新頻度

### 17. **台湾ディープフェイク広告規制 → TAIDE 使用 → 94% 減少**
- 検索キーワード案:
  - Taiwan deepfake TAIDE regulation
  - "TAIDE" deepfake detection "94%"
  - Taiwan "deepfake ad" regulation 2024 2025
  - site:audreyt.xyz TAIDE deepfake
- 期待: 一次ソース (台湾政府発表, Audrey Tang オフィシャル)

### 18. **ERC-8004 「Trustworthy AI Agents」**
- 検索キーワード案:
  - ERC-8004 specification ethereum
  - site:eips.ethereum.org "8004"
  - "Trustworthy AI Agents" ERC
- 期待: EIPS での正式定義

---

## 【優先度中】Sakana Doc-to-LoRA 公式情報

### 19. **Sakana Doc-to-LoRA 公式用途**
- リソース確認:
  - https://github.com/SakanaAI/doc-to-lora (README, ドキュメント)
  - site:blog.sakana.ai doc-to-lora
  - arXiv:2602.15902 (論文の評価対象)
  - HuggingFace Sakana モデル一覧
- 質問:
  - 公式は「個人ドキュメント数千件を LoRA 化」という用途を紹介しているか?
  - それとも SQuAD/DROP/ROPES 等 benchmark のみか?

---

## 検証出力形式

各項目につき:
```
N. <項目>: <実在 / 不明 / 部分的に実在>
- 根拠 URL: ...
- 補足: 何が確認できて何が確認できなかったか
```

最後に 400 字以内のサマリー（日本語）を付与。

**特に重要**: jdd-kami, Habermolt, OpenClaw, NemoClaw, 6-Pack of Care, 「2000 件 20 分」の 6 項目は複数検索クエリを試して念入りに調査。
