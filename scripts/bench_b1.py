"""B1: Doc-to-LoRA "神託体験" ベンチマーク

論文 (arXiv:2602.15902) の主張を「体感」する 3 モード比較:

- base: Gemma-2B-it に passage を見せず query だけ → ハルシネーション期待
- rag:  Gemma-2B-it に passage + query をプロンプト直貼り → input_tokens 大、正答
- d2l:  D2L で passage を LoRA 化 → query のみ送信 (input_tokens 小、正答)

Success Metric (論文準拠):
    "KV キャッシュを 1 トークンも使わずに、Gemma-2B が未学習の事実を即答する"

実行:
    # 前提: ato d2l デーモンが localhost:8765 で動作 (実 D2L バックエンド)
    uv run python scripts/bench_b1.py
    uv run python scripts/bench_b1.py --doc sec_4_niah  # 単一文書だけ

Output:
    1. コンソール: 質問ごとの 3 モード結果
    2. /Users/.../05_Projects/doc-to-lora/_logs/benchmark-YYYY-MM-DD.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
EXCERPTS_DIR = ROOT / "data" / "paper_excerpts"
QA_PATH = EXCERPTS_DIR / "qa_pairs.json"

OBSIDIAN_LOG_DIR = Path(
    "/Users/taku.me/Library/Mobile Documents/iCloud~md~obsidian/Documents/"
    "tkmych525@gmail.com/05_Projects/doc-to-lora/_logs"
)

DAEMON_URL = "http://127.0.0.1:8765"
TIMEOUT = httpx.Timeout(120.0, connect=5.0)


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def rouge_l_f1(pred: str, gold: str) -> float:
    """Word-level ROUGE-L F1. 論文 §5.1 と同じ指標 (簡易実装)."""
    p, g = _tokens(pred), _tokens(gold)
    if not p or not g:
        return 0.0
    m, n = len(p), len(g)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if p[i] == g[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return 2 * prec * rec / (prec + rec)


def token_overlap(pred: str, gold: str) -> float:
    """Bag-of-words token overlap (recall). gold token のうち pred に含まれる割合."""
    p = Counter(_tokens(pred))
    g = Counter(_tokens(gold))
    if not g:
        return 0.0
    common = sum((p & g).values())
    return common / sum(g.values())


def post(client: httpx.Client, path: str, payload: dict) -> dict:
    r = client.post(f"{DAEMON_URL}{path}", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def ensure_lora(client: httpx.Client, doc_id: str, text: str, source: Path) -> dict:
    """既存 LoRA があれば再利用、無ければ internalize."""
    listing = client.get(f"{DAEMON_URL}/loras", timeout=TIMEOUT).json()
    existing = {x["doc_id"]: x for x in listing.get("loras", [])}
    if doc_id in existing:
        print(f"  [reuse] LoRA exists for doc_id={doc_id}")
        return existing[doc_id]
    print(f"  [internalize] doc_id={doc_id} text_len={len(text)} ...")
    return post(
        client,
        "/internalize",
        {"doc_id": doc_id, "text": text, "source_path": str(source)},
    )


def run_question(
    client: httpx.Client, passage: str, doc_id: str, question: str
) -> dict[str, dict]:
    """1 問につき 3 モード (base/rag/d2l) を回す."""
    out: dict[str, dict] = {}
    for mode in ("base", "rag", "d2l"):
        payload = {"mode": mode, "query": question}
        if mode == "rag":
            payload["passage"] = passage
        if mode == "d2l":
            payload["doc_id"] = doc_id
        out[mode] = post(client, "/bench/answer", payload)
    return out


def render_question_block(
    qa: dict, results: dict[str, dict], idx: int
) -> tuple[str, dict]:
    """1 問分の Markdown と集計値 (mode → list of metrics) を返す."""
    q = qa["q"]
    gold = qa["a"]
    rows = []
    metrics = {}
    for mode in ("base", "rag", "d2l"):
        r = results[mode]
        rouge = rouge_l_f1(r["short_answer"], gold)
        overlap = token_overlap(r["short_answer"], gold)
        rows.append(
            f"| **{mode}** | {r['input_tokens']:>5d} | {r['output_tokens']:>4d} "
            f"| {r['elapsed_s']:>5.2f} s | {rouge:>4.2f} | {overlap:>4.2f} "
            f"| `{r['short_answer'][:80].replace(chr(10), ' ⏎ ')}` |"
        )
        metrics[mode] = {
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "elapsed_s": r["elapsed_s"],
            "rouge_l_f1": rouge,
            "token_overlap": overlap,
        }
    block = (
        f"### Q{idx}: {q}\n\n"
        f"**Gold answer:** `{gold}`\n\n"
        f"| mode | input_tok | out_tok | elapsed | ROUGE-L | overlap | answer |\n"
        f"|---|---:|---:|---:|---:|---:|---|\n"
        + "\n".join(rows)
        + "\n"
    )
    return block, metrics


def render_summary(per_doc_metrics: dict[str, list[dict[str, dict]]]) -> str:
    """全文書合算の mode 別平均を表に."""
    lines = ["## Summary (mean across all questions)\n"]
    lines.append("| mode | mean input_tok | mean elapsed | mean ROUGE-L | mean overlap | n |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    agg = {"base": [], "rag": [], "d2l": []}
    for _, qs in per_doc_metrics.items():
        for q_metrics in qs:
            for mode in ("base", "rag", "d2l"):
                agg[mode].append(q_metrics[mode])
    for mode in ("base", "rag", "d2l"):
        ms = agg[mode]
        if not ms:
            continue
        n = len(ms)
        avg_in = sum(m["input_tokens"] for m in ms) / n
        avg_t = sum(m["elapsed_s"] for m in ms) / n
        avg_rouge = sum(m["rouge_l_f1"] for m in ms) / n
        avg_over = sum(m["token_overlap"] for m in ms) / n
        lines.append(
            f"| **{mode}** | {avg_in:>7.1f} | {avg_t:>5.2f} s "
            f"| {avg_rouge:>4.2f} | {avg_over:>4.2f} | {n} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--doc", help="単一 doc_id だけ実行 (省略時は全 docs)")
    p.add_argument("--max-q", type=int, default=0, help="各 doc あたり最大質問数 (0=全部)")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="出力 md パス (default: Obsidian 05_Projects/doc-to-lora/_logs/benchmark-YYYY-MM-DD.md)",
    )
    args = p.parse_args()

    qa = json.loads(QA_PATH.read_text())
    docs = qa["datasets"]
    if args.doc:
        docs = [d for d in docs if d["doc_id"] == args.doc]
        if not docs:
            print(f"unknown doc_id: {args.doc}", file=sys.stderr)
            return 2

    print(f"[B1] benchmarking {len(docs)} doc(s)")

    md_blocks: list[str] = []
    per_doc_metrics: dict[str, list[dict[str, dict]]] = {}
    t_total0 = time.time()

    with httpx.Client() as client:
        h = client.get(f"{DAEMON_URL}/health", timeout=TIMEOUT)
        h.raise_for_status()
        info = h.json()
        if info.get("backend") != "real":
            print(
                f"[B1] WARN: backend={info.get('backend')!r}, expected 'real'. "
                f"Set ATO_D2L_MOCK=0 and restart the daemon.",
                file=sys.stderr,
            )
            return 3
        print(f"[B1] daemon healthy, backend={info['backend']}, "
              f"existing_loras={info['loras_count']}")

        for d in docs:
            doc_id = d["doc_id"]
            src_path = ROOT / d["source"]
            if not src_path.exists():
                print(f"[B1] missing source: {src_path}", file=sys.stderr)
                continue
            passage = src_path.read_text()
            print(f"\n[B1] === doc_id={doc_id} src={src_path.name} ===")

            ensure_lora(client, doc_id, passage, src_path)

            qa_pairs = d["qa_pairs"]
            if args.max_q > 0:
                qa_pairs = qa_pairs[: args.max_q]
            doc_blocks = [
                f"## Doc: `{doc_id}` ({d['approx_tokens']} tok approx)\n\n"
                f"Source: `{d['source']}`\n"
            ]
            doc_metrics: list[dict[str, dict]] = []
            for i, qa_pair in enumerate(qa_pairs, 1):
                print(f"  [Q{i}/{len(qa_pairs)}] {qa_pair['q'][:60]}")
                results = run_question(client, passage, doc_id, qa_pair["q"])
                block, metrics = render_question_block(qa_pair, results, i)
                doc_blocks.append(block)
                doc_metrics.append(metrics)
            md_blocks.append("\n".join(doc_blocks))
            per_doc_metrics[doc_id] = doc_metrics

    elapsed_total = time.time() - t_total0
    today = date.today().isoformat()
    out_path = args.output or (OBSIDIAN_LOG_DIR / f"benchmark-{today}.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_q = sum(len(v) for v in per_doc_metrics.values())
    header = (
        f"---\n"
        f"status: ACTIVE\n"
        f"project-type: DEV\n"
        f"updated: {today}\n"
        f"---\n\n"
        f"# D2L B1 Benchmark — {today}\n\n"
        f"Source: arXiv:2602.15902 (Doc-to-LoRA: Learning to Instantly Internalize Contexts)\n\n"
        f"**Setup:** Gemma-2B-it (gemma_demo checkpoint) on Apple Silicon M2 Max 64GB / MPS\n\n"
        f"**Benchmarked:** {len(per_doc_metrics)} doc(s), {n_q} question(s) × 3 modes "
        f"(total {n_q * 3} generations) in {elapsed_total:.1f}s\n\n"
        f"## Modes\n\n"
        f"- **base**: Gemma-2B-it (no LoRA), Listing 8 prompt with query only\n"
        f"- **rag**: Gemma-2B-it (no LoRA), passage + query in prompt (oracle context)\n"
        f"- **d2l**: Gemma-2B-it + D2L LoRA (passage internalized into weights), query only\n\n"
        f"## Success Metric\n\n"
        f"> KV キャッシュを 1 トークンも使わずに、Gemma-2B が未学習の事実を即答する\n\n"
        f"= base mode が当てられない事実を、d2l mode が **input_tokens がほぼ base と同じまま** で当てられること。"
        f"rag mode は精度のオラクルだが input_tokens が文書サイズに比例して膨らむ。\n\n"
    )
    body = render_summary(per_doc_metrics) + "\n---\n\n" + "\n\n---\n\n".join(md_blocks)
    out_path.write_text(header + body)
    print(f"\n[B1] report written to {out_path}")
    print(f"[B1] total elapsed: {elapsed_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
