"""B1 ベンチマーク (Qwen3-4B-Instruct-2507 版) — daemon 不要・in-process 実行

bench_b1.py の Gemma-2B + ato daemon 構成を、Qwen3-4B + ローカル checkpoint 直接ロードに
置き換えたもの。同じ 27 質問 × 3 モード (base/rag/d2l) を回し、Obsidian _logs/ に出力する。

Setup:
    base   : Qwen3-4B-Instruct-2507 (no LoRA), Listing 8 prompt with query only
    rag    : Qwen3-4B-Instruct-2507 (no LoRA), passage + Listing 8 prompt
    d2l    : Qwen3-4B + D2L LoRA (passage internalized into weights), query only

Checkpoint:
    trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin

Memory budget (M2 Max 64GB, MPS, bf16):
    base_model  ~8GB + ctx_encoder ~8GB + KV cache ≲1GB = ~17GB

Run:
    uv run python scripts/bench_b1_qwen.py
    uv run python scripts/bench_b1_qwen.py --doc sec_4_niah
    uv run python scripts/bench_b1_qwen.py --max-q 3   # 各 doc 先頭 3 問だけ

Output:
    {OBSIDIAN_LOG_DIR}/benchmark-YYYY-MM-DD-qwen.md
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

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ctx_to_lora.data.processing import tokenize_ctx_text  # noqa: E402
from ctx_to_lora.model_loading import get_tokenizer  # noqa: E402
from ctx_to_lora.modeling import hypernet  # noqa: E402
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel  # noqa: E402

# checkpoint state_dict が古い import path を参照するため alias を貼る (demo/app.py 参照)
sys.modules["ctx_to_lora.modeling_utils"] = hypernet

EXCERPTS_DIR = ROOT / "data" / "paper_excerpts"
QA_PATH = EXCERPTS_DIR / "qa_pairs.json"
CHECKPOINT_PATH = ROOT / "trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin"
MODEL_LABEL = "Qwen3-4B-Instruct-2507 (qwen_4b_d2l/checkpoint-20000)"

OBSIDIAN_LOG_DIR = (
    Path.home() / "ObsidianVaultSync" / "takumi-obsidian" / "05_Projects" / "doc-to-lora" / "_logs"
)

LISTING8 = "Answer the following question. Output only the answer and do not output any other words.\n\nQuestion: {q}"

MAX_NEW_TOKENS = 64

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def rouge_l_f1(pred: str, gold: str) -> float:
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
    p = Counter(_tokens(pred))
    g = Counter(_tokens(gold))
    if not g:
        return 0.0
    common = sum((p & g).values())
    return common / sum(g.values())


def build_chat_inputs(
    tokenizer, content: str, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Listing 8 / RAG プロンプトを chat_template でトークナイズ。
    Qwen は pad==eos なので attention_mask を明示的に組んで返す."""
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    attn = attn.to(device) if attn is not None else torch.ones_like(input_ids)
    return input_ids, attn


def tokenize_passage(
    passage: str, ctx_tokenizer, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """passage を ctx_encoder 用にトークナイズし、(ctx_ids, ctx_attn_mask) を返す."""
    tokenized = tokenize_ctx_text({"context": [passage]}, ctx_tokenizer)
    ctx_ids = tokenized["ctx_ids"]
    ctx_ids = [torch.tensor(x, dtype=torch.long, device=device) for x in ctx_ids]
    ctx_attn_mask = [torch.ones_like(ids) for ids in ctx_ids]
    ctx_ids = torch.nn.utils.rnn.pad_sequence(
        ctx_ids, batch_first=True, padding_value=0
    )
    ctx_attn_mask = torch.nn.utils.rnn.pad_sequence(
        ctx_attn_mask, batch_first=True, padding_value=0
    )
    return ctx_ids, ctx_attn_mask


def generate_base(
    model, tokenizer, content: str, device
) -> tuple[str, int, int, float]:
    """base / rag モード: LoRA 無し (reset 状態) で base_model.generate を呼ぶ。
    呼び出し側で model.reset() 済を前提."""
    input_ids, attn = build_chat_inputs(tokenizer, content, device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.base_model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    elapsed = time.time() - t0
    gen = out[0][input_ids.shape[1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text, int(input_ids.shape[1]), int(gen.shape[0]), elapsed


def generate_d2l(
    model, tokenizer, ctx_tokenizer, query: str, passage: str, device
) -> tuple[str, int, int, float]:
    """d2l モード: passage を ctx_encoder で LoRA 化し、query (Listing 8) のみ送る。
    呼び出し側で model.patch_lora_forward() 済を前提."""
    ctx_ids, ctx_attn_mask = tokenize_passage(passage, ctx_tokenizer, device)
    input_ids, _ = build_chat_inputs(tokenizer, LISTING8.format(q=query), device)
    scalers = torch.tensor([1.0], dtype=torch.float32, device=device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            ctx_ids=ctx_ids,
            ctx_attn_mask=ctx_attn_mask,
            n_ctx_chunks=torch.tensor([len(ctx_ids)], device=device),
            scalers=scalers,
            bias_scaler=1.0,
            input_ids=input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    elapsed = time.time() - t0
    gen = out[0][input_ids.shape[1] :]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text, int(input_ids.shape[1]), int(gen.shape[0]), elapsed


def run_question(
    model, tokenizer, ctx_tokenizer, passage: str, question: str, device
) -> dict[str, dict]:
    results: dict[str, dict] = {}

    # base / rag は LoRA 不要 → 一度 reset し、down_proj.forward を元の Linear に戻す
    model.reset()

    text, in_tok, out_tok, elapsed = generate_base(
        model, tokenizer, LISTING8.format(q=question), device
    )
    results["base"] = {
        "short_answer": text,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "elapsed_s": elapsed,
    }

    rag_content = f"{passage}\n\n{LISTING8.format(q=question)}"
    text, in_tok, out_tok, elapsed = generate_base(
        model, tokenizer, rag_content, device
    )
    results["rag"] = {
        "short_answer": text,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "elapsed_s": elapsed,
    }

    # d2l: LoRA forward を再 patch してから modulated 経路で生成
    model.patch_lora_forward()
    text, in_tok, out_tok, elapsed = generate_d2l(
        model, tokenizer, ctx_tokenizer, question, passage, device
    )
    results["d2l"] = {
        "short_answer": text,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "elapsed_s": elapsed,
    }
    return results


def render_question_block(
    qa: dict, results: dict[str, dict], idx: int
) -> tuple[str, dict]:
    q = qa["q"]
    gold = qa["a"]
    rows = []
    metrics = {}
    for mode in ("base", "rag", "d2l"):
        r = results[mode]
        rouge = rouge_l_f1(r["short_answer"], gold)
        overlap = token_overlap(r["short_answer"], gold)
        ans_short = r["short_answer"][:80].replace("\n", " ⏎ ")
        rows.append(
            f"| **{mode}** | {r['input_tokens']:>5d} | {r['output_tokens']:>4d} "
            f"| {r['elapsed_s']:>5.2f} s | {rouge:>4.2f} | {overlap:>4.2f} "
            f"| `{ans_short}` |"
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
        f"|---|---:|---:|---:|---:|---:|---|\n" + "\n".join(rows) + "\n"
    )
    return block, metrics


def render_summary(per_doc_metrics: dict[str, list[dict[str, dict]]]) -> str:
    lines = ["## Summary (mean across all questions)\n"]
    lines.append(
        "| mode | mean input_tok | mean elapsed | mean ROUGE-L | mean overlap | n |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    agg = {"base": [], "rag": [], "d2l": []}
    for qs in per_doc_metrics.values():
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


def load_model(device: torch.device) -> tuple[ModulatedPretrainedModel, object, object]:
    print(f"[B1-Qwen] loading checkpoint: {CHECKPOINT_PATH}")
    if not CHECKPOINT_PATH.exists():
        print(
            f"[B1-Qwen] FATAL: checkpoint not found: {CHECKPOINT_PATH}", file=sys.stderr
        )
        sys.exit(2)
    state_dict = torch.load(CHECKPOINT_PATH, weights_only=False, map_location="cpu")
    use_flash = device.type == "cuda"  # MPS/CPU は flash-attn 非対応
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict,
        train=False,
        use_flash_attn=use_flash,
        use_sequence_packing=False,
    )
    model = model.to(device).to(torch.bfloat16)
    model.eval()
    base_name = model.base_model.config.name_or_path
    ctx_name = model.ctx_encoder_args.ctx_encoder_model_name_or_path or base_name
    base_tokenizer = get_tokenizer(base_name, train=False)
    ctx_tokenizer = get_tokenizer(ctx_name, train=False)
    print(f"[B1-Qwen] base={base_name} ctx_encoder={ctx_name} device={device}")
    return model, base_tokenizer, ctx_tokenizer


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--doc", help="単一 doc_id だけ実行 (省略時は全 docs)")
    p.add_argument(
        "--max-q", type=int, default=0, help="各 doc あたり最大質問数 (0=全部)"
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"出力 md パス (default: {OBSIDIAN_LOG_DIR}/benchmark-YYYY-MM-DD-qwen.md)",
    )
    args = p.parse_args()

    qa = json.loads(QA_PATH.read_text())
    docs = qa["datasets"]
    if args.doc:
        docs = [d for d in docs if d["doc_id"] == args.doc]
        if not docs:
            print(f"unknown doc_id: {args.doc}", file=sys.stderr)
            return 2

    device = _pick_device()
    model, base_tokenizer, ctx_tokenizer = load_model(device)

    md_blocks: list[str] = []
    per_doc_metrics: dict[str, list[dict[str, dict]]] = {}
    t_total0 = time.time()

    for d in docs:
        doc_id = d["doc_id"]
        src_path = ROOT / d["source"]
        if not src_path.exists():
            print(f"[B1-Qwen] missing source: {src_path}", file=sys.stderr)
            continue
        passage = src_path.read_text()
        print(f"\n[B1-Qwen] === doc_id={doc_id} src={src_path.name} ===")

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
            results = run_question(
                model, base_tokenizer, ctx_tokenizer, passage, qa_pair["q"], device
            )
            block, metrics = render_question_block(qa_pair, results, i)
            doc_blocks.append(block)
            doc_metrics.append(metrics)
            for mode in ("base", "rag", "d2l"):
                r = results[mode]
                print(
                    f"    {mode}: in={r['input_tokens']} out={r['output_tokens']} "
                    f"t={r['elapsed_s']:.2f}s -> {r['short_answer'][:60]!r}"
                )
        md_blocks.append("\n".join(doc_blocks))
        per_doc_metrics[doc_id] = doc_metrics

    elapsed_total = time.time() - t_total0
    today = date.today().isoformat()
    out_path = args.output or (OBSIDIAN_LOG_DIR / f"benchmark-{today}-qwen.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_q = sum(len(v) for v in per_doc_metrics.values())
    header = (
        f"---\n"
        f"status: ACTIVE\n"
        f"project-type: DEV\n"
        f"updated: {today}\n"
        f"---\n\n"
        f"# D2L B1 Benchmark (Qwen) — {today}\n\n"
        f"Source: arXiv:2602.15902 (Doc-to-LoRA)\n\n"
        f"**Setup:** {MODEL_LABEL} on Apple Silicon M2 Max 64GB / MPS / bf16\n\n"
        f"**Benchmarked:** {len(per_doc_metrics)} doc(s), {n_q} question(s) × 3 modes "
        f"(total {n_q * 3} generations) in {elapsed_total:.1f}s\n\n"
        f"## Modes\n\n"
        f"- **base**: Qwen3-4B-Instruct-2507 (no LoRA), Listing 8 prompt with query only\n"
        f"- **rag**: Qwen3-4B-Instruct-2507 (no LoRA), passage + Listing 8 prompt (oracle context)\n"
        f"- **d2l**: Qwen3-4B + D2L LoRA (passage internalized into weights), Listing 8 prompt with query only\n\n"
        f"## Success Metric\n\n"
        f"> KV キャッシュを 1 トークンも使わずに、Qwen-4B が未学習の事実を即答する\n\n"
        f"= base mode が当てられない事実を、d2l mode が **input_tokens がほぼ base と同じまま** で当てられること。"
        f"rag mode は精度のオラクルだが input_tokens が文書サイズに比例して膨らむ。\n\n"
    )
    body = render_summary(per_doc_metrics) + "\n---\n\n" + "\n\n---\n\n".join(md_blocks)
    out_path.write_text(header + body)
    print(f"\n[B1-Qwen] report written to {out_path}")
    print(f"[B1-Qwen] total elapsed: {elapsed_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
