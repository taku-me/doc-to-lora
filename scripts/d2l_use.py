"""D2L をシンプルに使う CLI。

ドキュメントを 512-token チャンクに分割して内部化し、LoRA を 1 回だけ生成・キャッシュ。
あとは REPL で何度でも質問できる。Gemini レビュー (2026-05-17) で指摘された
「demo はチャンク分割していない」問題を `split_too_long_ctx` で解消した実装。

Usage:
    # REPL (load doc, then ask interactively)
    .venv/bin/python scripts/d2l_use.py \
        --checkpoint trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin \
        --doc data/paper_excerpts/sec_4_niah.md

    # One-shot
    .venv/bin/python scripts/d2l_use.py \
        --checkpoint trained_d2l/gemma_demo/checkpoint-80000/pytorch_model.bin \
        --doc path/to/your_doc.md \
        --question "..."
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ctx_to_lora.data.processing import (  # noqa: E402
    split_too_long_ctx,
    tokenize_ctx_text,
)
from ctx_to_lora.model_loading import get_tokenizer  # noqa: E402
from ctx_to_lora.modeling import hypernet  # noqa: E402
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel  # noqa: E402

# checkpoint state_dict が古い import path を参照するため alias (demo/app.py 参照)
sys.modules["ctx_to_lora.modeling_utils"] = hypernet


LISTING8 = (
    "Answer the following question. "
    "Output only the answer and do not output any other words.\n\n"
    "Question: {q}"
)

MAX_CHUNK_LEN = 512  # 学習時 args.yaml の max_ctx_chunk_len と一致
MIN_CHUNK_LEN = 25
DEFAULT_MAX_NEW_TOKENS = 128


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[ModulatedPretrainedModel, object, object]:
    print(f"[d2l] loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    use_flash = device.type == "cuda"
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
    print(f"[d2l] base={base_name} device={device}")
    return model, base_tokenizer, ctx_tokenizer


def internalize_doc(
    model: ModulatedPretrainedModel,
    ctx_tokenizer,
    doc_text: str,
    device: torch.device,
) -> int:
    """文書を 512-token チャンクに分割して内部化。返り値はチャンク数."""
    base_name = model.base_model.config.name_or_path
    tokenized = tokenize_ctx_text({"context": [doc_text]}, ctx_tokenizer)
    ctx_ids_flat = tokenized["ctx_ids"][0]
    print(f"[d2l] tokenized doc: {len(ctx_ids_flat)} tokens")

    chunked = split_too_long_ctx(
        {"ctx_ids": ctx_ids_flat},
        model_name_or_path=base_name,
        num_chunk_probs=None,
        max_chunk_len=MAX_CHUNK_LEN,
        min_chunk_len=MIN_CHUNK_LEN,
        max_num_split=None,
        is_train=False,
    )
    chunks = chunked["ctx_ids"]
    n_chunks = chunked["n_ctx_chunks"]
    print(f"[d2l] split into {n_chunks} chunk(s), sizes={[len(c) for c in chunks]}")

    chunk_tensors = [torch.tensor(c, dtype=torch.long, device=device) for c in chunks]
    attn_tensors = [torch.ones_like(t) for t in chunk_tensors]
    ctx_ids = torch.nn.utils.rnn.pad_sequence(
        chunk_tensors, batch_first=True, padding_value=0
    )
    ctx_attn_mask = torch.nn.utils.rnn.pad_sequence(
        attn_tensors, batch_first=True, padding_value=0
    )

    t0 = time.time()
    with torch.inference_mode():
        model._internalize_from_ids(ctx_ids=ctx_ids, ctx_attn_mask=ctx_attn_mask)
    print(f"[d2l] internalized in {time.time() - t0:.1f}s (LoRA cached)")
    return n_chunks


def _build_chat(base_tokenizer, content: str, device):
    enc = base_tokenizer.apply_chat_template(
        [
            {"role": "system", "content": ""},
            {"role": "user", "content": content},
        ],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    attn = attn.to(device) if attn is not None else torch.ones_like(input_ids)
    return input_ids, attn


def ask(
    model: ModulatedPretrainedModel,
    base_tokenizer,
    question: str,
    device: torch.device,
    max_new_tokens: int,
    use_listing8: bool,
    n_ctx_chunks: int,
) -> str:
    """キャッシュ済 LoRA を使って質問。input_ids は base のみ送る."""
    content = LISTING8.format(q=question) if use_listing8 else question
    input_ids, attn = _build_chat(base_tokenizer, content, device)

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(
            n_ctx_chunks=torch.tensor([n_ctx_chunks], device=device),
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.time() - t0
    gen = out[0][input_ids.shape[1] :]
    text = base_tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text, int(input_ids.shape[1]), int(gen.shape[0]), elapsed


def ask_base_only(
    model: ModulatedPretrainedModel,
    base_tokenizer,
    content: str,
    device: torch.device,
    max_new_tokens: int,
) -> tuple[str, int, int, float]:
    """LoRA を外して base_model.generate を直接呼ぶ。呼び出し側で model.reset() 済み前提."""
    input_ids, attn = _build_chat(base_tokenizer, content, device)
    t0 = time.time()
    with torch.inference_mode():
        out = model.base_model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.time() - t0
    gen = out[0][input_ids.shape[1] :]
    text = base_tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text, int(input_ids.shape[1]), int(gen.shape[0]), elapsed


def compare_modes(
    model: ModulatedPretrainedModel,
    base_tokenizer,
    ctx_tokenizer,
    question: str,
    doc_text: str,
    device: torch.device,
    max_new_tokens: int,
) -> None:
    """同じ質問を base / rag / d2l の 3 モードで回し、結果を並べて表示."""
    q = LISTING8.format(q=question)
    rag_content = f"{doc_text}\n\n{q}"

    model.reset()
    base_ans, b_in, b_out, b_t = ask_base_only(
        model, base_tokenizer, q, device, max_new_tokens
    )
    rag_ans, r_in, r_out, r_t = ask_base_only(
        model, base_tokenizer, rag_content, device, max_new_tokens
    )

    n_chunks = internalize_doc(model, ctx_tokenizer, doc_text, device)
    d2l_ans, d_in, d_out, d_t = ask(
        model, base_tokenizer, question, device, max_new_tokens, True, n_chunks
    )

    print(f"\n  ┌─[base] {b_in}tok→{b_out}tok in {b_t:.2f}s")
    print(f"  │  {base_ans}")
    print(f"  ├─[rag ] {r_in}tok→{r_out}tok in {r_t:.2f}s")
    print(f"  │  {rag_ans}")
    print(f"  └─[d2l ] {d_in}tok→{d_out}tok in {d_t:.2f}s")
    print(f"     {d2l_ans}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="例: trained_d2l/qwen_4b_d2l/checkpoint-20000/pytorch_model.bin",
    )
    p.add_argument("--doc", type=Path, required=True, help="内部化するテキストファイル")
    p.add_argument(
        "--question", type=str, default=None, help="一発質問 (省略時は REPL)"
    )
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument(
        "--raw",
        action="store_true",
        help="Listing 8 prompt を被せずに質問を素のまま送る",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="各質問について base / rag / d2l の 3 モードを並べて表示",
    )
    args = p.parse_args()

    if not args.checkpoint.exists():
        print(f"checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not args.doc.exists():
        print(f"doc not found: {args.doc}", file=sys.stderr)
        return 2

    device = _pick_device()
    model, base_tokenizer, ctx_tokenizer = load_model(args.checkpoint, device)

    doc_text = args.doc.read_text()
    print(f"[d2l] doc: {args.doc} ({len(doc_text)} chars)")

    use_listing8 = not args.raw

    if args.compare:
        if args.question is not None:
            compare_modes(
                model, base_tokenizer, ctx_tokenizer, args.question,
                doc_text, device, args.max_new_tokens,
            )
            return 0
        print("\n[d2l] COMPARE REPL ready. Type your question (empty line / Ctrl-D to quit).\n")
        while True:
            try:
                q = input("? ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                break
            compare_modes(
                model, base_tokenizer, ctx_tokenizer, q,
                doc_text, device, args.max_new_tokens,
            )
            print()
        return 0

    n_chunks = internalize_doc(model, ctx_tokenizer, doc_text, device)

    if args.question is not None:
        ans, in_t, out_t, t = ask(
            model, base_tokenizer, args.question, device,
            args.max_new_tokens, use_listing8, n_chunks,
        )
        print(f"[d2l] in={in_t}tok out={out_t}tok t={t:.2f}s")
        print(f"\n{ans}")
        return 0

    print("\n[d2l] REPL ready. Type your question (empty line / Ctrl-D to quit).\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            break
        ans, in_t, out_t, t = ask(
            model, base_tokenizer, q, device,
            args.max_new_tokens, use_listing8, n_chunks,
        )
        print(f"[d2l] in={in_t}tok out={out_t}tok t={t:.2f}s")
        print(f"\n{ans}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
