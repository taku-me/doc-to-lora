"""日記/transcript 群から D2L 学習用 QA データを自動生成する。

Ollama (qwen3:8b 等) に各文書から JSON 配列の QA ペアを生成させ、Sakana の
学習データ形式に近い jsonl で出力する。

出力 jsonl の 1 行 (1 QA ペア):
    {"context": <文書本文>, "prompt": <質問>, "response": <回答>, "source": <ファイル名>}

Usage:
    # smoke test (1 file, qwen3:8b)
    .venv/bin/python scripts/d2l_gen_qa.py \\
        --input <diary.md> \\
        --output /tmp/qa_smoke.jsonl \\
        --model qwen3:8b \\
        --n-qa 5

    # 本走 (directory 全部)
    .venv/bin/python scripts/d2l_gen_qa.py \\
        --input-dir /path/to/diaries \\
        --pattern '20*-*-*.md' \\
        --output data/raw_datasets/takumi_diary_qa.jsonl \\
        --model qwen3.6:35b-a3b-q8_0 \\
        --n-qa 5
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
import time
from pathlib import Path

import httpx

OLLAMA_URL = "http://127.0.0.1:11435"
DEFAULT_TIMEOUT = httpx.Timeout(180.0, connect=5.0)

PROMPT_TEMPLATE = """\
以下の文書 (日記または対話記録) から、その内容を理解しているか確認できる質問と回答のペアを {n} 個作成してください。

要件:
- 質問は文書中の事実・思考・固有名詞・概念に基づく
- 回答は文書の記述に即して簡潔に (1-2 文)
- 質問の種類を多様に: (a) 事実の確認 (誰が・いつ・どこで・何を), (b) 考察の要約, (c) 用語の意味, (d) 関連性
- 文書外の知識で答えられる一般常識クイズは避ける
- 質問と回答は日本語

出力は必ず JSON 配列のみ。説明文・コードブロックは不要:
[
  {{"q": "...", "a": "..."}},
  ...
]

# 文書

{context}
"""


def call_ollama(model: str, prompt: str, client: httpx.Client) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 2048},
    }
    r = client.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()["response"]


JSON_BLOCK = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)


def parse_qa_array(text: str) -> list[dict]:
    """LLM の応答から JSON 配列を抽出。説明文混入や ``` 包みにも耐性を持たせる."""
    text = text.strip()
    # 1) そのまま JSON か
    try:
        out = json.loads(text)
        if isinstance(out, list):
            return out
    except json.JSONDecodeError:
        pass
    # 2) ``` 包み剥がし
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            out = json.loads(text)
            if isinstance(out, list):
                return out
        except json.JSONDecodeError:
            pass
    # 3) 文中から [...] を抽出
    m = JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return []


def gen_for_file(
    path: Path, model: str, n_qa: int, client: httpx.Client
) -> tuple[str, list[dict]]:
    context = path.read_text()
    prompt = PROMPT_TEMPLATE.format(n=n_qa, context=context)
    raw = call_ollama(model, prompt, client)
    qa = parse_qa_array(raw)
    # 弾く: 必須 key が欠ける / 空文字
    cleaned = [
        {"q": x["q"].strip(), "a": x["a"].strip()}
        for x in qa
        if isinstance(x, dict)
        and isinstance(x.get("q"), str)
        and isinstance(x.get("a"), str)
        and x["q"].strip()
        and x["a"].strip()
    ]
    return context, cleaned


def collect_files(args) -> list[Path]:
    if args.input:
        return [Path(args.input)]
    files = []
    skipped_short = 0
    for p in Path(args.input_dir).rglob("*.md"):
        if args.pattern and not fnmatch.fnmatch(p.name, args.pattern):
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size < args.min_chars:
            skipped_short += 1
            continue
        files.append(p)
    if skipped_short:
        print(
            f"[gen_qa] skipped {skipped_short} short files (< {args.min_chars} chars)"
        )
    return sorted(files)


def load_done_sources(output_path: Path) -> set[str]:
    """resume 用: 既存出力 jsonl にすでに含まれる source 集合を返す."""
    if not output_path.exists():
        return set()
    done = set()
    with output_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add(r.get("source", ""))
            except json.JSONDecodeError:
                continue
    return done


def main() -> int:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="単一ファイル")
    src.add_argument("--input-dir", type=Path, help="ディレクトリ (.md を再帰検索)")
    p.add_argument(
        "--pattern",
        default="20*-*-*.md",
        help="ファイル名グロブ (default: 20*-*-*.md = 日付ファイルのみ)",
    )
    p.add_argument("--output", type=Path, required=True, help="出力 jsonl")
    p.add_argument("--model", default="qwen3:8b", help="Ollama モデル名")
    p.add_argument("--n-qa", type=int, default=5, help="1 ファイルあたり QA 数")
    p.add_argument(
        "--limit", type=int, default=0, help="最大ファイル数 (0=無制限、smoke test 用)"
    )
    p.add_argument(
        "--min-chars", type=int, default=0,
        help="文書のバイト数下限 (これ以下はスキップ)。例: 500"
    )
    p.add_argument(
        "--resume", action="store_true",
        help="既存出力 jsonl にある source はスキップして残りだけ追記",
    )
    args = p.parse_args()

    files = collect_files(args)
    if args.limit > 0:
        files = files[: args.limit]

    done_sources: set[str] = set()
    write_mode = "w"
    if args.resume:
        done_sources = load_done_sources(args.output)
        write_mode = "a"
        before = len(files)
        files = [f for f in files if str(f) not in done_sources]
        print(
            f"[gen_qa] resume: {len(done_sources)} already done, "
            f"{before - len(files)} skipped, {len(files)} to do"
        )

    print(f"[gen_qa] {len(files)} file(s) to process, model={args.model}, n_qa={args.n_qa}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_total_qa = 0
    n_total_files = 0
    t0 = time.time()
    with httpx.Client() as client, args.output.open(write_mode) as f:
        for i, path in enumerate(files, 1):
            t = time.time()
            try:
                context, qa = gen_for_file(path, args.model, args.n_qa, client)
            except Exception as e:
                print(f"  [{i}/{len(files)}] {path.name} FAILED: {e}", file=sys.stderr)
                continue
            for x in qa:
                f.write(
                    json.dumps(
                        {
                            "context": context,
                            "prompt": x["q"],
                            "response": x["a"],
                            "source": str(path),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            n_total_qa += len(qa)
            n_total_files += 1
            print(
                f"  [{i}/{len(files)}] {path.name}: {len(qa)} QA ({time.time() - t:.1f}s)"
            )

    elapsed = time.time() - t0
    print(
        f"\n[gen_qa] done: {n_total_files} files, {n_total_qa} QA pairs in {elapsed:.1f}s"
    )
    print(f"[gen_qa] output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
