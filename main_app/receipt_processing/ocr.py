#!/usr/bin/env python3
# ocr.py — prompt-driven receipt OCR to Markdown or JSON (LlamaParse)

import warnings
warnings.filterwarnings("ignore", message=r".*validate_default.*", category=UserWarning)

import os
import sys
import json
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from llama_parse import LlamaParse


# ---------------- CLI ----------------
def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OCR receipts with LlamaParse + prompt")
    p.add_argument("--src", required=True, type=Path, help="Input directory with receipts")
    p.add_argument("--out", required=True, type=Path, help="Output directory")
    p.add_argument("--env", default="./.env", type=Path, help="Path to .env with LLAMA_CLOUD_API_KEY")
    p.add_argument("--lang", default="ru", help="OCR language hint, e.g., ru/en")
    p.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Result format")
    p.add_argument("--prompt", default=None, help="Inline parsing instruction")
    p.add_argument("--prompt-file", type=Path, help="Path to a .txt with parsing instruction")
    p.add_argument("--concurrency", type=int, default=6, help="Parallel parses")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    p.add_argument("--premium", action="store_true", help="Enable premium mode OCR")
    p.add_argument("--model", default="openai-gpt-4-1-mini", help="LLM for synthesis")
    p.add_argument("--preset", default="invoice", help="LlamaParse preset")
    return p.parse_args()


# ---------------- Files ----------------
def collect_files(root: Path, recursive: bool) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".pdf"}
    if recursive:
        return [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]


# ---------------- Helpers ----------------
def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file and args.prompt_file.exists():
        return args.prompt_file.read_text(encoding="utf-8")
    # fallback: minimal but strict Markdown schema instruction
    return (
        "You are a precise receipt parser for Russian/English receipts.\n"
        "Return ONLY this Markdown:\n\n"
        "# Transaction: <transaction_id>\n\n"
        "| item | pcs | amount_money | category | category_ru |\n"
        "|------|-----|---------------|-----------|--------------|\n"
        "| <item> | <int> | <float> | <english category> | <russian category> |\n\n"
        "Rules: infer category, default pcs=1, decimals use dot, no currency symbols, "
        "ignore ads/QR, no text outside the table."
    )


def extract_markdown(res) -> str:
    # Prefer synthesized document-level markdown if present.
    for attr in ("text", "markdown", "md"):
        if hasattr(res, attr) and isinstance(getattr(res, attr), str) and getattr(res, attr).strip():
            return getattr(res, attr)
    # Some SDKs return list-like results
    if isinstance(res, (list, tuple)) and res:
        return extract_markdown(res[0])
    # Fallback to page concatenation
    if hasattr(res, "pages") and res.pages:
        parts = []
        for p in res.pages:
            if hasattr(p, "md") and isinstance(p.md, str) and p.md.strip():
                parts.append(p.md)
            elif hasattr(p, "text") and isinstance(p.text, str) and p.text.strip():
                parts.append(p.text)
        if parts:
            return "\n\n".join(parts)
    # Last resort: JSON dump
    try:
        return json.dumps(res.model_dump(), ensure_ascii=False, indent=2)  # type: ignore[attr-defined]
    except Exception:
        return ""


def extract_json_text(res) -> str:
    # Standard path
    if hasattr(res, "model_dump_json"):
        return res.model_dump_json(indent=2)  # type: ignore[attr-defined]
    # Some SDKs return dict-like
    try:
        if hasattr(res, "model_dump"):
            return json.dumps(res.model_dump(), ensure_ascii=False, indent=2)  # type: ignore[attr-defined]
    except Exception:
        pass
    # Pages fallback
    if hasattr(res, "pages") and res.pages:
        try:
            return json.dumps(res.pages[0].model_dump(), ensure_ascii=False, indent=2)  # type: ignore[attr-defined]
        except Exception:
            pass
    # Generic fallback
    try:
        return json.dumps(res, ensure_ascii=False, indent=2)  # may fail if not serializable
    except Exception:
        return ""


# ---------------- Parse one ----------------
async def parse_one(parser: LlamaParse, path: Path, out_dir: Path, fmt: str) -> None:
    res = await parser.aparse(str(path))
    if fmt == "markdown":
        md = extract_markdown(res)
        out_path = out_dir / f"{path.stem}.md"
        out_path.write_text(md, encoding="utf-8")
    else:
        js = extract_json_text(res)
        out_path = out_dir / f"{path.stem}.json"
        out_path.write_text(js, encoding="utf-8")


# ---------------- Main async ----------------
async def main_async(args: argparse.Namespace) -> int:
    load_dotenv(dotenv_path=args.env)
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        print("LLAMA_CLOUD_API_KEY is missing in env", file=sys.stderr)
        return 2

    prompt_text = read_prompt(args)

    parser = LlamaParse(
        model=args.model,
        high_res_ocr=True,
        preset=args.preset,
        language=args.lang,
        result_type=args.format,   # "markdown" or "json"
        user_prompt=prompt_text,   # synthesized output driven by prompt
        premium_mode=args.premium, # optional
        # IMPORTANT: do NOT set parse_mode="parse_page_with_agent"
    )

    files = collect_files(args.src, args.recursive)
    if not files:
        print("No input files found", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def _guarded(p: Path):
        async with sem:
            try:
                await parse_one(parser, p, args.out, args.format)
            except Exception as e:
                err_path = args.out / f"{p.stem}.error.txt"
                err_path.write_text(f"{type(e).__name__}: {e}", encoding="utf-8")

    await asyncio.gather(*(_guarded(f) for f in files))
    return 0


def main() -> None:
    args = build_args()
    rc = asyncio.run(main_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
