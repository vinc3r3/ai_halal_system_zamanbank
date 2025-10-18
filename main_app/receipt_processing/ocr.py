import warnings
warnings.filterwarnings("ignore", message=r".*validate_default.*", category=UserWarning)

import os
import sys
import argparse
import asyncio
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from llama_parse import LlamaParse

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OCR receipts with LlamaParse + prompt")
    p.add_argument("--src", required=True, type=Path, help="Input directory with receipts")
    p.add_argument("--out", required=True, type=Path, help="Output directory")
    p.add_argument("--env", default="./.env", type=Path, help="Path to .env with LLAMA_CLOUD_API_KEY")
    p.add_argument("--lang", default="ru", help="Language hint, e.g., ru/en")
    p.add_argument("--format", choices=["markdown", "json"], default="json", help="Result format to save")
    p.add_argument("--prompt", default=None, help="Inline parsing instruction")
    p.add_argument("--prompt-file", type=Path, help="Path to a .txt with parsing instruction")
    p.add_argument("--concurrency", type=int, default=6, help="Parallel parses")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    return p.parse_args()


# ---------- Files ----------
def collect_files(root: Path, recursive: bool) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".pdf"}
    if recursive:
        return [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]


# ---------- Parse ----------
async def parse_one(parser: LlamaParse, path: Path, out_dir: Path, fmt: str) -> None:
    # use async API for speed
    result = await parser.aparse(str(path))
    if fmt == "markdown":
        # join page markdowns
        text = "\n\n".join(page.md for page in result.pages)
        out_path = out_dir / f"{path.stem}.md"
        out_path.write_text(text, encoding="utf-8")
    else:
        # full JSON payload
        out_path = out_dir / f"{path.stem}.json"
        out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


async def main_async(args: argparse.Namespace) -> int:
    # load env and key
    load_dotenv(dotenv_path=args.env)
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        print("LLAMA_CLOUD_API_KEY is missing in env", file=sys.stderr)
        return 2

    # prompt
    prompt: Optional[str] = args.prompt
    if args.prompt_file and args.prompt_file.exists():
        prompt = args.prompt_file.read_text(encoding="utf-8")

    # default receipt-focused prompt if none supplied
    if not prompt:
        prompt = (
            "This is a purchase receipt. Extract only structured line items and totals. "
            "Return item name and price for each line. Also extract subtotal, tax/VAT, "
            "discounts, and final total. Ignore ads and survey text."
        )

    # initialize parser with agentic OCR + invoice preset
    parser = LlamaParse(
        # see docs: parse mode, model, high-res OCR, tables, and prompts
        parse_mode="parse_page_with_agent",
        model="openai-gpt-4-1-mini",
        high_res_ocr=True,
        adaptive_long_table=True,
        outlined_table_extraction=True,
        output_tables_as_HTML=False,
        preset="invoice",            # invoice/receipt tuned preset
        user_prompt=prompt,          # inject parsing instruction
        result_type=args.format,     # "markdown" or "json"
        language=args.lang,
    )

    files = collect_files(args.src, args.recursive)
    if not files:
        print("No input files found", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def _guarded(p: Path):
        async with sem:
            await parse_one(parser, p, args.out, args.format)

    await asyncio.gather(*[_guarded(f) for f in files])
    return 0


def main() -> None:
    args = build_args()
    rc = asyncio.run(main_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
