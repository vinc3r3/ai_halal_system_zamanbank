import warnings
warnings.filterwarnings("ignore", message=r".*validate_default.*", category=UserWarning)

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List
import time
import uuid

from dotenv import load_dotenv
from pydantic import BaseModel, Field, conint, confloat

# SDK
from llama_cloud_services import LlamaExtract
from llama_cloud import ExtractConfig, ExtractMode, ExtractTarget, ChunkMode
from llama_cloud.core.api_error import ApiError


# ---------- schema ----------
class LineItem(BaseModel):
    item: str = Field(description="Product or service name in the receipt line.")
    pcs: confloat(ge=0) = Field(default=1, description="Quantity; default 1 if missing.")
    amount_money_per_pc: confloat(ge=0) = Field(
        description="Unit price; numeric only, use dot for decimals, no currency sign."
    )
    amount_money: confloat(ge=0) = Field(
        description="Total line amount; numeric only, use dot for decimals, no currency sign."
    )
    category: str = Field(
        description=(
            "Category in English; must correspond to one of the following:\n"
            "• Utilities & Housing — payments for rent, water, electricity, heating, maintenance fees.\n"
            "• Home & Furniture — household items, decor, furniture, cleaning supplies.\n"
            "• Pocket Money — small personal or daily expenses, spontaneous cash spending.\n"
            "• Groceries — food and beverages purchased in stores and supermarkets.\n"
            "• Transport — public transit, taxi, fuel, parking, or car maintenance.\n"
            "• Dining & Cafes — eating out: restaurants, cafes, delivery services.\n"
            "• Entertainment — movies, games, events, subscriptions, leisure activities.\n"
            "• Communication — mobile plans, internet, subscriptions for connectivity.\n"
            "• Health — medicines, clinics, dental care, supplements, health insurance.\n"
            "• Car Expenses — repairs, maintenance, car wash, accessories.\n"
            "• Sports — gym memberships, equipment, sportswear, training.\n"
            "• Children — kids’ education, toys, clothes, activities.\n"
            "• Travel — tickets, accommodation, tours, vacation-related costs.\n"
            "• Clothing — apparel, shoes, accessories.\n"
            "• Beauty — cosmetics, salons, personal care services.\n"
            "• Gifts — presents for others, celebrations, holidays."
        )
    )
    category_ru: str = Field(
        description=(
            "Category in Russian; paired with the English category above:\n"
            "• ЖКХ → Utilities & Housing\n"
            "• Все для дома → Home & Furniture\n"
            "• Карманные → Pocket Money\n"
            "• Продукты → Groceries\n"
            "• Транспорт → Transport\n"
            "• Еда → Dining & Cafes\n"
            "• Развлечения → Entertainment\n"
            "• Связь → Communication\n"
            "• Здоровье → Health\n"
            "• Авто → Car Expenses\n"
            "• Спорт → Sports\n"
            "• Дети → Children\n"
            "• Путешествия → Travel\n"
            "• Одежда → Clothing\n"
            "• Красота → Beauty\n"
            "• Подарки → Gifts"
        )
    )

class Receipt(BaseModel):
    transaction_id: str = Field(description="Receipt/transaction/check number")
    items: List[LineItem]


# ---------- io ----------
def find_files(root: Path, recursive: bool) -> List[Path]:
    exts = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    if recursive:
        return [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts]

def write_csv(receipts: List[BaseModel], out_path: Path) -> None:
    """Append new receipt samples to CSV, create header only if file is new."""
    import csv
    # ensure directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = out_path.exists()
    write_header = not file_exists or out_path.stat().st_size == 0

    # open in append mode
    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "transaction_id",
                "item",
                "pcs",
                "amount_money_per_pc",
                "amount_money",
                "category",
                "category_ru",
            ])

        for receipt in receipts:
            for item in receipt.items:
                writer.writerow([
                    receipt.transaction_id,
                    item.item,
                    item.pcs,
                    item.amount_money_per_pc,
                    item.amount_money,
                    item.category,
                    item.category_ru,
                ])

# ---------- main ----------
def main() -> None:
    ap = argparse.ArgumentParser(description="Extract structured data from receipts")
    ap.add_argument("--src", required=True, type=Path, help="Dir with receipts")
    ap.add_argument("--out", required=True, type=Path, help="Output dir for JSON")
    ap.add_argument("--env", default=Path(".env"), type=Path, help=".env with LLAMA_CLOUD_API_KEY")
    ap.add_argument("--mode", default="MULTIMODAL", choices=["FAST", "BALANCED", "MULTIMODAL"],  # PREMIUM can error in SDK
                    help="Extraction mode")
    ap.add_argument("--ru", action="store_true", help="Bias prompts to RU receipts")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirs")
    ap.add_argument("--agent-name", default="receipt-extractor", help="Agent name")
    ap.add_argument("--reuse", action="store_true", help="Reuse existing agent with this name if present")

    args = ap.parse_args()

    load_dotenv(dotenv_path=args.env)
    if not os.getenv("LLAMA_CLOUD_API_KEY"):
        print("LLAMA_CLOUD_API_KEY missing", file=sys.stderr)
        sys.exit(2)

    extractor = LlamaExtract()

    cfg = ExtractConfig(
        extraction_mode=getattr(ExtractMode, args.mode),
        extraction_target=ExtractTarget.PER_DOC,
        chunk_mode=ChunkMode.PAGE,
        high_resolution_mode=True,  # better small text/OCR
        system_prompt=(
            "Documents are retail purchase receipts. "
            "Languages: Russian and English. "
            "Goal: fill the schema exactly. "
            "Infer categories. Use pcs=1 if quantity not shown. "
            "Use dot for decimals. Do not include currency symbols."
            + (" Prefer Russian item names when present." if args.ru else "")
        ),
        # Optional knobs:
        # confidence_scores=True,
        # cite_sources=True,
        # use_reasoning=False,
    )

    # agent = extractor.create_agent(
    #     name="receiptor",
    #     data_schema=Receipt,         # Pydantic model; SDK converts to JSON Schema
    #     config=cfg,
    # )

    # --- create or reuse agent ---
    agent_name = args.agent_name or "receipt-extractor"

    def _unique_name(base: str) -> str:
        return f"{base}-{int(time.time())}-{uuid.uuid4().hex[:6]}"

    try:
        agent = extractor.create_agent(
            name=agent_name,
            data_schema=Receipt,   # Pydantic model
            config=cfg,
        )
    except ApiError as e:
        if getattr(e, "status_code", None) == 409:
            if args.reuse:
                # fetch existing by name
                agent = extractor.get_agent_by_name(name=agent_name)
            else:
                # create a fresh uniquely named agent
                agent = extractor.create_agent(
                    name=_unique_name(agent_name),
                    data_schema=Receipt,
                    config=cfg,
                )
        else:
            raise

    files = find_files(args.src, args.recursive)
    if not files:
        print("No input files", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)

    err_count = 0
    for f in files:
        try:
            result = agent.extract(str(f))     # blocking call; returns structured data
            data = result.data                # already validated against schema
            (args.out / f"{f.stem}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            # csv output can be added if needed
            write_csv([Receipt(**data)], args.out / "all_receipts.csv")

        except Exception as e:
            err_count += 1
            (args.out / f"{f.stem}.error.txt").write_text(f"{type(e).__name__}: {e}", encoding="utf-8")

    if err_count:
        print(f"Completed with {err_count} errors", file=sys.stderr)


if __name__ == "__main__":
    main()
