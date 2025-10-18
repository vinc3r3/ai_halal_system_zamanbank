import csv
import io
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from gtts import gTTS
from openai import OpenAI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent

PRODUCTS_CSV = BASE_DIR / "zamanbank_products.csv"
TRANSACTIONS_CSV = BASE_DIR / "zamanbank_transactions.csv"
FINANCES_CSV = BASE_DIR / "zamanbank_finances.csv"


class Settings(BaseModel):
    openai_api_key: str = Field(default="")
    openai_base_url: Optional[str] = Field(default=None)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        cors_origins=[
            origin.strip()
            for origin in os.getenv("BACKEND_CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ],
    )


settings = get_settings()

if not settings.openai_api_key:
    raise RuntimeError("OPENAI_API_KEY must be set before starting the backend service.")

client_kwargs = {"api_key": settings.openai_api_key}
if settings.openai_base_url:
    client_kwargs["base_url"] = settings.openai_base_url.rstrip("/")

client = OpenAI(**client_kwargs)


def load_user_transactions() -> pd.DataFrame:
    if not TRANSACTIONS_CSV.exists() or not FINANCES_CSV.exists():
        return pd.DataFrame()
    transactions_df = pd.read_csv(TRANSACTIONS_CSV)
    finances_df = pd.read_csv(FINANCES_CSV)
    return pd.merge(transactions_df, finances_df, on="transaction_id", how="left")


user_transactions = load_user_transactions()


def ensure_csv_headers(path: Path, headers: List[str]) -> None:
    if not path.exists():
        path.write_text(",".join(headers) + "\n", encoding="utf-8")
        return
    if path.stat().st_size == 0:
        path.write_text(",".join(headers) + "\n", encoding="utf-8")


CHAT_SYSTEM_PROMPT = """
You are ZamanAI, a caring and practical conversational assistant for Zaman Bank, Kazakhstan's first digital Islamic bank.
You help users manage their finances in a halal way, recommend appropriate bank products, and keep the tone warm and concise.
When you reference bank products, use information from the provided CSV files only.
"""


class ChatHistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatHistoryTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str


class TranscriptionResponse(BaseModel):
    text: str


class SpeechRequest(BaseModel):
    text: str
    language: Literal["ru", "en"] = "ru"


class ParsedTransaction(BaseModel):
    item: str
    amount: float
    category: str
    category_ru: str
    success: bool
    error_message: Optional[str] = None


class TextParseRequest(BaseModel):
    text: str


app = FastAPI(title="ZamanAI Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_chat_messages(request: ChatRequest) -> List[dict]:
    messages: List[dict] = [{"role": "system", "content": CHAT_SYSTEM_PROMPT.strip()}]

    for turn in request.history:
        if not turn.content.strip():
            continue
        messages.append({"role": turn.role, "content": turn.content})

    if request.message.strip():
        messages.append({"role": "user", "content": request.message})

    if not user_transactions.empty:
        category_summary = (
            user_transactions.groupby("category")["amount_money"].sum().to_dict()
        )
        context = f"User expenses summary (amount_money by category): {category_summary}. Use this for personalised insights."
        messages.append({"role": "system", "content": context})

    return messages


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, _settings: Settings = Depends(get_settings)) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=build_chat_messages(request),
        temperature=0.3,
    )
    reply = completion.choices[0].message.content.strip()
    return ChatResponse(response=reply)


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio file.")

    audio_file = io.BytesIO(data)
    audio_file.name = file.filename or "audio.webm"

    transcription = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file,
        response_format="json",
    )
    text = transcription.text.strip() if transcription.text else ""
    return TranscriptionResponse(text=text)


def synthesize_speech(text: str, language: str) -> bytes:
    try:
        tts = gTTS(text=text, lang="ru" if language == "ru" else "en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"TTS failed: {exc}") from exc


@app.post("/tts")
def tts(request: SpeechRequest) -> StreamingResponse:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    audio_bytes = synthesize_speech(request.text, request.language)
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")


@app.post("/speech")
def speech(request: SpeechRequest) -> StreamingResponse:
    return tts(request)


def parse_financial_text(text: str) -> ParsedTransaction:
    prompt = f"""
You are a structured data parser for financial diary entries written in Russian or Kazakh.
Extract a single purchase from the following message:
\"\"\"{text}\"\"\"
Return JSON with exactly these keys:
- item: product or service name in Russian (or Kazakh). If unknown, put "Unknown item".
- amount: numeric value with dot decimal separator.
- category: English category chosen from this list: Utilities & Housing, Home & Furniture, Pocket Money, Groceries, Transport, Dining & Cafes, Entertainment, Communication, Health, Car Expenses, Sports, Children, Travel, Clothing, Beauty, Gifts, Other.
- category_ru: Russian category paired with the English list above, mapping as:
  Utilities & Housing -> ЖКХ
  Home & Furniture -> Все для дома
  Pocket Money -> Карманные
  Groceries -> Продукты
  Transport -> Транспорт
  Dining & Cafes -> Еда
  Entertainment -> Развлечения
  Communication -> Связь
  Health -> Здоровье
  Car Expenses -> Авто
  Sports -> Спорт
  Children -> Дети
  Travel -> Путешествия
  Clothing -> Одежда
  Beauty -> Красота
  Gifts -> Подарки
  Other -> Прочее
- success: true if amount and item were found, false otherwise.
- error_message: null when success is true, otherwise short reason in Russian.
Output only valid JSON without extra commentary.
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You convert diary messages into structured JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        content = completion.choices[0].message.content.strip()
        data = ParsedTransaction.model_validate_json(content)
        return data
    except Exception as exc:  # pragma: no cover - fallback
        return ParsedTransaction(
            item="Unknown item",
            amount=0.0,
            category="Other",
            category_ru="Прочее",
            success=False,
            error_message=f"Parsing failed: {exc}",
        )


@app.post("/parse-text", response_model=ParsedTransaction)
def parse_text(request: TextParseRequest) -> ParsedTransaction:
    if not request.text.strip():
        return ParsedTransaction(
            item="Unknown item",
            amount=0.0,
            category="Other",
            category_ru="Прочее",
            success=False,
            error_message="Text cannot be empty",
        )
    return parse_financial_text(request.text)


def append_transaction_to_csv(amount: float, item: str, category: str, category_ru: str, pcs: int) -> str:
    transaction_id = str(uuid.uuid4())
    now = datetime.now()
    date_str = now.strftime("%m/%d/%Y")
    time_str = now.strftime("%H:%M:%S")

    ensure_csv_headers(
        TRANSACTIONS_CSV,
        ["date", "time", "amount", "transaction_id", "transactioner_id"],
    )
    ensure_csv_headers(
        FINANCES_CSV,
        ["transaction_id", "category", "amount_money", "item", "pcs", "category_ru"],
    )

    with TRANSACTIONS_CSV.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["date", "time", "amount", "transaction_id", "transactioner_id"],
        )
        writer.writerow(
            {
                "date": date_str,
                "time": time_str,
                "amount": amount,
                "transaction_id": transaction_id,
                "transactioner_id": f"CUST-{now.strftime('%Y%m%d%H%M%S')}",
            }
        )

    with FINANCES_CSV.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "transaction_id",
                "category",
                "amount_money",
                "item",
                "pcs",
                "category_ru",
            ],
        )
        writer.writerow(
            {
                "transaction_id": transaction_id,
                "category": category,
                "amount_money": amount,
                "item": item or "Unknown item",
                "pcs": pcs,
                "category_ru": category_ru,
            }
        )

    return transaction_id


@app.post("/save-transaction")
def save_transaction(transaction_data: dict) -> dict:
    try:
        amount = float(transaction_data.get("amount", 0.0))
        item = str(transaction_data.get("item") or "Unknown item")
        category = str(transaction_data.get("category") or "Other")
        category_ru = str(transaction_data.get("category_ru") or "Прочее")
        pcs = int(transaction_data.get("quantity") or transaction_data.get("pcs") or 1)

        if amount <= 0:
            raise ValueError("Amount must be greater than zero.")

        transaction_id = append_transaction_to_csv(amount, item, category, category_ru, pcs)
        return {
            "success": True,
            "message": "Transaction saved successfully",
            "transaction_id": transaction_id,
        }
    except Exception as exc:
        return {"success": False, "message": f"Error saving transaction: {exc}"}


def merge_transactions() -> List[dict]:
    if not TRANSACTIONS_CSV.exists():
        return []

    with TRANSACTIONS_CSV.open("r", encoding="utf-8") as csvfile:
        transaction_rows = list(csv.DictReader(csvfile))

    finance_rows = []
    if FINANCES_CSV.exists():
        with FINANCES_CSV.open("r", encoding="utf-8") as csvfile:
            finance_rows = list(csv.DictReader(csvfile))

    finance_lookup = {row["transaction_id"]: row for row in finance_rows}

    merged: List[dict] = []
    for row in transaction_rows:
        finance = finance_lookup.get(row["transaction_id"], {})
        merged.append(
            {
                "transaction_id": row["transaction_id"],
                "date": row["date"],
                "time": row["time"],
                "amount": float(row.get("amount", 0) or 0),
                "transactioner_id": row.get("transactioner_id", ""),
                "category": finance.get("category", "Other"),
                "category_ru": finance.get("category_ru", "Прочее"),
                "amount_money": float(finance.get("amount_money") or row.get("amount") or 0),
                "item": finance.get("item", "Unknown item"),
                "pcs": int(finance.get("pcs") or 1),
            }
        )
    return merged


@app.get("/get-parsed-transactions")
def get_parsed_transactions() -> dict:
    try:
        return {"transactions": merge_transactions()}
    except Exception as exc:
        return {"transactions": [], "error": str(exc)}


@app.get("/healthz")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.get("/test-parse")
def test_parse() -> dict:
    sample_text = "Я потратил 4500 тенге на продукты в магазине Magnum."
    parsed = parse_financial_text(sample_text)
    return {
        "input": sample_text,
        "parsed_result": parsed.model_dump(),
    }
