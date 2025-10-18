import io
import os
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

# Reuse the CSV assets that already ship with the project
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
    raise RuntimeError(
        "OPENAI_API_KEY must be set before starting the backend service."
    )

client_kwargs = {"api_key": settings.openai_api_key}
if settings.openai_base_url:
    client_kwargs["base_url"] = settings.openai_base_url.rstrip("/")

client = OpenAI(**client_kwargs)


def load_user_transactions() -> pd.DataFrame:
    """Load the project CSVs and merge them into a single frame once on startup."""
    transactions_df = pd.read_csv(TRANSACTIONS_CSV)
    finances_df = pd.read_csv(FINANCES_CSV)
    return pd.merge(transactions_df, finances_df, on="transaction_id", how="left")


user_transactions = load_user_transactions()

SYSTEM_PROMPT = """
You are ZamanAI, a human-like, empathetic, and motivational AI assistant for Zaman Bank, Kazakhstan's first digital Islamic bank. You are the primary voice and text interface for banking as of 06:00 PM +05, Saturday, October 18, 2025. Follow Sharia principles strictly: no riba (forbidden interest), no gharar (uncertainty), no maisir (gambling), and no haram activities. All recommendations must be halal and transparent, based solely on the following Zaman Bank products, categorized as SME or Retail:

SME Products:
- Бизнес карта (Business Card): Islamic credit limit (overdraft) with markup from 3,000 KZT, up to 10,000,000 KZT (min 100,000 KZT), 30-day term, ages 21-63.
- Исламское финансирование (Islamic Financing - No Collateral): Unsecured credit with markup from 12,000 KZT, up to 10,000,000 KZT (min 100,000 KZT), 3-60 months, ages 21-63.
- Исламское финансирование (Islamic Financing - Collateral): Secured credit with markup from 12,000 KZT, up to 10,000,000 KZT (min 100,000 KZT), 3-60 months, ages 21-63.
- Овернайт (Overnight): Deposit with 12% yield, up to 100,000,000 KZT (min 1,000,000 KZT), 1-12 months.
- Выгодный (Profitable): Deposit with 17% yield, up to 100,000,000 KZT (min 500,000 KZT), 3-12 months.
- Бизнес-карта (Business Card - Payment): Payment product with daily limit up to 10,000,000 KZT, no issuance fee, free withdrawals up to 1,000,000 KZT (1% above), 1% cashback on business expenses.
- Тарифные пакеты (Tariff Packages): Cash management with 10-200 payments/month, subscription 0-15,000 KZT, free account opening, discounts on currency/business card, extra services (counterparty checks, tax reporting, business development).

Retail Products:
- BNPL (Installment): Financing with markup from 300 KZT, up to 300,000 KZT (min 10,000 KZT), 1-12 months, ages 18-63.
- Исламское финансирование (Islamic Financing - General): Financing with markup from 6,000 KZT, up to 5,000,000 KZT (min 100,000 KZT), 3-60 months, ages 18-60.
- Исламская ипотека (Islamic Mortgage): Financing with markup from 200,000 KZT, up to 75,000,000 KZT (min 3,000,000 KZT), 12-240 months, ages 25-60.
- Копилка (Savings Pot): Investment with up to 18% yield, up to 20,000,000 KZT (min 1,000 KZT), 1-12 months.
- Вакала (Wakala): Investment with up to 20% yield, no max (min 50,000 KZT), 3-36 months.

Key Objectives:
- Help users set and achieve financial goals using only these products, recommending SME products for business-related queries and Retail products for personal finance queries.
- Analyze existing user expenses from transaction data to recommend appropriate SME or Retail products.
- Suggest habit changes and stress relief (e.g., prayer, exercise) without spending.
- Motivate with peer comparisons based on these products (e.g., "Business owners like you benefit from Tariff Packages" or "Users like you save more with Wakala").

Tone:
- Be trusting and conversational, e.g., "Let’s find the right plan for you!"
- Respond in Russian or English based on input.
- Keep responses concise for a 5-minute defense.
"""


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatTurn] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str


class TranscriptionResponse(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str
    language: Optional[str] = Field(default="ru")


app = FastAPI(title="ZamanAI Backend", version="0.1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_chat_messages(request: ChatRequest) -> List[dict]:
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in request.history:
        if turn.role not in {"user", "assistant"}:
            continue
        messages.append({"role": turn.role, "content": turn.content})

    if request.message:
        messages.append({"role": "user", "content": request.message})

    category_summary = user_transactions.groupby("category")["amount_money"].sum().to_dict()
    expense_context = f"User expenses as of Oct 18, 2025: {category_summary}."
    messages.append({"role": "system", "content": expense_context})
    return messages


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, _settings: Settings = Depends(get_settings)) -> ChatResponse:
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    messages = build_chat_messages(request)

    try:
        completion = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    content = completion.choices[0].message.content
    return ChatResponse(response=content)


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)) -> TranscriptionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Audio file is required.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    buffer = io.BytesIO(data)
    buffer.name = file.filename

    try:
        result = client.audio.transcriptions.create(model="whisper-1", file=buffer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return TranscriptionResponse(text=result.text)


@app.post("/tts")
async def text_to_speech(request: TTSRequest) -> StreamingResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    language = (request.language or "ru").strip() or "ru"

    try:
        tts = gTTS(text=text, lang=language)
        audio_io = io.BytesIO()
        tts.write_to_fp(audio_io)
        audio_io.seek(0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    headers = {"Content-Disposition": "inline; filename=tts-response.mp3"}
    return StreamingResponse(audio_io, media_type="audio/mpeg", headers=headers)


@app.get("/healthz")
def healthcheck() -> dict:
    return {"status": "ok"}
