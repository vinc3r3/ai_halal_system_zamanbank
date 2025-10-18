import io
import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Dict
import math

import numpy as np
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
KNOWLEDGE_DB = BASE_DIR / "zamanbank_database.csv"
# INCOME_CSV = BASE_DIR / "zamanbank_income.csv"

# cust_id = "CUST-24962"

class Settings(BaseModel):
    openai_api_key: str = Field(default="")
    openai_base_url: Optional[str] = Field(default=None)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "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"),
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

# Embedding model to use
EMBEDDING_MODEL = "text-embedding-3-small"  # change if desired
TOP_K = 5  # number of rows to retrieve per CSV

# Utility: cosine similarity
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Build a simple in-memory "vector store" for each CSV.
# Each store: list of dicts with keys: "id", "text", "metadata", "embedding"
class VectorStore:
    def __init__(self, name: str):
        self.name = name
        self.rows: List[Dict] = []

    def add_row(self, row_id: str, text: str, metadata: dict, embedding: np.ndarray):
        self.rows.append({"id": row_id, "text": text, "metadata": metadata, "embedding": embedding})

    def retrieve(self, query_embedding: np.ndarray, top_k: int = TOP_K) -> List[Dict]:
        scores = []
        for row in self.rows:
            sim = cosine_similarity(query_embedding, row["embedding"])
            scores.append((sim, row))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [row for score, row in scores[:top_k]]

# Create vector stores (populated at startup)
knowledge_store = VectorStore("knowledge")
products_store = VectorStore("products")
transactions_store = VectorStore("transactions")  # transaction rows
finances_store = VectorStore("finances")  # finances rows

# Helper to convert a DataFrame row into a short text chunk for embedding (one chunk per row)
def row_to_text_chunk(df_row: pd.Series, source_name: str) -> str:
    # Convert row to a compact string representation
    # Keep simple: include key fields for the CSV type
    if source_name == "knowledge":
        # assume knowledge CSV has e.g. "id", "title", "content" or similar
        parts = []
        for col in df_row.index:
            if pd.isna(df_row[col]):
                continue
            parts.append(f"{col}: {str(df_row[col])}")
        return " | ".join(parts)
    else:
        parts = []
        for col in df_row.index:
            if pd.isna(df_row[col]):
                continue
            parts.append(f"{col}: {str(df_row[col])}")
        return " | ".join(parts)

def compute_embeddings(texts: List[str]) -> List[np.ndarray]:
    # chunk embeddings call to OpenAI
    if not texts:
        return []
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    embeddings = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
    return embeddings

def index_csv_to_store(csv_path: Path, store: VectorStore, source_name: str, id_column: Optional[str] = None):
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    # Build chunks: one per row
    chunks = []
    ids = []
    metas = []
    for idx, row in df.iterrows():
        text = row_to_text_chunk(row, source_name)
        # choose id
        if id_column and id_column in row.index:
            row_id = str(row[id_column])
        else:
            row_id = f"{source_name}_{idx}"
        chunks.append(text)
        ids.append(row_id)
        # store row as metadata (you may want to reduce size in production)
        metas.append({c: (None if pd.isna(row[c]) else row[c]) for c in row.index})
    # compute embeddings in batches
    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch_texts = chunks[i:i+BATCH]
        batch_embeddings = compute_embeddings(batch_texts)
        for j, emb in enumerate(batch_embeddings):
            store.add_row(ids[i+j], batch_texts[j], metas[i+j], emb)

# Index all CSVs at startup (each row → one chunk + embedding)
# NOTE: this will make embedding calls on startup. For large CSVs persist embeddings instead.
index_csv_to_store(KNOWLEDGE_DB, knowledge_store, "knowledge", id_column="id" if "id" in pd.read_csv(KNOWLEDGE_DB, nrows=0).columns else None)
index_csv_to_store(PRODUCTS_CSV, products_store, "products", id_column="product_id" if "product_id" in pd.read_csv(PRODUCTS_CSV, nrows=0).columns else None)
index_csv_to_store(TRANSACTIONS_CSV, transactions_store, "transactions", id_column="transaction_id" if "transaction_id" in pd.read_csv(TRANSACTIONS_CSV, nrows=0).columns else None)
index_csv_to_store(FINANCES_CSV, finances_store, "finances", id_column="transaction_id" if "transaction_id" in pd.read_csv(FINANCES_CSV, nrows=0).columns else None)


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


SYSTEM_PROMPT = """ You are ZamanAI, a human-like, empathetic, and motivational AI assistant for Zaman Bank, Kazakhstan's first digital Islamic bank. You are the primary voice and text interface for banking as of 06:00 PM +05, Saturday, October 18, 2025. Follow Sharia principles strictly: no riba (forbidden interest), no gharar (uncertainty), no maisir (gambling), and no haram activities. All recommendations must be halal and transparent, based solely on the following Zaman Bank products, categorized as SME or Retail: SME Products: - Бизнес карта (Business Card): Islamic credit limit (overdraft) with markup from 3,000 KZT, up to 10,000,000 KZT (min 100,000 KZT), 30-day term, ages 21-63. - Исламское финансирование (Islamic Financing - No Collateral): Unsecured credit with markup from 12,000 KZT, up to 10,000,000 KZT (min 100,000 KZT), 3-60 months, ages 21-63. - Исламское финансирование (Islamic Financing - Collateral): Secured credit with markup from 12,000 KZT, up to 10,000,000 KZT (min 100,000 KZT), 3-60 months, ages 21-63. - Овернайт (Overnight): Deposit with 12% yield, up to 100,000,000 KZT (min 1,000,000 KZT), 1-12 months. - Выгодный (Profitable): Deposit with 17% yield, up to 100,000,000 KZT (min 500,000 KZT), 3-12 months. - Бизнес-карта (Business Card - Payment): Payment product with daily limit up to 10,000,000 KZT, no issuance fee, free withdrawals up to 1,000,000 KZT (1% above), 1% cashback on business expenses. - Тарифные пакеты (Tariff Packages): Cash management with 10-200 payments/month, subscription 0-15,000 KZT, free account opening, discounts on currency/business card, extra services (counterparty checks, tax reporting, business development). Retail Products: - BNPL (Installment): Financing with markup from 300 KZT, up to 300,000 KZT (min 10,000 KZT), 1-12 months, ages 18-63. - Исламское финансирование (Islamic Financing - General): Financing with markup from 6,000 KZT, up to 5,000,000 KZT (min 100,000 KZT), 3-60 months, ages 18-60. - Исламская ипотека (Islamic Mortgage): Financing with markup from 200,000 KZT, up to 75,000,000 KZT (min 3,000,000 KZT), 12-240 months, ages 25-60. - Копилка (Savings Pot): Investment with up to 18% yield, up to 20,000,000 KZT (min 1,000 KZT), 1-12 months. - Вакала (Wakala): Investment with up to 20% yield, no max (min 50,000 KZT), 3-36 months. Key Objectives: - Help users set and achieve financial goals using only these products, recommending SME products for business-related queries and Retail products for personal finance queries. - Analyze existing user expenses from transaction data to recommend appropriate SME or Retail products. - Suggest habit changes and stress relief (e.g., prayer, exercise) without spending. - Motivate with peer comparisons based on these products (e.g., "Business owners like you benefit from Tariff Packages" or "Users like you save more with Wakala"). Tone: - Be trusting and conversational, e.g., "Let’s find the right plan for you!" - Respond in Russian or English based on input. - Keep responses concise for a 5-minute defense. """


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


def select_data_sources_for_query(message: str) -> Dict[str, bool]:
    """
    Always use all sources.
    """
    return {
        "knowledge": True,
        "products": True,
        "transactions": True,
        "finances": True,
    }

def retrieve_relevant_rows(message: str, sources: Dict[str, bool], top_k: int = TOP_K) -> Dict[str, List[Dict]]:
    # compute embedding for the message
    emb_resp = client.embeddings.create(model=EMBEDDING_MODEL, input=message)
    query_emb = np.array(emb_resp.data[0].embedding, dtype=np.float32)
    results = {}
    if sources.get("knowledge"):
        results["knowledge"] = knowledge_store.retrieve(query_emb, top_k=top_k)
    if sources.get("products"):
        results["products"] = products_store.retrieve(query_emb, top_k=top_k)
    if sources.get("transactions"):
        results["transactions"] = transactions_store.retrieve(query_emb, top_k=top_k)
    if sources.get("finances"):
        results["finances"] = finances_store.retrieve(query_emb, top_k=top_k)
    return results

def format_retrieved_for_prompt(retrieved: Dict[str, List[Dict]]) -> str:
    parts = []
    for source_name, rows in retrieved.items():
        if not rows:
            continue
        parts.append(f"--- Retrieved from {source_name.upper()} (most relevant rows) ---")
        for r in rows:
            # include metadata and text: concise
            meta_summary = ", ".join(f"{k}={v}" for k, v in (r["metadata"] or {}).items() if v is not None)
            parts.append(f"{r['text']}\n({meta_summary})")
    return "\n\n".join(parts)

def build_chat_messages(request: ChatRequest) -> List[dict]:
    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # add history
    for turn in request.history:
        if turn.role not in {"user", "assistant"}:
            continue
        messages.append({"role": turn.role, "content": turn.content})

    if not request.message.strip():
        return messages

    # Decide which CSVs to consult for retrieval
    sources = select_data_sources_for_query(request.message)
    # Retrieve only from relevant CSVs
    retrieved = retrieve_relevant_rows(request.message, sources, top_k=TOP_K)

    # Attach retrieved rows as system content (so model sees them as facts)
    retrieved_text = format_retrieved_for_prompt(retrieved)
    if retrieved_text:

        
        retrieval_wrapper = (
    "You are given the following relevant rows from the database:\n"
    "1) You must explicitly check all retrieved rows for rulings about whether the product or action is halal or haram. "
    "2) When quoting the 'ответ/объяснение', put it in quotation marks and include the source chapter (column глава) in parentheses as in ({cite: глава}).\n"
    "3) Combine information from multiple rows to give a complete, well-rounded answer.\n"
    "4) Explain all key points clearly so a non-expert can understand, elaborating when necessary.\n"
    "5) Avoid omitting relevant details; do not focus on only one row.\n"
    "6) Do not invent any product parameters or facts not present in these rows.\n\n"
    "If no retrieved row explicitly mentions halal/haram, you may explain based on general knowledge, but always indicate that no direct citation was found.\n"
    + retrieved_text
)


        messages.append({"role": "system", "content": retrieval_wrapper})

    # Finally append user's message
    messages.append({"role": "user", "content": request.message})
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



if __name__ == "__main__":
    print("Testing RAG + LLM pipeline...\n")

    user_query = "можно ли торговать свининой?"

    test_request = ChatRequest(message=user_query, history=[])

    # Build messages including retrieval
    messages = build_chat_messages(test_request)

    # Print system messages (context + retrieved rows)
    print("=== SYSTEM CONTEXT + RETRIEVED CSV ROWS ===\n")
    for m in messages:
        if m["role"] == "system":
            # Print first 1500 characters per system message for readability
            print(m["content"][:1500])
            print("\n---\n")

    print("\nGenerating response with LLM...\n")

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        response = completion.choices[0].message.content
        print("💬 ZamanAI:", response)
    except Exception as e:
        print("Error while querying the model:", e)
