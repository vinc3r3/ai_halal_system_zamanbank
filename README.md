# AI Halal System - Zamanbank

AI Halal System - Zamanbank is a prototype assistant that blends retrieval augmented chat, financial diary parsing, and lightweight voice features to help users answer halal-compliance questions and track spending. The project ships with a FastAPI backend, a Vite React client, CSV knowledge bases, and optional scripts for offline receipt OCR.

## Core Features

- Halal compliance chatbot backed by OpenAI and retrieval over curated CSV knowledge bases.
- Financial diary ingestion that turns free-form text or receipt uploads into structured transactions.
- Spending analytics dashboard with category breakdowns and item level drill downs.
- Voice support: audio transcription, text-to-speech playback, and microphone driven chat input.
- Profile tab with theme toggle and editable demographics for future personalization.

## Architecture at a Glance

- **Backend (`main_app/backend/main.py`)**  
  FastAPI application that preloads CSV data into in-memory vector stores, performs embedding powered retrieval, and exposes chat, transcription, TTS, diary parsing, and receipt extraction endpoints.
- **Frontend (`main_app/src`)**  
  Single-page app built with React 18, Vite, and shadcn/ui components. It offers three primary tabs (Chatbot, Finances, Profile) and calls the backend through `VITE_API_BASE_URL`.
- **Receipt processing (`main_app/receipt_processing`)**  
  Command line workflow that leverages LlamaExtract to turn batches of image/PDF receipts into normalized CSV rows ready for import.
- **Data assets (`main_app/*.csv`)**  
  Bundled CSV files with product catalogs, historical transactions, and halal rulings used by both the RAG pipeline and the analytics tab.
- **Utilities**  
  `update_backend_snippet.py` patches knowledge field mappings, while `tests.ipynb` contains exploratory notebooks.

## Repository Layout

```
ai_halal_system_zamanbank/
├── main_app/
│   ├── backend/                # FastAPI service (main.py) and temp artifacts
│   ├── src/                    # React + Vite client (components, contexts, styles)
│   ├── receipt_processing/     # LlamaExtract CLI for receipt OCR
│   ├── chroma_knowledge/       # Chroma vector store snapshot (optional)
│   ├── chroma_products/        # Chroma vector store snapshot (optional)
│   ├── build/                  # Frontend build artifacts (if generated)
│   ├── package.json            # Frontend dependencies and scripts
│   └── requirements.txt         # Backend dependency list
├── old_ui/                     # Archived web client prototype
├── .env                        # Environment variables (do not commit real keys)
├── tests.ipynb                 # Notebook with ad-hoc experiments
├── update_backend_snippet.py   # Helper script to rewrite knowledge mappings
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
```

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, OpenAI Python SDK, NumPy, Pandas, gTTS, python-dotenv
- **AI/RAG:** OpenAI `text-embedding-3-small`, configurable chat completions, optional Llama Cloud APIs for receipt extraction
- **Frontend:** React 18, Vite, TypeScript, Tailwind CSS utilities, shadcn/ui (Radix primitives), Recharts, lucide-react
- **Tooling:** npm or pnpm for the client, uvicorn for local API hosting, optional Streamlit prototypes

## Prerequisites

- Python 3.10 or newer
- Node.js 18+ (recommended) and npm (or pnpm/yarn if preferred)
- An OpenAI API key with access to GPT-4o-mini, GPT-4o-transcribe, and text-embedding-3-small
- Optional: Llama Cloud API key for production grade receipt OCR, FFmpeg for richer audio format support

## Environment Configuration

The backend reads environment variables via `dotenv` from the repository root. Replace secrets before running locally and avoid committing real keys.

Create a `.env` file in the project root:

```dotenv
# OpenAI settings
OPENAI_API_KEY=sk-your-key
# Optional overrides
# OPENAI_BASE_URL=https://api.openai.com/v1
# BACKEND_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# Receipt OCR (optional)
# LLAMA_CLOUD_API_KEY=llx-your-key
# AGENT_NAME=receipt-extractor
# MODE=MULTIMODAL
# PREFER_RU=false

# Frontend -> Backend bridge
VITE_API_BASE_URL=http://localhost:8000

# Optional Gemini or other provider keys can be added as needed
# GEMINI_API_KEY=your-key
```

> Rotate any example keys that shipped with the repo; they are placeholders and should not be used in production.

## Backend Setup

```powershell
cd main_app
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
uvicorn main_app.backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Key behaviors:

- On startup the service loads the CSV datasets, creates embeddings with OpenAI, and caches them in memory.
- `/chat` combines user prompts with retrieved rows to ground the answer and returns citation metadata.
- `/parse-text` and `/save-transaction` support the diary workflow, while `/receipt/extract` batches LlamaExtract responses into CSV rows.
- gTTS is used for `/tts` and `/speech`; ensure outbound network access is permitted.

## Frontend Setup

```powershell
cd main_app
npm install
npm run dev
```

The development server defaults to `http://localhost:5173` and proxies API traffic to `VITE_API_BASE_URL`.

## Running the Stack Locally

1. Start the FastAPI backend (`uvicorn main_app.backend.main:app --reload`).
2. Launch the Vite dev server (`npm run dev`).
3. Open the app in a browser (default `http://localhost:5173`) and interact with the Chatbot, Finances, and Profile tabs.

## API Surface

| Method | Path | Purpose | Notes |
| ------ | ---- | ------- | ----- |
| POST | `/chat` | Chat with retrieval augmented answers and halal rulings | Body: `{ "message": string, "history": MessageHistory[] }` |
| POST | `/transcribe` | Convert uploaded audio to text via OpenAI Whisper | Accepts `multipart/form-data` with `file` |
| POST | `/tts` | Generate speech audio from text using gTTS | Returns an `audio/mpeg` stream |
| POST | `/speech` | Same as `/tts` with language detection helper | |
| POST | `/parse-text` | Parse free-form financial diary entries | Returns normalized item, amount, category |
| POST | `/save-transaction` | Persist parsed transactions into CSV ledger | Appends to `zamanbank_transactions.csv` and finances CSV |
| POST | `/receipt/extract` | Batch receipt OCR via LlamaExtract | Requires `LLAMA_CLOUD_API_KEY` |
| GET | `/get-parsed-transactions` | Return merged transaction ledger for the dashboard | Used by the Finances tab |
| GET | `/healthz` | Lightweight readiness probe | Returns `{ "status": "ok" }` |
| GET | `/test-parse` | Diagnostics endpoint for the text parser | Helpful during development |

The backend returns structured Pydantic models (see `main_app/backend/main.py`) for all JSON endpoints, making it straightforward to integrate additional clients.

## Data Pipeline and Storage

- **Knowledge base:** `zamanbank_database.csv` contains halal rulings, fed into the knowledge vector store.
- **Products:** `zamanbank_products.csv` provides catalog metadata for matching and enrichment.
- **Transactions:** `zamanbank_transactions.csv` stores diary derived entries, merged with `zamanbank_finances.csv` for analytics.
- **Category colors:** Defined in `main_app/src/data/financialData.ts` and generated on the fly for charts.

All CSV writes happen within the workspace; consider migrating to a database for multi-user scenarios.

## Receipt Processing CLI

The directory `main_app/receipt_processing` hosts a standalone tool to OCR batches of receipts.

```powershell
cd main_app/receipt_processing
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python receipts_extract.py --src .\data\input --out .\output --recursive
```

Outputs include per-receipt JSON files and a consolidated `receipts.csv` that can be merged into the main ledger.

## Development Notes

- `update_backend_snippet.py` rewrites the knowledge field mappings if CSV headers change; run it after updating source data.
- The `old_ui` folder keeps an earlier frontend; it is not part of the current build but can serve as reference.
- `tests.ipynb` demonstrates ad-hoc experiments; convert critical checks into automated tests as the project matures.

## Troubleshooting

- **Embedding or chat failure:** Verify `OPENAI_API_KEY` and ensure the specified models are allowed for your account.
- **CORS issues:** Update `BACKEND_CORS_ORIGINS` to match your frontend origin.
- **gTTS errors:** The service must reach Google TTS endpoints; check network permissions or switch to an offline TTS engine.
- **Receipt extraction errors:** Confirm `LLAMA_CLOUD_API_KEY`, agent name, and extraction mode; retry without `--reuse` if the agent was deleted.
- **Encoding artifacts:** CSV files mix UTF-8 and Windows-1251; the backend already retries encodings, but ensure new data is saved with UTF-8 BOM.

## License

Released under the [MIT License](LICENSE).
