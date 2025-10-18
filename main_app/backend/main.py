import io
import os
import re
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

CURRENCY_WORDS = (
    "С‚РµРЅРіРµ",
    "С‚РµТЈРіРµ",
    "С‚Рі",
    "tg",
    "kzt",
    "в‚ё",
)

AMOUNT_REGEX = re.compile(
    r"(?P<value>\d{1,3}(?:[ \u00A0]\d{3})*(?:[.,]\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
    r"(?:\s*(?P<currency>С‚РµРЅРіРµ|С‚РµТЈРіРµ|С‚Рі|tg|kzt|в‚ё))?",
    re.IGNORECASE,
)

STOP_WORDS = {
    "СЏ",
    "РјС‹",
    "РїРѕС‚СЂР°С‚РёР»",
    "РїРѕС‚СЂР°С‚РёР»Р°",
    "РїРѕС‚СЂР°С‚РёР»Рё",
    "РєСѓРїРёР»",
    "РєСѓРїРёР»Р°",
    "РєСѓРїРёР»Рё",
    "РѕРїР»Р°С‚РёР»",
    "РѕРїР»Р°С‚РёР»Р°",
    "РѕРїР»Р°С‚РёР»Рё",
    "Р·Р°РїР»Р°С‚РёР»",
    "Р·Р°РїР»Р°С‚РёР»Р°",
    "Р·Р°РїР»Р°С‚РёР»Рё",
    "СЃРґРµР»Р°Р»",
    "СЃРґРµР»Р°Р»Р°",
    "СЃРґРµР»Р°Р»Рё",
    "РІР·СЏР»",
    "РІР·СЏР»Р°",
    "РІР·СЏР»Рё",
    "РѕС„РѕСЂРјРёР»",
    "РѕС„РѕСЂРјРёР»Р°",
    "РѕС„РѕСЂРјРёР»Рё",
    "РїРѕРїРѕР»РЅРёР»",
    "РїРѕРїРѕР»РЅРёР»Р°",
    "РїРѕРїРѕР»РЅРёР»Рё",
    "РїРµСЂРµРІРµР»",
    "РїРµСЂРµРІРµР»Р°",
    "РїРµСЂРµРІРµР»Рё",
    "РЅР°",
    "Р·Р°",
    "РґР»СЏ",
    "РїРѕ",
    "РІ",
    "РІРѕ",
    "Сѓ",
    "СЃ",
    "Рё",
    "РёР»Рё",
    "СЃРµРіРѕРґРЅСЏ",
    "РІС‡РµСЂР°",
    "СЃРµР№С‡Р°СЃ",
    "СѓС‚СЂРѕРј",
    "РІРµС‡РµСЂРѕРј",
    "РґРЅРµРј",
    "РЅРѕС‡СЊСЋ",
    "С‚РѕР»СЊРєРѕ",
    "С‡С‚Рѕ",
    "РїСЂРёРјРµСЂРЅРѕ",
    "РїСЂРёРјРµСЂ",
    "РѕРєРѕР»Рѕ",
    "РїРѕС‡С‚Рё",
    "РіРґРµ-С‚Рѕ",
    "РїСЂРёРјРµСЂРЅРѕ",
    "РїСЂРёРјРµСЂРЅРѕ",
    "РїРѕРєСѓРїРєР°",
    "РїРѕРєСѓРїРєСѓ",
    "СЂР°СЃС…РѕРґ",
    "СЃСѓРјРјР°",
    "С‚СЂР°РЅР·Р°РєС†РёСЏ",
    "РїР»Р°С‚РµР¶",
    "РѕРїР»Р°С‚Р°",
    "СЃС‡РµС‚",
    "СЃС‡С‘С‚",
    "С‚РѕРІР°СЂ",
    "СѓСЃР»СѓРіР°",
    "СѓСЃР»СѓРіРё",
    "СЂР°Р·РЅС‹С…",
    "РЅРµРјРЅРѕРіРѕ",
    "РїСЂРёРјРµСЂРЅРѕ",
    "РїСЂРёРјРµСЂРЅРѕ",
    "С‚РµРЅРіРµ",
    "С‚РµТЈРіРµ",
    "С‚Рі",
    "tg",
    "kzt",
    "в‚ё",
}

CATEGORY_RULES = [
    {
        "category": "Electronics",
        "category_ru": "Р­Р»РµРєС‚СЂРѕРЅРёРєР°",
        "keywords": [
            "СЌР»РµРєС‚СЂРѕРЅ",
            "С‚РµР»РµС„РѕРЅ",
            "СЃРјР°СЂС‚С„РѕРЅ",
            "Р°Р№С„РѕРЅ",
            "iphone",
            "airpods",
            "РЅР°СѓС€РЅ",
            "headphon",
            "РјРѕРЅРёС‚РѕСЂ",
            "РєР»Р°РІРёР°С‚",
            "keyboard",
            "РјС‹С€",
            "mouse",
            "РїР»Р°РЅС€РµС‚",
            "tablet",
            "РЅРѕСѓС‚Р±СѓРє",
            "laptop",
            "РєРѕРјРїСЊСЋС‚",
            "pc",
            "С‚РµР»РµРІРёР·",
            "tv",
            "РєР°РјРµСЂР°",
            "РјРёРєСЂРѕС„РѕРЅ",
            "РіР°РґР¶РµС‚",
            "СЂРѕСѓС‚РµСЂ",
            "router",
        ],
    },
    {
        "category": "Clothing",
        "category_ru": "РћРґРµР¶РґР°",
        "keywords": [
            "РѕРґРµР¶Рґ",
            "РїР»Р°С‚СЊРµ",
            "РґР¶РёРЅСЃ",
            "РєСЂРѕСЃСЃРѕРІРє",
            "РѕР±СѓРІ",
            "РєСѓСЂС‚Рє",
            "С„СѓС‚Р±РѕР»Рє",
            "СЂСѓР±Р°С€Рє",
            "hoodie",
            "С‚РѕР»СЃС‚РѕРІРє",
            "С€Р°РїРє",
            "С€С‚Р°РЅС‹",
            "Р±СЂСЋРє",
            "РїР°Р»СЊС‚",
            "СЃРІРёС‚РµСЂ",
            "СЋР±Рє",
            "Р°РєСЃРµСЃСЃСѓР°СЂ",
            "СЂРµРјРµРЅСЊ",
            "РЅРѕСЃРє",
        ],
    },
    {
        "category": "Groceries",
        "category_ru": "РџСЂРѕРґСѓРєС‚С‹",
        "keywords": [
            "РїСЂРѕРґСѓРєС‚",
            "РµРґР°",
            "РјР°РіР°Р·РёРЅ",
            "РјР°СЂРєРµС‚",
            "СЃСѓРїРµСЂРјР°СЂРєРµС‚",
            "РЅР°РїРёС‚",
            "РєРѕС„Рµ",
            "С‡Р°Р№",
            "РєРѕР»Р°",
            "cola",
            "Р±СѓР»Рє",
            "РјРѕР»РѕРє",
            "СЃС‹СЂ",
            "Р№РѕРіСѓСЂ",
            "С…Р»РµР±",
            "С„СЂСѓРєС‚",
            "РѕРІРѕС‰",
            "РїРµСЂРµРєСѓСЃ",
            "Р±Р°РєР°Р»Рµ",
            "РіРёРїРµСЂРјР°СЂРєРµС‚",
            "snack",
            "РїСЂРѕРґСѓРєС‚РѕРІ",
        ],
    },
    {
        "category": "Restaurant",
        "category_ru": "Р•РґР°",
        "keywords": [
            "СЂРµСЃС‚РѕСЂР°РЅ",
            "РєР°С„Рµ",
            "Р±Р°СЂ",
            "Р±СѓСЂРіРµСЂ",
            "С„Р°СЃС‚С„СѓРґ",
            "С€Р°СѓСЂРј",
            "РїРёС†С†",
            "СЃСѓС€Рё",
            "РґРѕРЅРµСЂ",
            "РґРѕРЅР°СЂ",
            "РѕР±РµРґ",
            "СѓР¶РёРЅ",
            "Р»Р°РЅС‡",
            "delivery",
            "РґРѕСЃС‚Р°РІРє",
            "РєРѕС„РµР№РЅ",
            "coffee",
            "coffeeshop",
            "bistro",
            "РіСЂРёР»СЊ",
            "grill",
        ],
    },
    {
        "category": "Transport",
        "category_ru": "РўСЂР°РЅСЃРїРѕСЂС‚",
        "keywords": [
            "С‚Р°РєСЃРё",
            "uber",
            "bolt",
            "СЏРЅРґРµРєСЃ",
            "РјР°СЂС€СЂСѓС‚",
            "Р°РІС‚РѕР±СѓСЃ",
            "РјРµС‚СЂРѕ",
            "РїСЂРѕРµР·Рґ",
            "РїРѕРµР·Рґ",
            "Р¶Рґ",
            "С‚СЂРѕР»Р»РµР№Р±СѓСЃ",
            "СЃР°РјРѕР»РµС‚",
            "Р°РІРёР°Р±РёР»",
            "РїРµСЂРµР»РµС‚",
            "Р°РІС‚РѕСЃС‚РѕСЏРЅ",
            "РїР°СЂРєРѕРІ",
            "С‚РѕРїР»РёРІ",
            "Р±РµРЅР·РёРЅ",
            "Р·Р°РїСЂР°РІ",
            "СЃР°РјРѕРєР°С‚",
            "РєР°СЂС€РµСЂ",
            "carshar",
            "ride",
        ],
    },
    {
        "category": "Car Expenses",
        "category_ru": "РђРІС‚Рѕ",
        "keywords": [
            "Р°РІС‚Рѕ",
            "РјР°С€РёРЅ",
            "С€РёРЅРѕРјРѕРЅ",
            "РјР°СЃР»Рѕ",
            "РјРѕР№Рє",
            "carwash",
            "Р°РЅС‚РёС„СЂРёР·",
            "С€РёРЅР°",
            "РєРѕР»РµСЃ",
            "С‚РµС…РѕСЃРјРѕС‚СЂ",
            "РѕСЃР°РіР°",
            "РєР°СЃРєРѕ",
            "СЂРµРјРѕРЅС‚",
        ],
    },
    {
        "category": "Utilities",
        "category_ru": "Р–РљРҐ",
        "keywords": [
            "Р¶РєС…",
            "РєРѕРјРјСѓРЅ",
            "РєРІР°СЂС‚РїР»Р°С‚",
            "Р°СЂРµРЅРґ",
            "СЌР»РµРєС‚СЂ",
            "РіР°Р·",
            "РІРѕРґР°",
            "С‚РµРїР»",
            "РёРЅС‚РµСЂРЅРµС‚",
            "СЃРІСЏР·СЊ",
            "РѕРїР»Р°С‚Р° СѓСЃР»СѓРі",
            "СЃС‡РµС‚",
            "СЃС‡С‘С‚",
            "РїР»Р°С‚РµР¶",
            "РїР»Р°С‚С‘Р¶",
        ],
    },
    {
        "category": "Home & Furniture",
        "category_ru": "Р’СЃРµ РґР»СЏ РґРѕРјР°",
        "keywords": [
            "РјРµР±РµР»",
            "РґРёРІР°РЅ",
            "СЃС‚РѕР»",
            "СЃС‚СѓР»",
            "РїРѕР»Рє",
            "РїРѕРґСѓС€",
            "РѕРґРµСЏР»",
            "РїРѕСЃСѓРґ",
            "РєР°СЃС‚СЂСЋР»",
            "С‚Р°СЂРµР»",
            "СЂРµРјРѕРЅС‚",
            "РёРЅСЃС‚СЂСѓРјРµРЅС‚",
            "Р»Р°РјРї",
            "СЃРІРµС‚РёР»СЊ",
            "РєРѕРІРµСЂ",
            "СѓР±РѕСЂРє",
            "РёРЅС‚РµСЂСЊРµСЂ",
            "С€РєР°С„",
            "РїРѕР»РѕС‚РµРЅС†",
            "РґРѕРј",
            "С…РѕР·СЏР№СЃС‚РІ",
        ],
    },
    {
        "category": "Healthcare",
        "category_ru": "Р—РґРѕСЂРѕРІСЊРµ",
        "keywords": [
            "Р°РїС‚РµРє",
            "Р»РµРєР°СЂ",
            "С‚Р°Р±Р»РµС‚",
            "РІРёС‚Р°РјРёРЅ",
            "РєР»РёРЅРёРє",
            "СЃС‚РѕРјР°С‚",
            "РјРµРґРёС†",
            "РІСЂР°С‡",
            "Р°РЅР°Р»РёР·",
            "РїРѕР»РёРєР»РёРЅ",
            "РґРёР°РіРЅРѕСЃС‚",
        ],
    },
    {
        "category": "Beauty",
        "category_ru": "РљСЂР°СЃРѕС‚Р°",
        "keywords": [
            "СЃР°Р»РѕРЅ",
            "РјР°РЅРёРєСЋСЂ",
            "РїРµРґРёРєСЋСЂ",
            "РєРѕСЃРјРµС‚",
            "СЃРїР°",
            "РїР°СЂРёРєРјР°С…",
            "Р±Р°СЂР±РµСЂ",
            "beauty",
            "РјР°РєРёСЏР¶",
            "lashes",
            "СЂРµСЃРЅРёС†",
            "Р±СЂРѕРІ",
            "СЃРѕР»СЏСЂ",
        ],
    },
    {
        "category": "Entertainment",
        "category_ru": "Р Р°Р·РІР»РµС‡РµРЅРёСЏ",
        "keywords": [
            "РєРёРЅРѕ",
            "cinema",
            "С‚РµР°С‚СЂ",
            "РєРѕРЅС†РµСЂС‚",
            "РёРіСЂ",
            "game",
            "РїРѕРґРїРёСЃРє",
            "netflix",
            "spotify",
            "РјСѓР·С‹Рє",
            "СЃС‚СЂРёРј",
            "club",
            "РІРµС‡РµСЂРёРЅ",
            "Р°С‚С‚СЂР°РєС†РёРѕРЅ",
            "СЂР°Р·РІР»РµРє",
            "С„РµСЃС‚РёРІ",
        ],
    },
    {
        "category": "Gifts",
        "category_ru": "РџРѕРґР°СЂРєРё",
        "keywords": [
            "РїРѕРґР°СЂ",
            "Р±СѓРєРµС‚",
            "С†РІРµС‚",
            "flower",
            "СЃСѓРІРµРЅРёСЂ",
            "РїСЂР°Р·РґРЅ",
            "С‚РѕСЂС‚",
        ],
    },
    {
        "category": "Travel",
        "category_ru": "РџСѓС‚РµС€РµСЃС‚РІРёСЏ",
        "keywords": [
            "РѕС‚РµР»",
            "hotel",
            "РіРѕСЃС‚РёРЅРёС†",
            "Р±РёР»РµС‚",
            "Р°РІРёР°Р±РёР»",
            "РїРµСЂРµР»РµС‚",
            "tur",
            "trip",
            "РїРѕРµР·РґРє",
            "Р°СЌСЂРѕРїРѕСЂС‚",
            "booking",
            "travel",
            "СЃР°РјРѕР»РµС‚",
            "rail",
        ],
    },
    {
        "category": "Sports",
        "category_ru": "РЎРїРѕСЂС‚",
        "keywords": [
            "СЃРїРѕСЂС‚",
            "С„РёС‚РЅРµСЃ",
            "gym",
            "С‚СЂРµРЅР°Р¶",
            "Р·Р°Р»",
            "Р°Р±РѕРЅРµРјРµРЅС‚",
            "РїСЂРѕС‚РµРёРЅ",
            "С‚СЂРµРЅРёСЂ",
            "Р№РѕРіР°",
            "Р±Р°СЃСЃРµР№РЅ",
            "РІРµР»РѕСЃРёРїРµРґ",
            "РєСЂРѕСЃСЃС„РёС‚",
            "marathon",
            "С€С‚Р°РЅРі",
            "С‚РµРЅРЅРёСЃ",
        ],
    },
    {
        "category": "Children",
        "category_ru": "Р”РµС‚Рё",
        "keywords": [
            "СЂРµР±РµРЅ",
            "РґРµС‚СЃРє",
            "РёРіСЂСѓС€",
            "С€РєРѕР»",
            "СЃР°РґРёРє",
            "РїР°РјРїРµСЂСЃ",
            "РїРѕРґРіСѓР·",
            "РєСЂСѓР¶РѕРє",
            "baby",
            "child",
        ],
    },
]

DEFAULT_CATEGORY = {"category": "Other", "category_ru": "РџСЂРѕС‡РµРµ"}

SYSTEM_PROMPT = """
You are ZamanAI, a human-like, empathetic, and motivational AI assistant for Zaman Bank, Kazakhstan's first digital Islamic bank. You are the primary voice and text interface for banking as of 06:00 PM +05, Saturday, October 18, 2025. Follow Sharia principles strictly: no riba (forbidden interest), no gharar (uncertainty), no maisir (gambling), and no haram activities. All recommendations must be halal and transparent, based solely on the following Zaman Bank products, categorized as SME or Retail:

SME Products:
- Р‘РёР·РЅРµСЃ РєР°СЂС‚Р° (Business Card): Islamic credit limit (overdraft) with markup from 3,000 KZT, up to 10,000,000 KZT (min 100,000 KZT), 30-day term, ages 21-63.
- РСЃР»Р°РјСЃРєРѕРµ С„РёРЅР°РЅСЃРёСЂРѕРІР°РЅРёРµ (Islamic Financing - No Collateral): Unsecured credit with markup from 12,000 KZT, up to 10,000,000 KZT (min 100,000 KZT), 3-60 months, ages 21-63.
- РСЃР»Р°РјСЃРєРѕРµ С„РёРЅР°РЅСЃРёСЂРѕРІР°РЅРёРµ (Islamic Financing - Collateral): Secured credit with markup from 12,000 KZT, up to 10,000,000 KZT (min 100,000 KZT), 3-60 months, ages 21-63.
- РћРІРµСЂРЅР°Р№С‚ (Overnight): Deposit with 12% yield, up to 100,000,000 KZT (min 1,000,000 KZT), 1-12 months.
- Р’С‹РіРѕРґРЅС‹Р№ (Profitable): Deposit with 17% yield, up to 100,000,000 KZT (min 500,000 KZT), 3-12 months.
- Р‘РёР·РЅРµСЃ-РєР°СЂС‚Р° (Business Card - Payment): Payment product with daily limit up to 10,000,000 KZT, no issuance fee, free withdrawals up to 1,000,000 KZT (1% above), 1% cashback on business expenses.
- РўР°СЂРёС„РЅС‹Рµ РїР°РєРµС‚С‹ (Tariff Packages): Cash management with 10-200 payments/month, subscription 0-15,000 KZT, free account opening, discounts on currency/business card, extra services (counterparty checks, tax reporting, business development).

Retail Products:
- BNPL (Installment): Financing with markup from 300 KZT, up to 300,000 KZT (min 10,000 KZT), 1-12 months, ages 18-63.
- РСЃР»Р°РјСЃРєРѕРµ С„РёРЅР°РЅСЃРёСЂРѕРІР°РЅРёРµ (Islamic Financing - General): Financing with markup from 6,000 KZT, up to 5,000,000 KZT (min 100,000 KZT), 3-60 months, ages 18-60.
- РСЃР»Р°РјСЃРєР°СЏ РёРїРѕС‚РµРєР° (Islamic Mortgage): Financing with markup from 200,000 KZT, up to 75,000,000 KZT (min 3,000,000 KZT), 12-240 months, ages 25-60.
- РљРѕРїРёР»РєР° (Savings Pot): Investment with up to 18% yield, up to 20,000,000 KZT (min 1,000 KZT), 1-12 months.
- Р’Р°РєР°Р»Р° (Wakala): Investment with up to 20% yield, no max (min 50,000 KZT), 3-36 months.

Key Objectives:
- Help users set and achieve financial goals using only these products, recommending SME products for business-related queries and Retail products for personal finance queries.
- Analyze existing user expenses from transaction data to recommend appropriate SME or Retail products.
- Suggest habit changes and stress relief (e.g., prayer, exercise) without spending.
- Motivate with peer comparisons based on these products (e.g., "Business owners like you benefit from Tariff Packages" or "Users like you save more with Wakala").

Tone:
- Be trusting and conversational, e.g., "LetвЂ™s find the right plan for you!"
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


class TextParseRequest(BaseModel):
    text: str


class ParsedTransaction(BaseModel):
    item: str
    amount: float
    category: str
    category_ru: str
    success: bool
    error_message: Optional[str] = None


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


def parse_financial_text(text: str) -> ParsedTransaction:
    """Parse Russian/Kazakh expense text into structured data (amount, item, categories)."""
    message = text.strip()
    if not message:
        return ParsedTransaction(
            item="Unknown item",
            amount=0.0,
            category=DEFAULT_CATEGORY["category"],
            category_ru=DEFAULT_CATEGORY["category_ru"],
            success=False,
            error_message="Text cannot be empty",
        )

    amount_match: Optional[re.Match[str]] = None
    observed_matches: List[re.Match[str]] = []

    for match in AMOUNT_REGEX.finditer(message):
        observed_matches.append(match)
        currency_group = match.group("currency")
        trailing = message[match.end(): match.end() + 6].lower()
        if currency_group or any(trailing.startswith(word) for word in CURRENCY_WORDS):
            amount_match = match
            break

    if not amount_match and observed_matches:
        amount_match = max(
            observed_matches,
            key=lambda candidate: normalize_amount(candidate.group("value")),
        )

    amount_value = normalize_amount(
        amount_match.group("value") if amount_match else None
    )

    if amount_value <= 0:
        return ParsedTransaction(
            item="Unknown item",
            amount=0.0,
            category=DEFAULT_CATEGORY["category"],
            category_ru=DEFAULT_CATEGORY["category_ru"],
            success=False,
            error_message="Не удалось определить сумму в сообщении",
        )

    item_value = extract_item(message, amount_match)
    category_context = classify_item(item_value)

    return ParsedTransaction(
        item=item_value,
        amount=amount_value,
        category=category_context["category"],
        category_ru=category_context["category_ru"],
        success=True,
        error_message=None,
    )


@app.post("/parse-text", response_model=ParsedTransaction)
def parse_text(request: TextParseRequest) -> ParsedTransaction:
    """Parse Russian financial text and extract transaction details."""
    return parse_financial_text(request.text)
def parse_text(request: TextParseRequest) -> ParsedTransaction:
    """Parse Russian financial text and extract transaction details."""
    if not request.text.strip():
        return ParsedTransaction(
            item="РџРѕРєСѓРїРєР°",
            amount=0.0,
            category="Other",
            category_ru="РџСЂРѕС‡РµРµ",
            success=False,
            error_message="Text cannot be empty"
        )
    
    return parse_financial_text(request.text)


@app.post("/save-transaction")
def save_transaction(transaction_data: dict) -> dict:
    """Save a parsed transaction to existing CSV files."""
    try:
        import csv
        from datetime import datetime
        import uuid
        
        # Generate unique transaction ID
        transaction_id = str(uuid.uuid4())
        now = datetime.now()
        date_str = now.strftime('%m/%d/%Y')
        time_str = now.strftime('%H:%M:%S')
        
        # Create transaction record for zamanbank_transactions.csv
        transaction_record = {
            'date': date_str,
            'time': time_str,
            'amount': transaction_data.get('amount', 0.0),
            'transaction_id': transaction_id,
            'transactioner_id': f"CUST-{now.strftime('%Y%m%d%H%M%S')}"
        }
        
        # Create finance record for zamanbank_finances.csv
        finance_record = {
            'transaction_id': transaction_id,
            'category': transaction_data.get('category', 'Other'),
            'amount_money': transaction_data.get('amount', 0.0),
            'item': transaction_data.get('item', 'РџРѕРєСѓРїРєР°'),
            'pcs': transaction_data.get('quantity', 1),
            'category_ru': transaction_data.get('category', 'РџСЂРѕС‡РµРµ')
        }
        
        # Append to zamanbank_transactions.csv
        transactions_path = BASE_DIR / "zamanbank_transactions.csv"
        with open(transactions_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['date', 'time', 'amount', 'transaction_id', 'transactioner_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(transaction_record)
        
        # Append to zamanbank_finances.csv
        finances_path = BASE_DIR / "zamanbank_finances.csv"
        with open(finances_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['transaction_id', 'category', 'amount_money', 'item', 'pcs', 'category_ru']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(finance_record)
        
        return {"success": True, "message": "Transaction saved successfully", "transaction_id": transaction_id}
        
    except Exception as e:
        return {"success": False, "message": f"Error saving transaction: {str(e)}"}


@app.get("/get-parsed-transactions")
def get_parsed_transactions() -> dict:
    """Get all parsed transactions from existing CSV files."""
    try:
        import csv
        
        # Read transactions CSV
        transactions = []
        with open(TRANSACTIONS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            transactions_data = list(reader)
        
        # Read finances CSV
        with open(FINANCES_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            finances_data = list(reader)
        
        # Create a lookup for finances by transaction_id
        finances_lookup = {f['transaction_id']: f for f in finances_data}
        
        # Merge the data
        merged_transactions = []
        for t in transactions_data:
            finance_info = finances_lookup.get(t['transaction_id'], {})
            merged_transaction = {
                'transaction_id': t['transaction_id'],
                'date': t['date'],
                'time': t['time'],
                'amount': float(t['amount']),
                'transactioner_id': t['transactioner_id'],
                'category': finance_info.get('category', 'Other'),
                'amount_money': float(finance_info.get('amount_money', t['amount'])),
                'item': finance_info.get('item', 'РџРѕРєСѓРїРєР°'),
                'pcs': int(finance_info.get('pcs', 1)),
                'category_ru': finance_info.get('category_ru', 'РџСЂРѕС‡РµРµ')
            }
            merged_transactions.append(merged_transaction)
        
        return {"transactions": merged_transactions}
        
    except Exception as e:
        return {"transactions": [], "error": str(e)}


@app.get("/healthz")
def healthcheck() -> dict:
    return {"status": "ok"}


@app.get("/test-parse")
def test_parse() -> dict:
    """Test endpoint to verify parsing functionality"""
    test_text = "СЏ РєСѓРїРёР» РєРѕРєР° РєРѕР»Сѓ Р·Р° 450 С‚РµРЅРіРµ СЃРµРіРѕРґРЅСЏ"
    result = parse_financial_text(test_text)
    return {
        "test_text": test_text,
        "parsed_result": result.model_dump(),
        "api_status": "working"
    }
def normalize_amount(value: Optional[str]) -> float:
    """Convert raw numeric text into float amount."""
    if not value:
        return 0.0
    cleaned = value.replace(" ", "").replace("\u00A0", "").replace(",", ".")
    if cleaned.count(".") > 1:
        first = cleaned.find(".")
        cleaned = cleaned[: first + 1] + cleaned[first + 1 :].replace(".", "")
    try:
        return round(float(cleaned), 2)
    except ValueError:
        return 0.0


def _clean_item_fragment(fragment: str) -> str:
    lowered = fragment.lower()
    tokens = [
        token
        for token in re.split(r"[\s,\.;:!?\-\(\)\[\]\{\}]+", lowered)
        if token and token not in STOP_WORDS and not token.isdigit()
    ]
    if not tokens:
        return ""
    return " ".join(tokens)


def extract_item(text: str, amount_match: Optional[re.Match[str]]) -> str:
    working = text
    if amount_match:
        start, end = amount_match.span()
        working = f"{text[:start]} {text[end:]}"

    working = re.sub(
        r"\b(?:С‚РµРЅРіРµ|С‚РµТЈРіРµ|С‚Рі|tg|kzt|в‚ё)\b", " ", working, flags=re.IGNORECASE
    )
    working = re.sub(r"\d+(?:[.,]\d+)?", " ", working)

    lowered = working.lower()
    item_patterns = [
        r"(?:РЅР°|Р·Р°|РґР»СЏ)\s+(?P<item>[Р°-СЏС‘a-z0-9\s\-]{2,})",
        r"(?:РєСѓРїРёР»(?:Р°|Рё)?|РєСѓРїР»СЋ|РєСѓРїРёРј|РІР·СЏР»(?:Р°|Рё)?|РєРѕРЅРІРµСЂС‚РёСЂРѕРІР°Р»|РѕРїР»Р°С‚РёР»(?:Р°|Рё)?|Р·Р°РєР°Р·Р°Р»(?:Р°|Рё)?)\s+(?P<item>[Р°-СЏС‘a-z0-9\s\-]{2,})",
        r"(?:СЂР°СЃС…РѕРґ|РїРѕРєСѓРїРєР°)\s+(?P<item>[Р°-СЏС‘a-z0-9\s\-]{2,})",
    ]

    for pattern in item_patterns:
        match = re.search(pattern, lowered)
        if match and "item" in match.groupdict():
            span = match.span("item")
            candidate = working[span[0] : span[1]]
            cleaned = _clean_item_fragment(candidate)
            if cleaned:
                return cleaned.title()

    cleaned_full = _clean_item_fragment(working)
    if cleaned_full:
        return cleaned_full.title()

    return "Unknown item"


def classify_item(item: str) -> dict:
    lowered = item.lower()
    for rule in CATEGORY_RULES:
        if any(keyword in lowered for keyword in rule["keywords"]):
            return {"category": rule["category"], "category_ru": rule["category_ru"]}
    return DEFAULT_CATEGORY


