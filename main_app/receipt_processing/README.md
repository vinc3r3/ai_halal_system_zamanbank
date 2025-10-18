# Receipt Extractor (LlamaExtract + CSV)

A lightweight command-line tool for **OCR + structured extraction** of payment receipts using [LlamaParse/LlamaExtract](https://developers.llamaindex.ai/).  
It parses image/PDF receipts in Russian or English, returns structured JSON per receipt, and builds one combined `receipts.csv`.

---

## Features
- Uses **LlamaExtract** for OCR and schema-based field extraction  
- Outputs both **per-receipt JSON** and **aggregate CSV**  
- Enforces a **fixed category taxonomy** with English–Russian pairs  
- Automatically infers categories if missing  
- Handles Russian & English text equally  
- Runs on all common receipt file types (`.png`, `.jpg`, `.jpeg`, `.pdf`, `.tif`)

---

## Installation

```bash
git clone <your-repo-url>
cd receipt_processing
pip install -r requirements.txt
````


Then create a `.env` file with your API key:

```
LLAMA_CLOUD_API_KEY=llx-your-key-here
```

---

## Usage

```bash
python receipts_extract_csv.py --src ./receipts --out ./out --recursive --ru
```

### Arguments

| Flag           | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `--src`        | Directory with receipt images or PDFs                                       |
| `--out`        | Output directory (JSON + CSV will be written here)                          |
| `--env`        | Path to `.env` file with `LLAMA_CLOUD_API_KEY` (default `.env`)             |
| `--mode`       | Extraction mode: `FAST`, `BALANCED`, or `MULTIMODAL` (default `MULTIMODAL`) |
| `--recursive`  | Recurse into subdirectories                                                 |
| `--ru`         | Bias extraction toward Russian text                                         |
| `--agent-name` | Optional custom agent name                                                  |
| `--reuse`      | Reuse existing agent if it already exists in Llama Cloud                    |

---

## Output Example

### `12345.json`

```json
{
  "transaction_id": "12345",
  "items": [
    {
      "item": "Bread",
      "pcs": 1,
      "amount_money": 30.00,
      "category": "grocery",
      "category_ru": "Продукты"
    },
    {
      "item": "Milk 2.5%",
      "pcs": 1,
      "amount_money": 85.50,
      "category": "grocery",
      "category_ru": "Продукты"
    }
  ]
}
```

### `receipts.csv`

```csv
transaction_id,item,pcs,amount_money,category,category_ru
12345,Bread,1,30.0,grocery,Продукты
12345,Milk 2.5%,1,85.5,grocery,Продукты
```

---

## Category Mapping

| category (EN) | category_ru (RU) |
| ------------- | ---------------- |
| utilities     | ЖКХ              |
| household     | Все для дома     |
| allowance     | Карманные        |
| grocery       | Продукты         |
| transport     | Транспорт        |
| restaurant    | Еда              |
| entertainment | Развлечения      |
| telecom       | Связь            |
| health        | Здоровье         |
| auto          | Авто             |
| sports        | Спорт            |
| kids          | Дети             |
| travel        | Путешествия      |
| clothing      | Одежда           |
| beauty        | Красота          |
| gifts         | Подарки          |
| other         | Прочее           |

---
