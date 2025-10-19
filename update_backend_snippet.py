# -*- coding: utf-8 -*-
from pathlib import Path
import re

path = Path('main_app/backend/main.py')
data = path.read_text(encoding='utf-8')

block_re = re.compile(r"KNOWLEDGE_CHAPTER_KEYS = .*?KNOWLEDGE_ID_KEYS = \[\"id\", \"ID\", \"row_id\", \"rowId\"\]\n\n\n", re.S)
block_new = '''KNOWLEDGE_CHAPTER_KEYS = ["глава", "\\ufeffглава", "chapter", "Chapter", "Глава"]
KNOWLEDGE_TOPIC_KEYS = [
    "общий вопрос/тема",
    "общий вопрос / тема",
    "вопрос/тема",
    "вопрос / тема",
    "тема",
    "question",
]
KNOWLEDGE_EXPLANATION_KEYS = [
    "ответ/объяснение",
    "ответ/обьяснение",
    "объяснение",
    "ответ",
    "explanation",
    "answer",
]
KNOWLEDGE_TYPE_KEYS = ["тип", "type"]
KNOWLEDGE_ID_KEYS = ["id", "ID", "row_id", "rowId"]


'''
if not block_re.search(data):
    raise SystemExit('knowledge block not found')
data = block_re.sub(lambda _: block_new, data, count=1)

func_re = re.compile(r"def build_citations_payload\(retrieved: Dict\[str, List\[Dict]]\) -> List\[CitationInfo]:\n(?:    .*\n)+?    return citations\n\n", re.S)
func_new = '''def resolve_field(metadata: Dict[str, object], row: Dict[str, object], candidates: List[str]) -> Optional[str]:
    for key in candidates:
        if key in metadata:
            value = coerce_optional_str(metadata.get(key))
            if value:
                return value

    text_value = coerce_optional_str(row.get("text"))
    if text_value:
        for key in candidates:
            pattern = re.compile(rf"{re.escape(key)}\\s*:\\s*([^|]+)")
            match = pattern.search(text_value)
            if match:
                extracted = coerce_optional_str(match.group(1))
                if extracted:
                    return extracted
    return None


def build_citations_payload(retrieved: Dict[str, List[Dict]]) -> List[CitationInfo]:
    citations: List[CitationInfo] = []
    seen: set[str] = set()
    knowledge_rows = retrieved.get("knowledge") or []

    for row in knowledge_rows:
        metadata = row.get("metadata") or {}
        chapter = resolve_field(metadata, row, KNOWLEDGE_CHAPTER_KEYS)
        topic = resolve_field(metadata, row, KNOWLEDGE_TOPIC_KEYS)
        explanation = resolve_field(metadata, row, KNOWLEDGE_EXPLANATION_KEYS)
        record_type = resolve_field(metadata, row, KNOWLEDGE_TYPE_KEYS)
        record_id = resolve_field(metadata, row, KNOWLEDGE_ID_KEYS)

        dedupe_key = chapter or record_id or coerce_optional_str(row.get("id")) or coerce_optional_str(row.get("text"))
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        citations.append(
            CitationInfo(
                id=record_id,
                source="knowledge",
                chapter=chapter,
                topic=topic,
                explanation=explanation,
                type=record_type,
            )
        )

    return citations

'''
if not func_re.search(data):
    raise SystemExit('citations function not found')
data = func_re.sub(lambda _: func_new, data, count=1)

path.write_text(data, encoding='utf-8')
