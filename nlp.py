 import re
import unicodedata

from langdetect import detect, LangDetectException
import spacy
import yake

import config

_nlp = spacy.load("en_core_web_sm")

HEADING_PATTERNS = [
    re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,&/-]{2,80}$"),
    re.compile(r"^[A-Z][A-Z0-9 ,&/-]{3,80}$"),
    re.compile(r"^(Chapter|Section|Appendix|Part)\s+\d+[:.]?\s*.*$", re.IGNORECASE),
]

OCR_SUBSTITUTIONS = [
    (re.compile(r"(?<=[a-zA-Z])-\n(?=[a-zA-Z])"), ""),
    (re.compile(r"\n(?=[a-z])"), " "),
    (re.compile(r"[ \t]{2,}"), " "),
    (re.compile(r"(?<=[a-zA-Z])0(?=[a-zA-Z])"), "o"),
    (re.compile(r"(?<=[a-zA-Z])1(?=[a-zA-Z])"), "l"),
]

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
URL_PATTERN = re.compile(r"https?://[^\s]+")
DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b"
)


def normalize_unicode(text):
    return unicodedata.normalize(config.UNICODE_NORMALIZATION_FORM, text)


def correct_ocr_noise(text):
    for pattern, replacement in OCR_SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    return text


def clean_text(text):
    text = normalize_unicode(text)
    text = correct_ocr_noise(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_language(text):
    if not config.DETECT_LANGUAGE:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def split_sentences(text):
    doc = _nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def detect_headings(raw_text):
    headings = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        for level, pattern in enumerate(HEADING_PATTERNS, start=1):
            if pattern.match(stripped):
                headings.append({
                    "line_number": line_number,
                    "text": stripped,
                    "level": level,
                })
                break
    return headings


def extract_entities(text):
    doc = _nlp(text)
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
    ]


def extract_keywords(text, max_keywords=10):
    if not text:
        return []
    extractor = yake.KeywordExtractor(lan="en", n=2, top=max_keywords)
    keywords = extractor.extract_keywords(text)
    return [kw for kw, score in sorted(keywords, key=lambda x: x[1])]


def extract_metadata(text):
    return {
        "emails": list(set(EMAIL_PATTERN.findall(text))),
        "urls": list(set(URL_PATTERN.findall(text))),
        "dates": list(set(DATE_PATTERN.findall(text))),
    }


def rewrite_query(query):
    cleaned = clean_text(query)
    doc = _nlp(cleaned)
    keywords = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and token.text.strip()
    ]
    expansion = " ".join(dict.fromkeys(keywords))
    return f"{cleaned} {expansion}".strip()


def process_pages(pages):
    processed = []
    current_section = None

    for page in pages:
        raw_text = page["text"]
        headings = detect_headings(raw_text)
        if headings:
            current_section = headings[-1]["text"]

        cleaned = clean_text(raw_text)

        processed.append({
            "page_number": page["page_number"],
            "source": page["source"],
            "text": cleaned,
            "sentences": split_sentences(cleaned) if cleaned else [],
            "language": detect_language(cleaned) if cleaned else None,
            "headings": headings,
            "section": current_section,
            "entities": extract_entities(cleaned) if cleaned else [],
            "keywords": extract_keywords(cleaned) if cleaned else [],
            "metadata": extract_metadata(raw_text),
        })

    return processed
