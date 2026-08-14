import re

from transformers import AutoTokenizer

import config

_tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)

MAX_CHUNK_TOKENS = 512
MIN_CHUNK_TOKENS = 128
CHUNK_OVERLAP_TOKENS = 64

NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+")
LIST_MARKER = re.compile(r"(?:^|\s)(?:[-\u2022*]|\d+[.)])\s")


def count_tokens(text):
    return len(_tokenizer.encode(text, add_special_tokens=False))


def _heading_depth(text):
    match = NUMBERED_HEADING.match(text)
    if match:
        return match.group(1).count(".") + 1
    return 1


def _match_heading(sentence, headings):
    normalized = sentence.strip().lower()
    for heading in headings:
        if normalized.startswith(heading["text"].strip().lower()):
            return heading["text"]
    return None


def _group_list_units(sentences):
    units = []
    buffer = []
    for sentence in sentences:
        if LIST_MARKER.search(sentence):
            buffer.append(sentence)
            continue
        if buffer:
            units.append(" ".join(buffer))
            buffer = []
        units.append(sentence)
    if buffer:
        units.append(" ".join(buffer))
    return units


def _split_oversized_unit(unit_text):
    list_items = [item for item in LIST_MARKER.split(unit_text) if item.strip()]
    if len(list_items) > 1:
        pieces = []
        for item in list_items:
            pieces.extend(_split_oversized_unit(item.strip()))
        return pieces

    token_ids = _tokenizer.encode(unit_text, add_special_tokens=False)
    if len(token_ids) <= MAX_CHUNK_TOKENS:
        return [unit_text]

    pieces = []
    start = 0
    while start < len(token_ids):
        end = min(start + MAX_CHUNK_TOKENS, len(token_ids))
        pieces.append(_tokenizer.decode(token_ids[start:end]))
        start = end - CHUNK_OVERLAP_TOKENS if end < len(token_ids) else end
    return pieces


def _take_overlap(units, budget):
    overlap = []
    tokens_used = 0
    for unit in reversed(units):
        unit_tokens = count_tokens(unit)
        if tokens_used + unit_tokens > budget and overlap:
            break
        overlap.insert(0, unit)
        tokens_used += unit_tokens
    return overlap


def _merge_small_chunks(chunks):
    merged = []
    for chunk in chunks:
        if (
            merged
            and chunk["token_count"] < MIN_CHUNK_TOKENS
            and merged[-1]["page_number"] == chunk["page_number"]
            and merged[-1]["section"] == chunk["section"]
        ):
            previous = merged[-1]
            previous["text"] = f"{previous['text']} {chunk['text']}"
            previous["token_count"] = count_tokens(previous["text"])
            continue
        merged.append(chunk)

    for index, chunk in enumerate(merged):
        chunk["chunk_id"] = index

    return merged


def chunk_pages(pages):
    heading_stack = []
    chunks = []
    chunk_id = 0

    for page in pages:
        headings = page.get("headings", [])
        units = _group_list_units(page.get("sentences", []))

        buffer_units = []
        buffer_tokens = 0

        def current_heading():
            return heading_stack[-1]["text"] if heading_stack else page.get("section")

        def current_section():
            return heading_stack[0]["text"] if heading_stack else page.get("section")

        def flush():
            nonlocal chunk_id, buffer_units, buffer_tokens
            if not buffer_units:
                return []
            text = " ".join(buffer_units)
            chunk = {
                "chunk_id": chunk_id,
                "document_name": page.get("source"),
                "page_number": page.get("page_number"),
                "heading": current_heading(),
                "section": current_section(),
                "text": text,
                "token_count": buffer_tokens,
                "language": page.get("language", "unknown"),
                "keywords": page.get("keywords", []),
            }
            chunk_id += 1
            emitted = [chunk]
            buffer_units = []
            buffer_tokens = 0
            return emitted

        for unit_text in units:
            heading_match = _match_heading(unit_text, headings)
            if heading_match:
                chunks.extend(flush())
                depth = _heading_depth(heading_match)
                heading_stack = [h for h in heading_stack if h["depth"] < depth]
                heading_stack.append({"text": heading_match, "depth": depth})

            unit_tokens = count_tokens(unit_text)

            if unit_tokens > MAX_CHUNK_TOKENS:
                chunks.extend(flush())
                for piece in _split_oversized_unit(unit_text):
                    piece_tokens = count_tokens(piece)
                    chunks.append({
                        "chunk_id": chunk_id,
                        "document_name": page.get("source"),
                        "page_number": page.get("page_number"),
                        "heading": current_heading(),
                        "section": current_section(),
                        "text": piece,
                        "token_count": piece_tokens,
                        "language": page.get("language", "unknown"),
                        "keywords": page.get("keywords", []),
                    })
                    chunk_id += 1
                continue

            if buffer_tokens + unit_tokens > MAX_CHUNK_TOKENS:
                previous_units = list(buffer_units)
                chunks.extend(flush())
                buffer_units = _take_overlap(previous_units, CHUNK_OVERLAP_TOKENS)
                buffer_tokens = sum(count_tokens(u) for u in buffer_units)

            buffer_units.append(unit_text)
            buffer_tokens += unit_tokens

        chunks.extend(flush())

    return _merge_small_chunks(chunks)
