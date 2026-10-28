from __future__ import annotations

import json
import re
import unicodedata

from core.memory import contains_term, term_pattern

def _normalize_punctuation(text, protected=()):
    spans = sorted(match.span() for term in protected
                   for match in re.finditer(term_pattern(term), text, re.IGNORECASE))
    pieces, cursor = [], 0
    for start, end in [*spans, (len(text), len(text))]:
        pieces.extend(unicodedata.normalize("NFKC", char)
                      if unicodedata.category(char).startswith("P") else char
                      for char in text[cursor:start])
        if end > cursor:
            pieces.append(text[max(cursor, start):end])
            cursor = end
    return "".join(pieces)


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_response(raw):
    if not isinstance(raw, str):
        raise ValueError("Model response is not text")
    raw = raw.strip()
    if raw.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if not match:
            raise ValueError("Incomplete JSON code fence")
        raw = match.group(1)
    return json.loads(raw, object_pairs_hook=_strict_object)


def parse_group_response(raw, expected_ids):
    """单行失败保留其他有效译文；未知 ID 拒绝整组。"""
    expected = set(expected_ids)
    try:
        obj = _json_response(raw)
        if not isinstance(obj, dict) or not isinstance(obj.get("translations"), list):
            raise ValueError("Expected an object containing translations[]")
        rows = obj["translations"]
        if any(not isinstance(row, dict) or not isinstance(row.get("id"), str)
               or row["id"] not in expected for row in rows):
            raise ValueError("Unexpected or invalid bubble ID")
    except (ValueError, TypeError) as exc:
        return {}, {key: f"Invalid structured response: {exc}" for key in expected}, []
    result, errors, seen = {}, {}, set()
    for row in rows:
        key = row["id"]
        if key in seen:
            errors[key] = "Duplicate bubble ID"
            result.pop(key, None)
            continue
        seen.add(key)
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            errors[key] = "Empty or invalid translation"
        else:
            result[key] = text.strip()
    for key in expected - seen:
        errors[key] = "Missing bubble ID"
    candidates = obj.get("terms", [])
    return result, errors, candidates if isinstance(candidates, list) else []


def _parse_individual(raw, block_id):
    if not isinstance(raw, str) or not raw.strip():
        return "", "Empty translation", []
    text = raw.strip()
    if text.startswith(("{", "[", "```")):
        result, errors, terms = parse_group_response(text, [block_id])
        return result.get(block_id, ""), errors.get(block_id, ""), terms
    return text, "", []


def terminology_conflicts(source, translated, terms):
    """提示缺失译名，不替换译文；合理的代词也可能触发提示。"""
    return [f"Expected term {term['source']!r} → {term['target']!r} is absent"
            for term in terms if any(contains_term(source, spelling)
                                    for spelling in (term["source"], *term["aliases"]))
            and not contains_term(translated, term["target"], ignore_case=True)]


def _has_japanese_text(text):
    # 用规范化副本识别半角假名；不改译文，日文标点可保留。
    for char in unicodedata.normalize("NFKC", text):
        name = unicodedata.name(char, "")
        if char.isalpha() and name.startswith(("HIRAGANA", "KATAKANA", "CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
            return True
    return False


def _has_prohibition(text):
    """只识别明确的禁止表达；普通否定或含糊转述不能作为依据。"""
    text = text.casefold().replace("’", "'")
    permissive = (r"\bno\s+(?:problem|need|reason\s+not)\b"
                  r"|\bnot\s+(?:wrong|bad|forbidden|prohibited|banned|illegal|necessary|required|only)\b"
                  r"|\b(?:isn't|aren't|wasn't|weren't)\s+(?:wrong|bad|forbidden|prohibited|banned|illegal)\b"
                  r"|\b(?:do\s+not|don't|need\s+not|needn't)\s+(?:have|need)\s+to\b"
                  r"|\b(?:cannot|can't|can\s+not)\s+help\b")
    if re.search(permissive, text):
        return False
    return bool(re.search(
        r"\b(?:do|must|should|shall|may|can)\s+not\b"
        r"|\b(?:don't|mustn't|shouldn't|shan't|can't|cannot|never)\b"
        r"|\b(?:forbidden|prohibited|banned)\b"
        r"|\bnot\s+(?:allowed|permitted)\b"
        r"|\b(?:avoid|refrain\s+from|stop)\s+\w+ing\b", text))

