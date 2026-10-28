"""VNTL 原生提示词与 CUDA llama.cpp 服务；跨页复用，由 ModelPool 关闭。"""
from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from importlib.util import find_spec
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
from typing import Any
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from core.runtime import ROOT, Config, TextBlock
from core.memory import GlossaryMemory, term_pattern
from .shared import InvalidTranslationInput


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Local translation server redirects are disabled")


def _header(role):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n"


def prompt_id(block_id):
    """为长 ID 或非 ASCII ID 生成固定别名，页面 ID 保持不变。

    VNTL 会把部分标签当人名改写；响应按别名映射，冲突时拒绝，不按行序猜测。
    """
    if re.fullmatch(r"[A-Za-z0-9-]{1,32}", block_id):
        return block_id
    return "b-" + hashlib.sha256(block_id.encode("utf-8")).hexdigest()[:10]


def _fragment_pair(left, right, direction):
    """只合并位置相邻且有未完句或短假名停顿特征的片段，不改 OCR。"""
    first, second = left.source_text.strip(), right.source_text.strip()
    if (not first or not second or (re.search(r'[。．.!！?？」』）)]$', first)
            and not re.search(r'(?:\.{2,}|．{2,})$', first))
            or second.startswith(('「', '『', '（', '('))):
        return False
    a = re.sub(r'[\s、，,…‥⋯.．]+$', '', first)
    b = re.sub(r'[\s、，,…‥⋯.．!！?？。]+$', '', second)
    if not a or not b:
        return False
    if a in {"こんにちは", "こんばんは", "ありがとう", "おはよう", "さようなら", "おやすみ",
             "どうも", "はい", "いいえ", "うん", "ええ", "なに", "そうなの"}:
        return False
    kana_pause = (2 <= len(a) <= 6 and 1 <= len(b) <= 4 and len(a) + len(b) <= 8
                  and re.fullmatch(r'[ぁ-ゖァ-ヺー]+', a + b)
                  and len(first) > len(a))
    # 不能仅凭末尾假名判残句，例如「こんにちは」。
    unfinished = (re.search(r'(?:について|として|ために|ながら|けれども|けれど|ならば|ければ)$', a)
                  or len(a) >= 4 and a.endswith("を") and re.search(r'[一-鿿ァ-ヺ]', a))
    if not (kana_pause or unfinished):
        return False
    # 简短应答可能来自另一说话人，应保持独立。
    if re.match(r'^(?:はい|いいえ|うん|ええ|ありがとう|こんにちは|おはよう)(?:$|[、，。!！?？…])', second):
        return False
    x1, y1, x2, y2 = left.box
    u1, v1, u2, v2 = right.box
    w, h, rw, rh = x2-x1, y2-y1, u2-u1, v2-v1
    if min(w, h, rw, rh) <= 0:
        return False
    # 竖排用框宽衡量间距，相邻两列尺寸接近且不能重叠。
    if h >= 1.4*w and rh >= 1.4*rw and max(w, rw) <= 2*min(w, rw):
        gap = x1-u2 if direction == 'rtl' else u1-x2
        return 0 <= gap <= 1.25*min(w, rw) and min(y2, v2)-max(y1, v1) >= .5*min(h, rh)
    # 横排行始终从上往下，与页面左右阅读方向无关。
    if w >= 1.4*h and rw >= 1.4*rh and max(h, rh) <= 2*min(h, rh):
        return 0 <= v1-y2 <= 1.25*min(h, rh) and min(x2, u2)-max(x1, u1) >= .5*min(w, rw)
    return False


def _continuation_eligible(left: TextBlock, right: TextBlock,
                           bindings: dict[str, dict[str, Any]]) -> bool:
    failed_ocr = any(block.status == "failed" and block.error.startswith("OCR:")
                     for block in (left, right))
    roles = [bindings.get(block.block_id, {}) for block in (left, right)]
    different_speakers = (all(role.get("confirmed") and role.get("character_id") not in (None, "", "unknown")
                              for role in roles)
                          and roles[0]["character_id"] != roles[1]["character_id"])
    return (not failed_ocr and not different_speakers
            and not left.metadata.get("continuation") and not right.metadata.get("continuation"))


def _continuation_kind(first: str, second: str) -> str | None:
    if (re.search(r"(?:んじゃ|ちゃ|ては|では)$", first)
            and re.match(r"(?:(?:ダメ|駄目|だめ)(?=$|[っッ!！?？。．、，…\s]|です|だ|よ|ね|な|ぞ|わ|か)"
                         r"|いけない|いけません|ならない|なりません|ならぬ)", second)):
        return "prohibition"
    elif (re.search(r"(?:ても|でも)$", first)
            and re.match(r"(?:いい|良い|よい)(?=$|[っッ!！?？。．、，…\s]|です|だ|よ|ね|ん|の|から)", second)):
        return "permission"
    return None


def continuation_pairs(blocks: list[TextBlock], bindings: dict[str, dict[str, Any]] | None = None,
                       *, fragments: bool = True, direction: str = "rtl") -> list[tuple[int, int, str]]:
    """单次扫描选出不重叠的相邻续句，复杂度 O(n + 原文字符数)，n 为气泡数。

    优先匹配明确语法；其他短片段须同时满足位置、残句和孤立配对条件。
    完整句、闭合引号、OCR 失败或已确认不同说话人的气泡保持独立。
    bindings 是 SQLite 中本页的角色绑定表，只使用已确认的角色。
    """
    if direction not in {"rtl", "ltr"}:
        raise ValueError("reading direction must be rtl or ltr")
    bindings = bindings or {}
    pairs, fragment_edges, index = [], [], 0
    while index + 1 < len(blocks):
        left, right = blocks[index:index + 2]
        first = re.sub(r"(?:(?:[.．]{2,}|[‥…⋯]+)\s*)+$", "", left.source_text.strip())
        second = right.source_text.lstrip()
        eligible = _continuation_eligible(left, right, bindings)
        kind = _continuation_kind(first, second) if eligible else None
        if kind:
            pairs.append((index, index + 1, kind))
            index += 2
        else:
            if fragments and eligible and _fragment_pair(left, right, direction):
                fragment_edges.append(index)
            index += 1
    occupied = {i for left, right, _ in pairs for i in (left, right)}
    edges = set(fragment_edges)
    pairs.extend((i, i+1, "fragment") for i in fragment_edges
                 if i-1 not in edges and i+1 not in edges and i not in occupied and i+1 not in occupied)
    by_start = {pair[0]: pair for pair in pairs}
    return [by_start[i] for i in range(len(blocks)) if i in by_start]


_word_parser = threading.local()


@lru_cache(maxsize=512)
def _word_tokens(text):
    """复用 Manga-OCR 的词典，每个线程延迟创建一个分词器。

    只缓存不可变特征，避免保留分词器节点；词典不可用时保持原文。
    """
    try:
        if not hasattr(_word_parser, "tagger"):
            import fugashi
            import unidic_lite

            _word_parser.tagger = fugashi.Tagger(f'-d "{unidic_lite.DICDIR}"')
        tokens = tuple((node.surface, node.feature.pos1, node.feature.pos2,
                        bool(node.is_unk), node.feature.lemma)
                       for node in _word_parser.tagger(text))
        return tokens if "".join(token[0] for token in tokens) == text else None
    except (ImportError, OSError, RuntimeError, AttributeError):
        return None


def normalize_word_boundary(left, right, terms=()):
    """只在已关联片段中去掉词典确认的词内逗号，不改原气泡。

    拼成词还不够：「はな + そう」虽能成词，「そう」也可能是独立应答。
    歧义、未知词、已确认术语和普通标点保持原样。
    """
    first, second = left.source_text.strip(), right.source_text.strip()
    original = first + second
    a = re.fullmatch(r'([ぁ-ゖァ-ヺー]{2,6})([、，,])', first)
    b = re.fullmatch(r'([ぁ-ゖァ-ヺー]{1,4})([、，,。.!?！？…‥⋯．]*)', second)
    if not a or not b or len(a[1]) + len(b[1]) > 8:
        return original, None
    if any(spelling in {a[1], b[1]} for term in terms
           if term.get("locked") or term.get("status") == "confirmed"
           for spelling in (term["source"], *term.get("aliases", []))):
        return original, None
    suffix = _word_tokens(b[1])
    if not suffix or any(unknown or pos not in {"助詞", "助動詞", "接尾辞"}
                         for _, pos, _, unknown, _ in suffix):
        return original, None
    tokens = _word_tokens(a[1] + b[1])
    if not tokens:
        return original, None
    offset, boundary = 0, len(a[1])
    for surface, pos, subtype, unknown, lemma in tokens:
        end = offset + len(surface)
        # 名词或副词复合词可能误吞正常停顿，例如「いか、に」。
        if (offset < boundary < end and not unknown and subtype != "固有名詞"
                and pos in {"動詞", "形容詞", "代名詞"}):
            return a[1] + second, {"method": "dictionary_intra_word_comma_v1",
                "boundary_after": boundary, "removed": a[2], "word": surface,
                "lemma": lemma, "analyzer": "fugashi/unidic_lite"}
        offset = end
    return original, None


def translation_source(block):
    """只在构造提示词时使用仍匹配当前原文的预处理文本。"""
    unit = block.metadata.get("continuation", {}) if isinstance(block.metadata, dict) else {}
    if isinstance(unit, dict) and unit.get("source_text") == block.source_text:
        prepared = unit.get("translation_source_text")
        if isinstance(prepared, str) and prepared.strip():
            return prepared
    return block.source_text


def _split_constraints(text: str, left: TextBlock, right: TextBlock, terms: list[dict[str, Any]]
                       ) -> tuple[list[tuple[int, int]], int, int]:
    from core.memory import contains_term

    protected, lower, upper = [], 0, len(text)
    for term in terms:
        target = term["target"]
        spans = [match.span() for match in re.finditer(term_pattern(target), text, re.IGNORECASE)]
        protected.extend(spans)
        if not term.get("locked"):
            continue
        source_hits = [any(contains_term(block.source_text, name)
                           for name in (term["source"], *term.get("aliases", [])))
                       for block in (left, right)]
        if any(source_hits) and not spans:
            raise ValueError(f"Locked term {target!r} is missing from the continued sentence")
        if source_hits[0]:
            lower = max(lower, min(end for _, end in spans))
        if source_hits[1]:
            upper = min(upper, max(start for start, _ in spans))
    return protected, lower, upper


def _split_fraction(left: TextBlock, right: TextBlock) -> float:
    areas = []
    for block in (left, right):
        box = block.safe_box or block.metadata.get("layout", {}).get("safe_box") or block.box
        areas.append(max(1, (box[2] - box[0]) * (box[3] - box[1])))
    source_lengths = [max(1, len(re.sub(r"\s+", "", block.source_text))) for block in (left, right)]
    return .65 * source_lengths[0] / sum(source_lengths) + .35 * areas[0] / sum(areas)


def split_sentence(text: str, left: TextBlock, right: TextBlock, terms: list[dict[str, Any]],
                   *, fixed_parts: dict[str, str] | None = None) -> list[str]:
    """按词界切分生成整句，保留词序和标点，只去掉切分处空白。

    译名保持完整，锁词留在含对应原词或别名的气泡，无法放置则报错。
    fixed_parts 固定未选邻泡的译文；排版面积变化不能挪动这些词。
    """
    text = text.strip()
    fixed_parts = fixed_parts or {}
    fixed = {key: " ".join(value.split()) for key, value in fixed_parts.items()}
    tokens = list(re.finditer(r"\S+", text))
    word_counts = [0]
    for token in tokens:
        word_counts.append(word_counts[-1] + bool(re.search(r"\w", token[0])))
    if word_counts[-1] < 2:
        raise ValueError("Continued sentence is too short for two nonempty English bubbles")
    minimum = 2 if not fixed and word_counts[-1] >= 4 else 1
    protected, lower, upper = _split_constraints(text, left, right, terms)
    fraction = _split_fraction(left, right)
    hanging = {"a", "an", "the", "this", "that", "my", "your", "his", "her", "our", "their",
               "to", "of", "for", "with", "from", "at", "in", "on", "and", "or", "but",
               "not", "never", "must", "should", "can", "do", "does", "did", "don't", "can't"}
    candidates = []
    for index in range(1, len(tokens)):
        if word_counts[index] < minimum or word_counts[-1] - word_counts[index] < minimum:
            continue
        cut = tokens[index - 1].end()
        if not lower <= cut <= upper or any(start < cut < end for start, end in protected):
            continue
        if any(block.block_id in fixed and " ".join(part.split()) != fixed[block.block_id]
               for block, part in ((left, text[:cut]), (right, text[tokens[index].start():]))):
            continue
        score = abs(cut / len(text) - fraction)
        score += .12 * abs(word_counts[index] / word_counts[-1] - fraction)
        before = tokens[index - 1][0]
        after = tokens[index][0].strip('"\'“”‘’.,!?;:').lower()
        if re.search(r"[,;:!?—](?:[\"'”’])?$", before):
            score -= .18
        if after in {"and", "but", "because", "although", "if", "when"}:
            score -= .08
        if before.strip('"\'“”‘’.,!?;:').lower().replace("’", "'") in hanging:
            score += .30
        candidates.append((score, cut, tokens[index].start()))
    if not candidates:
        if fixed:
            kept = "; ".join(f"[{key}]: {value}" for key, value in fixed.items())
            raise ValueError("Cannot split continued sentence while preserving the existing neighboring "
                             "translation; translate both bubbles together or retain exactly: " + kept)
        raise ValueError("Cannot split continued sentence without breaking words or moving locked terms")
    _, end, start = min(candidates)
    return [text[:end], text[start:]]


def locked_mentions(text, terms):
    """按当前原文的精确匹配范围解析锁词，重叠范围的译名有歧义则拒绝。"""
    matches = []
    for term in terms:
        if not term["locked"]:
            continue
        for spelling in (term["source"], *term["aliases"]):
            matches.extend((match.start(), match.end(), term["target"])
                           for match in re.finditer(term_pattern(spelling), text))
    resolved = []
    for start, end, name in sorted(matches):
        if resolved and start < resolved[-1][1]:
            previous = resolved[-1]
            if name != previous[2]:
                raise InvalidTranslationInput(f"Ambiguous locked names for overlapping source {text[previous[0]:max(end, previous[1])]!r}: "
                                 f"{previous[2]!r} / {name!r}; correct the glossary or aliases")
            resolved[-1] = (previous[0], max(previous[1], end), name)
        else:
            resolved.append((start, end, name))
    return resolved


def _locked_source(text, terms):
    # 锁词替换只用于模型输入，存盘 OCR、点读原文和生成译文均不改写。
    pieces, cursor = [], 0
    for start, end, name in locked_mentions(text, terms):
        pieces.extend((text[cursor:start], name))
        cursor = end
    return "".join([*pieces, text[cursor:]])


def validate_input(block, terms, *, target=True, history=False):
    """加载模型前校验原文、术语及可选历史译文，不修改存盘内容。

    只有 target=True 才检查锁词要求，邻泡中的名字仅作上下文。
    """
    texts = [block.source_text]
    for term in terms:
        texts.extend((term["source"], term["target"], *term.get("aliases", [])))
    if history:
        texts.append(block.translated_text)
    if target:
        source = translation_source(block)
        texts.append(source)
        _locked_targets([name for _, _, name in locked_mentions(block.source_text, terms)])
        if source != block.source_text:
            locked_mentions(source, terms)
    if any(token in text for text in texts for token in ("<|", "|>")):
        raise InvalidTranslationInput("OCR/terminology contains reserved model control tokens; correct the text first")


def _nearby_blocks(blocks: list[TextBlock], positions: list[int], first: int, config: Config) -> list[int]:
    budget, nearby = config.context_chars, []
    # 邻泡整段选取，截断可能丢掉句尾否定。
    for i in sorted((i for i in range(len(blocks)) if i not in positions),
                    key=lambda i: (min(abs(i-j) for j in positions), i)):
        text = blocks[i].source_text.strip()
        history_length = (len(blocks[i].translated_text)
                          if config.use_translation_history and i < first and blocks[i].status == "translated" else 0)
        cost = len(text) + history_length
        if text and cost <= budget:
            nearby.append(i)
            budget -= cost
    nearby.sort()
    return nearby


def _prompt_metadata(blocks: list[TextBlock], selected: list[TextBlock], positions: list[int],
                     nearby: list[int], terms: list[dict[str, Any]], config: Config,
                     speakers: dict[str, dict[str, Any]] | None) -> list[str]:
    metadata = ["[context] Japanese manga dialogue in reading order. Dialogue may continue across bubbles. "
                "Keep every bracketed bubble ID unchanged and translate each requested line exactly once. "
                "Translate only the final Japanese section; other dialogue is context."]
    if config.use_speaker_context and config.context_chars > 0 and speakers:
        from core.memory import speaker_context, speaker_context_json, SPEAKER_CONTEXT_RULE

        identities = speaker_context([blocks[i] for i in sorted(set(positions + nearby))], speakers,
                                     id_transform=prompt_id)
        if identities:
            metadata.append("[speaker information] " + SPEAKER_CONTEXT_RULE)
            metadata.append("[speaker data] " + speaker_context_json(identities))
    if any(b.metadata.get("continuation") for b in selected):
        metadata.append("[translation requirement] A linked target contains consecutive source fragments. "
                        "Identify their logical connection; short kana may be part of a word, not a person's name. "
                        "Translate its complete meaning once, preserving hesitation where appropriate; "
                        "do not translate each fragment independently or repeat the sentence.")
    for term in terms:
        line = f"[element] Name: {term['target']} ({term['source']})"
        if term.get("aliases"):
            line += " | Aliases: " + ", ".join(f"{term['target']} ({alias})" for alias in term["aliases"])
        if term.get("locked"):
            line += " | Translation: use this exact English name"
        metadata.append(line)
    if any(term["locked"] for term in terms):
        metadata.append("[translation requirement] Locked names in the final Japanese section already use their exact English spelling. Keep those names unchanged.")
    return metadata


def _prompt_history(blocks: list[TextBlock], nearby: list[int], first: int, last: int,
                    config: Config, metadata: list[str]) -> tuple[list[str], list[str]]:
    history, history_text = [], []
    for i in nearby:
        block = blocks[i]
        if config.use_translation_history and i < first and block.status == "translated" and block.translated_text:
            history_text.extend((block.source_text, block.translated_text))
            history.append(_header("Japanese") + f"[{prompt_id(block.block_id)}]: {block.source_text}<|eot_id|>"
                           + _header("English") + f"[{prompt_id(block.block_id)}]: {block.translated_text}<|eot_id|>")
        else:
            where = "before" if i < first else "after" if i > last else "between"
            metadata.append(f"[context {where}] [{prompt_id(block.block_id)}]: {block.source_text}")
    return history, history_text


def build_prompt(blocks: list[TextBlock], selected: list[TextBlock], memory: GlossaryMemory,
                 config: Config, *, previous_error: str = "",
                 speakers: dict[str, dict[str, Any]] | None = None) -> tuple[str, list[dict[str, Any]]]:
    """使用 VNTL 的 Metadata / Japanese / English 格式。

    先前译文作对齐示例，后续原文放入限长上下文；只翻译最后一轮 Japanese。
    """
    selected_ids = {b.block_id for b in selected}
    positions = [i for i, b in enumerate(blocks) if b.block_id in selected_ids]
    first, last = min(positions), max(positions)
    nearby = _nearby_blocks(blocks, positions, first, config)
    terms = memory.match("\n".join(b.source_text for b in selected)
                         + "\n" + "\n".join(blocks[i].source_text for i in nearby))
    metadata = _prompt_metadata(blocks, selected, positions, nearby, terms, config, speakers)
    history, history_text = _prompt_history(blocks, nearby, first, last, config, metadata)
    if previous_error:
        metadata.append("[translation requirement] " + previous_error[:300])
    source = "\n".join(f"[{prompt_id(b.block_id)}]: {_locked_source(translation_source(b), terms)}" for b in selected)
    if any(token in "\n".join(metadata + history_text) + source for token in ("<|", "|>")):
        raise InvalidTranslationInput("OCR/terminology contains reserved model control tokens; correct the text first")
    prompt = ("<|begin_of_text|>" + _header("Metadata") + "\n".join(metadata) + "<|eot_id|>"
              + "".join(history) + _header("Japanese") + source + "<|eot_id|>" + _header("English"))
    return prompt, terms


def parse_response(raw, expected_ids):
    """按显式 ID 对应译文，不用行序补猜缺失 ID。"""
    expected = set(expected_ids)
    # VNTL 可能把标签转成大写；忽略大小写匹配，但拒绝冲突。
    canonical = {}
    for key in expected:
        for label in {key, prompt_id(key)}:
            folded = label.casefold()
            if folded in canonical and canonical[folded] != key:
                return {}, {key: "Ambiguous case-insensitive bubble IDs" for key in expected}
            canonical[folded] = key
    if not isinstance(raw, str) or not raw.strip():
        return {}, {key: "Empty translation" for key in expected}
    raw = raw.strip()
    matches = list(re.finditer(r"(?m)^\[([^\]\r\n]+)\]:[ \t]*", raw))
    if not matches:
        # 只有单目标请求能明确接收无标签译文。
        if len(expected) == 1 and not raw.startswith(("[", "{", "```", "<|")):
            return {next(iter(expected)): raw}, {}
        return {}, {key: "Missing labelled bubble IDs" for key in expected}
    if raw[:matches[0].start()].strip() or any(m[1].casefold() not in canonical for m in matches):
        return {}, {key: "Unexpected bubble ID or text before first ID" for key in expected}
    result, errors, seen = {}, {}, set()
    for index, match in enumerate(matches):
        key = canonical[match[1].casefold()]
        if key in seen:
            result.pop(key, None)
            errors[key] = "Duplicate bubble ID"
            continue
        seen.add(key)
        text = raw[match.end():matches[index+1].start() if index+1 < len(matches) else len(raw)].strip()
        if not text:
            errors[key] = "Empty translation"
        else:
            result[key] = text
    for key in expected - seen:
        errors[key] = "Missing bubble ID"
    return result, errors


def _locked_targets(targets):
    from core.memory import contains_term

    if not isinstance(targets, (list, tuple)):
        raise InvalidTranslationInput("Locked targets must be a list of English terms")
    if any(not isinstance(term, str) or not term.strip()
           or any(ord(char) < 32 for char in term) for term in targets):
        raise InvalidTranslationInput("Locked targets must be nonempty, single-line English terms")
    terms = list(dict.fromkeys(term.strip() for term in targets))
    if len({term.casefold() for term in terms}) != len(terms):
        raise InvalidTranslationInput("Locked targets have conflicting capitalization")
    # 「Ann Marie」已满足词界内的「Ann」，不要求重复生成。
    terms = [term for term in terms
             if not any(term != other and contains_term(other, term) for other in terms)]
    if len(terms) > 4 or sum(map(len, terms)) > 256:
        raise InvalidTranslationInput("At most 4 distinct locked terms / 256 characters per bubble are supported; "
                         "split the bubble or simplify its glossary entries")
    for left in terms:
        for right in terms:
            if left == right:
                continue
            for length in range(1, min(len(left), len(right))):
                if left[-length:] == right[:length]:
                    combined = left + right[length:]
                    if contains_term(combined, left) and contains_term(combined, right):
                        raise InvalidTranslationInput(f"Overlapping locked targets {left!r} and {right!r}; "
                                         "resolve the overlapping glossary names")
    return terms


def _locked_line_grammar(prefix, terms):
    """用“已出现词集合 + 右词界要求”表示状态，最多 2 * 2**4 个。

    无需枚举 n! 种词序；词界防止用 Hannah 满足 Ann，普通词不受限。
    每个锁词气泡独占一行，所需译名必须在本行出现。
    """
    rules = {}
    complete = (1 << len(terms)) - 1

    def word(char):
        return char.isascii() and char.isalnum()

    def state(seen, boundary):
        name = f"{prefix}-{seen}-{int(boundary)}"
        if name in rules:
            return name
        if seen == complete:
            rules[name] = "locked-gap10" if boundary else "locked-gap00"
        else:
            choices = []
            for index, term in enumerate(terms):
                if not seen & (1 << index):
                    following = state(seen | (1 << index), word(term[-1]))
                    gap = f"locked-gap{int(boundary)}{int(word(term[0]))}"
                    choices.append(f"{gap} {json.dumps(term, ensure_ascii=False)} {following}")
            rules[name] = " | ".join(choices)
        return name

    first = state(0, False)
    return first, [f"{name} ::= {body}" for name, body in rules.items()]


def validate_request(ids, required_terms=None):
    """本地校验分组标签和锁词要求，不发 HTTP 请求。"""
    ids = list(ids)
    labels = [prompt_id(key) for key in ids]
    if len(set(label.casefold() for label in labels)) != len(labels):
        # 分组冲突可拆成单泡处理，不标记为原文错误。
        raise ValueError("Ambiguous prompt bubble IDs")
    required_terms = required_terms or {}
    if set(required_terms) - set(ids):
        raise InvalidTranslationInput("Locked terms refer to an unrequested bubble ID")
    return labels, {key: _locked_targets(required_terms.get(key, [])) for key in ids}


def label_grammar(ids, required_terms=None):
    """在解码时用 GBNF 固定气泡标签和锁定译名，不事后替换译文。

    required_terms 只包含当前气泡原文命中的锁词，不含仅在邻泡出现的名字。
    """
    ids = list(ids)
    labels, required_terms = validate_request(ids, required_terms)
    parts, rules = [], []
    for index, (key, label) in enumerate(zip(ids, labels)):
        terms = required_terms[key]
        text_rule = "text"
        if terms:
            text_rule, lines = _locked_line_grammar(f"locked-{index}", terms)
            rules.extend(lines)
        parts.append(json.dumps(("\n" if index else "") + f"[{label}]: ") + " " + text_rule)
    grammar = "root ::= " + " ".join(parts) + '\ntext ::= [^\\n]+ ("\\n" [^\\[\\n] [^\\n]*)*\n'
    if rules:
        grammar += "\n".join(rules) + '\n'
        grammar += ('locked-gap00 ::= [^\\n]*\n'
                    'locked-gap01 ::= "" | [^\\n]* [^A-Za-z0-9_\\n]\n'
                    'locked-gap10 ::= "" | [^A-Za-z0-9_\\n] [^\\n]*\n'
                    'locked-gap11 ::= [^A-Za-z0-9_\\n] | [^A-Za-z0-9_\\n] [^\\n]* [^A-Za-z0-9_\\n]\n')
    return grammar


class VntlServer:
    def __init__(self, config):
        self.process = None
        self.log = None
        self.context_size = config.llama_context_size
        model = Path(config.gguf_path).expanduser().resolve()
        executable = Path(config.llama_server_path).expanduser().resolve()
        if not model.is_file() or not executable.is_file():
            raise FileNotFoundError("VNTL GGUF/CUDA llama-server missing; see README installation. "
                                    f"Model: {model}; server: {executable}")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())
        log_dir = ROOT / "data" / "vntl"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"llama-{time.time_ns()}.log"
        device = "CUDA" + (config.device.split(":", 1)[1] if ":" in config.device else "0")
        command = [str(executable), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
                   "--device", device, "--gpu-layers", "99", "--split-mode", "none", "--fit", "off",
                   "--ctx-size", str(self.context_size), "--parallel", "1", "--batch-size", "256",
                   "--ubatch-size", "128", "--flash-attn", "on", "--cache-ram", "0",
                   "--threads", "4", "--threads-batch", "4", "--verbosity", "4",
                   "--no-context-shift", "--no-webui"]
        self.info = {"backend": "vntl", "model_file": str(model), "requested_quantization": "Q4_K_M",
                     "model_bytes": model.stat().st_size, "context_size": self.context_size,
                     "command": command, "log_path": str(self.log_path), "generation_calls": 0,
                     "generation_seconds": 0.0, "temperature": 0, "repeat_penalty": 1.0}
        started = time.perf_counter()
        try:
            self.log = self.log_path.open("wb")
            env = os.environ.copy()
            # 只定位当前环境的 CUDA DLL，不导入 torch。
            torch_spec = find_spec("torch")
            if torch_spec is not None:
                for package_dir in torch_spec.submodule_search_locations or ():
                    torch_lib = Path(package_dir) / "lib"
                    if torch_lib.is_dir():
                        env["PATH"] = str(torch_lib) + os.pathsep + env.get("PATH", "")
                        break
            self.process = subprocess.Popen(command, cwd=executable.parent, stdout=self.log, stderr=subprocess.STDOUT,
                env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("CUDA llama-server exited during model loading")
                try:
                    if self._request("/health", timeout=1).get("status") == "ok":
                        break
                except (URLError, OSError):
                    time.sleep(.2)
            else:
                raise TimeoutError("CUDA llama-server did not become ready within 120 seconds")
            log = self.log_path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
            if not match or int(match[1]) != int(match[2]) or int(match[1]) == 0:
                raise RuntimeError("Full GPU layer offload was not confirmed; CPU fallback is disabled")
            buffers = [line for line in log.splitlines() if "buffer size" in line or "offloaded" in line]
            if not any("CUDA" in line and "model buffer" in line for line in buffers):
                raise RuntimeError("No CUDA model buffer was reported; refusing CPU execution")
            file_type = re.search(r"file type\s*=\s*([^\r\n]+)", log)
            self.info.update(load_seconds=time.perf_counter()-started, offloaded_layers=int(match[1]),
                             quantization=file_type[1].strip() if file_type else "not reported by server",
                             total_layers=int(match[2]), buffers=buffers, pid=self.process.pid)
        except Exception as exc:
            self.close()
            detail = self.log_path.read_text(encoding="utf-8", errors="replace")[-4500:]
            raise RuntimeError(f"VNTL CUDA loading failed: {exc}. Inactive project models were offloaded; "
                               f"free GPU memory is still required. Log: {self.log_path}\n{detail}") from exc

    def _request(self, path, body=None, *, timeout=120):
        request = Request(self.base_url + path,
                          data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
                          headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=timeout) as response:
            data = response.read(8*1024*1024+1)
        if len(data) > 8*1024*1024:
            raise ValueError("Oversized local translation response")
        return json.loads(data.decode("utf-8"))

    def generate(self, prompt, max_new_tokens, *, ids=None, required_terms=None):
        from .shared import ContextTooLong

        grammar = label_grammar(ids, required_terms) if ids else None
        if not ids and required_terms:
            raise InvalidTranslationInput("Locked terms require explicit bubble IDs")
        tokens = self._request("/tokenize", {"content": prompt, "add_special": False, "parse_special": True})["tokens"]
        if len(tokens) + max_new_tokens > self.context_size:
            raise ContextTooLong(f"VNTL prompt {len(tokens)} + output {max_new_tokens} > context {self.context_size}; "
                                 "reduce group size/context or correct excessively long OCR")
        started = time.perf_counter()
        body = {"prompt": prompt, "n_predict": max_new_tokens,
            "temperature": 0, "repeat_penalty": 1.0, "repeat_last_n": 0, "presence_penalty": 0,
            "frequency_penalty": 0, "top_k": 0, "top_p": 1.0, "min_p": 0,
            "cache_prompt": True, "seed": 0, "stop": ["<|eot_id|>", "<|end_of_text|>"]}
        if grammar is not None:
            body["grammar"] = grammar
        result = self._request("/completion", body)
        self.info["generation_calls"] += 1
        self.info["generation_seconds"] += time.perf_counter()-started
        self.info["last_timings"] = result.get("timings", {})
        if result.get("truncated") or result.get("stopped_limit") or result.get("stop_type") == "limit":
            raise ValueError("VNTL reached its token limit; output is incomplete")
        text = result.get("content")
        if not isinstance(text, str):
            raise ValueError("Invalid llama.cpp completion response")
        return text

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()
