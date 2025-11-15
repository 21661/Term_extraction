from utils.workflow_adapter import unwrap_state, rewrap_state
from utils.candidate_tool import (
    normalize_candidate, is_noise_candidate, score_term,
    _COMMON_GENERIC_WORDS
)
import spacy

nlp = spacy.load("en_core_web_trf")


def extract_candidates(state):
    """
    高质量术语提取（不扩展、不切片、不创造新术语版本）：
    - 仅从原文中提取：noun_chunk、原文 token、NER 实体（实体文本来自原文，因此合法）
    - 更严格噪声过滤
    - 去除常见通用词（必须也原文出现）
    - normalize 去重，但保留原始大小写与原文顺序
    """

    inner, parent, key = unwrap_state(state)
    text = inner.get("text", "") or ""

    try:
        doc = nlp(text)
    except Exception:
        inner["candidates"] = []
        inner["scores"] = []
        inner["cleaned_text"] = text.strip()
        return rewrap_state(state, parent, key, inner)

    # -----------------------------
    # 1. 提取原文术语（严格遵守原文，只取 text）
    # -----------------------------

    def collect_noun_chunks(doc):
        """原始 noun chunk（不扩展，不加工）"""
        terms = []
        for chunk in doc.noun_chunks:
            t = chunk.text.strip()
            if t:
                terms.append(t)
        return terms

    def collect_nouns(doc):
        """单 token 名词（不扩展）"""
        terms = []
        for tok in doc:
            if tok.pos_ in ("NOUN", "PROPN"):
                t = tok.text.strip()
                if t:
                    terms.append(t)
        return terms

    def collect_entities(doc):
        """实体直接从原文提取（不加工，不扩展）"""
        ents = []
        for ent in doc.ents:
            # 只保留实体原文（不扩展、不构造）
            ents.append(ent.text.strip())
        return ents

    # 原文术语候选
    terms = []
    terms += collect_noun_chunks(doc)
    terms += collect_nouns(doc)
    terms += collect_entities(doc)

    # -----------------------------
    # 2. 过滤逻辑（只过滤，不创造）
    # -----------------------------

    def is_generic_word(t):
        return t.lower() in _COMMON_GENERIC_WORDS

    def filter_noise(terms):
        return [t for t in terms if not is_noise_candidate(t)]

    def filter_generic_words(terms):
        """去掉常见泛词（仍然是原文出现的词，不创造）"""
        return [t for t in terms if not is_generic_word(t)]

    def filter_min_length(terms):
        """去掉长度为1 或纯符号"""
        res = []
        for t in terms:
            if len(t) <= 1:
                continue
            if all(not c.isalnum() for c in t):
                continue
            res.append(t)
        return res

    def dedupe_preserve_original(terms):
        """根据 normalize 去重，但保留最先出现的原文大小写"""
        seen = set()
        out = []
        for t in terms:
            k = normalize_candidate(t)
            if not k:
                continue
            if k in seen:
                continue
            seen.add(k)
            out.append(t)
        return out

    # 过滤流水线
    terms = filter_noise(terms)
    terms = filter_generic_words(terms)
    terms = filter_min_length(terms)
    terms = dedupe_preserve_original(terms)

    # -----------------------------
    # 3. 打分（不改变原文词语）
    # -----------------------------
    term_scores = [(t, score_term(t, doc)) for t in terms]
    term_scores.sort(key=lambda x: x[1], reverse=True)

    # -----------------------------
    # 4. 写回
    # -----------------------------
    inner["candidates"] = terms
    inner["scores"] = term_scores
    inner["cleaned_text"] = text.strip()

    return rewrap_state(state, parent, key, inner)
