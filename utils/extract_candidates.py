from utils.workflow_adapter import unwrap_state, rewrap_state
from utils.candidate_tool import (
    normalize_candidate, is_noise_candidate, score_term,dedupe_keep_longest,
    _COMMON_GENERIC_WORDS
)
from utils.TermState import TermState
from utils.workflow_adapter import _rewrap,_unwrap
import spacy

nlp = spacy.load("en_core_web_trf")


def extract_candidates(state: TermState) -> TermState:
    inner, parent, key = _unwrap(state)
    text = inner.get("text", "") or ""

    try:
        doc = nlp(text)
    except Exception:
        inner["candidates"] = []
        inner["terms"] = []
        inner["scores"] = []
        inner["cleaned_text"] = text.strip()
        return rewrap_state(state, parent, key, inner)

    # -----------------------------
    # 1. 收集原文候选词
    # -----------------------------
    terms = []
    terms += [chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()]
    terms += [tok.text.strip() for tok in doc if tok.pos_ in ("NOUN", "PROPN") and tok.text.strip()]
    terms += [ent.text.strip() for ent in doc.ents if ent.text.strip()]

    # -----------------------------
    # 2. 应用你的 normalize + noise filter
    # -----------------------------
    filtered_terms = []
    for t in terms:
        t_norm = normalize_candidate(t)
        if not t_norm or is_noise_candidate(t_norm):
            continue
        filtered_terms.append(t)

    # -----------------------------
    # 3. 去重保留最长版本
    # -----------------------------
    deduped_terms = filtered_terms

    # -----------------------------
    # 4. 打分
    # -----------------------------
    term_scores = [(t, score_term(t, doc)) for t in deduped_terms]
    term_scores.sort(key=lambda x: x[1], reverse=True)

    # -----------------------------
    # 5. 写回 state
    # -----------------------------
    inner["candidates"] = deduped_terms
    inner["terms"] = deduped_terms  # 供 select_top_terms 使用
    inner["scores"] = term_scores
    inner["cleaned_text"] = text.strip()

    return rewrap_state(state, parent, key, inner)