import re
from typing import List, Dict

# ------------------ 参数 ------------------
_MAX_TERM_WORDS = 4
_NOISE_MIN_CHAR = 2
_LLM_RETRIES = 2
_RETRY_BACKOFF = 1.0
_COMMON_GENERIC_WORDS = {
    "that", "this", "an", "a", "the", "function", "input", "output",
    "example", "learning", "machine", "maps", "pairs", "task",
    "system", "model", "data", "information", "value", "values",
    "method", "methods", "approach", "approaches",
    "results", "analysis", "performance", "study", "studies"
}
# 正则
_MULTI_SPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9\-_\s]+")
_ALLOWED_CHARS_RE = re.compile(r"^[A-Za-z \-]+$")
_DISALLOWED_PUNCT_RE = re.compile(r"[,.:;.!?\"'(){}\[\]<>]")
_REPEAT_LETTERS_RE = re.compile(r"([A-Za-z])\1{3,}")
_BULLET_PREFIX_RE = re.compile(r"^\s*[#*\u2022-]?\s*\d+[.)]?\s+")
_MD_IMAGE_RE = re.compile(r"!\[[^]]*\]\([^)]*\)")
_MATH_LATEX_RE = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\\\(|\\\)|\\\[|\\\]|\\begin\{.*?\}|\\end\{.*?\})")
_PLACEHOLDER_RE = re.compile(r"\{\{\{.*?}}}|\{\{.*?}}")
_CITATION_BRACKET_RE = re.compile(r"\[[0-9]+\]")
_CITATION_AUTHOR_YEAR_RE = re.compile(r"\([A-Za-z][^)]*?\d{4}[^)]*?\)")

_EXTRA_NOISE = {
    "that","this","these","those","it","they","he","she","a","an","the",
    "use","uses","used","based","which","table","json","figure","fig","e.g",
    "i.e","etc","et al","data","information","model","models","method","methods",
    "approach","approaches","result","results","analysis","performance",
    "study","studies","paper","article","section","chapter","book","author","authors",
    "introduction","conclusion"
}

# ------------------ 核心函数 ------------------

def normalize_candidate(term: str) -> str:
    """更强的候选词标准化"""
    if not isinstance(term, str):
        term = str(term)

    t = term.strip()

    # --- Markdown 标题 (# H1, ## H2, ### ...)
    t = re.sub(r"^\s*#{1,6}\s+", "", t)

    # --- 去掉常见 latex 残片（你原来的 regex 只能抓完整表达式）
    latex_fragments = [
        r"\\mathrm", r"\\mathcal", r"\\mathbf", r"\\text", r"\\operatorname",
        r"\\left", r"\\right", r"\\cdot", r"\\times", r"\\frac", r"\\sum",
        r"\\begin", r"\\end", r"\\alpha", r"\\beta", r"\\gamma"
    ]
    for frag in latex_fragments:
        t = re.sub(frag, "", t)

    # --- 删除 bullet / 编号前缀
    t = re.sub(r"^\s*[\-*•·●]?\s*\d*[.)]?\s*", "", t)

    # --- 你原本已有的符号清洗
    t = t.replace("•", "").replace("·", "").replace("●", "")
    t = t.replace("\u3000", " ")

    # --- 合并多空格
    t = _MULTI_SPACE.sub(" ", t)

    # --- 去除所有非字母数字（保留空格和连字符）
    t = _NON_ALNUM.sub(" ", t)

    t = _MULTI_SPACE.sub(" ", t)

    return t.strip()

def is_noise_candidate(candidate: str) -> bool:
    """更严格噪声过滤"""
    if not candidate or len(candidate.strip()) <= _NOISE_MIN_CHAR:
        return True

    s = candidate.strip()

    # ---- 彻底阻断 Latex 残片
    if re.search(r"[{}\\]", s):
        return True

    # ---- 阻断 Markdown 头部碎片
    if re.match(r"^#+\s*", candidate):
        return True

    # ---- 阻断包含奇怪的大写分词，如 "N A P" "A P"（你最常出现的问题）
    if re.match(r"^([A-Z]\s+){1,5}[A-Z]$", s):
        return True

    # ---- 阻断纯大写碎词（非缩写）
    if len(s) <= 5 and s.isupper():
        return True

    # ---- 阻断 Technical Report / Section / Chapter 等非术语
    NOISE_PHRASES = {
        "technical report", "related work", "introduction",
        "conclusion", "appendix", "future work", "summary"
    }
    if s.lower() in NOISE_PHRASES:
        return True

    # ---- 阻断 URL
    if re.match(r"https?://", s):
        return True

    # ---- 阻断字母之间混入空格但不是术语（最典型例子：N A P）
    if re.match(r"^[A-Za-z](\s[A-Za-z]){2,}$", s):
        return True

    # ---- 你已有的一堆规则（保留）
    if (_MD_IMAGE_RE.search(s) or _MATH_LATEX_RE.search(s)
        or _CITATION_BRACKET_RE.search(s) or _CITATION_AUTHOR_YEAR_RE.search(s)
        or _PLACEHOLDER_RE.search(s)):
        return True

    if _BULLET_PREFIX_RE.match(s):
        return True

    if not _ALLOWED_CHARS_RE.match(s):
        return True
    if _DISALLOWED_PUNCT_RE.search(s):
        return True

    words = [w for w in re.split(r"\s+", s) if w]
    if len(words) > _MAX_TERM_WORDS:
        return True

    if s.lower() in _EXTRA_NOISE:
        return True

    if _REPEAT_LETTERS_RE.search(s):
        return True

    if not re.search(r"[A-Za-z]", s):
        return True

    return False

def dedupe_keep_longest(candidates: List[str]) -> List[str]:
    """去重保留最长词"""
    seen_map: Dict[str, str] = {}
    for t in candidates:
        if not t.strip():
            continue
        key = t.lower().strip()
        if key not in seen_map or len(t) > len(seen_map[key]):
            seen_map[key] = t
    return list(seen_map.values())

def score_term(term: str, doc) -> float:
    """评分"""
    t = term.lower().strip()
    if not t:
        return 0.0
    tokens = [tok.text.lower() for tok in doc]
    text_len = len(tokens)
    freq = tokens.count(t)
    try:
        first_idx = tokens.index(t)
        pos_score = (text_len - first_idx) / text_len
    except ValueError:
        pos_score = 0.1
    length_score = min(len(t)/10, 1.0)
    short_penalty = -0.5 if len(t) <= 2 else 0
    freq_score = min(freq, 5) * 0.3
    alpha_bonus = 0.2 if t.isalpha() else -0.2
    score = freq_score + pos_score + length_score + short_penalty + alpha_bonus
    return round(score, 4)
