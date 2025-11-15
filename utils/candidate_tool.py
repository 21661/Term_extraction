import re
from typing import List, TypedDict, Dict, Set

# 可调参数
_MAX_TERMS_TO_PROCESS = 10  # 目标提取 5-6 个术语（最大候选数）
_LLM_RETRIES = 2
_RETRY_BACKOFF = 1.0
_NOISE_MIN_CHAR = 2
# 新增：候选词最大词数限制
_MAX_TERM_WORDS = 5

# Performance tuning params
_MAX_NOUN_CHUNKS = 80
_TRANSLATE_WORKERS = 6
_TRANSLATE_TIMEOUT = 6.0  # seconds per translation task

# LLM call tuning
_LLM_TIMEOUT = 12.0  # seconds per LLM request
_LLM_CACHE_ENABLED = True

# counters
LLM_CALLS = 0
DICT_CALLS = 0

# ===================== 辅助函数：标准化/噪声过滤/去重 =====================
LEADING_ARTICLES = re.compile(r'^(?:a|an|the)\s+', flags=re.I)
# 新增：占位符与噪音检测相关正则
_PLACEHOLDER_RE = re.compile(r"\{\{\{.*?}}}|\{\{.*?}}")
_DISALLOWED_PUNCT_RE = re.compile(r"[,.:;.!?\"'(){}\[\]<>]")
# 只允许字母、空格、连字符（-）
_ALLOWED_CHARS_RE = re.compile(r"^[A-Za-z \-]+$")
_REPEAT_LETTERS_RE = re.compile(r"([A-Za-z])\1{3,}")  # 4 个及以上连续重复字母
_BULLET_PREFIX_RE = re.compile(r"^\s*[#*\u2022-]?\s*\d+[.)]?\s+")
# Markdown 图片 ![alt](url)
_MD_IMAGE_RE = re.compile(r"!\[[^]]*\]\([^)]*\)")
# LaTeX/数学公式（$...$, $$...$$, \\(...\\), \\[...\\], \begin{...}）
_MATH_LATEX_RE = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\\\(|\\\)|\\\[|\\\]|\\begin\{.*?\}|\\end\{.*?\})")
# 引用样式：[12] 或 (Smith 2020)
_CITATION_BRACKET_RE = re.compile(r"\[[0-9]+\]")
_CITATION_AUTHOR_YEAR_RE = re.compile(r"\([A-Za-z][^)]*?\d{4}[^)]*?\)")


_NON_ALNUM = re.compile(r"[^A-Za-z0-9\-_\s]+")
_MULTI_SPACE = re.compile(r"\s+")

# 常见非术语噪声词（小写）
_EXTRA_NOISE = {
    "that", "this", "these", "those", "it", "they", "he", "she",
    "a", "an", "the", "use", "uses", "used", "based",
}
# 明显不应入库的普通词（可扩展）
_COMMON_GENERIC_WORDS = {
    "that", "this", "an", "a", "the", "function", "input", "output",
    "example", "learning", "machine", "maps", "pairs", "task",
}


def normalize_candidate(term: str) -> str:
    """保持原功能的完整等价版：标准化术语的清洗流程。"""

    if not isinstance(term, str):
        term = str(term)

    t = term.strip()

    # 去掉常见噪声字符
    t = t.replace("•", "").replace("·", "").replace("●", "")
    t = t.replace("\u3000", " ")  # 全角空格

    # 压缩多余空白
    t = _MULTI_SPACE.sub(" ", t)

    # 去掉术语中非字母数字的内容（你原逻辑如此）
    t = _NON_ALNUM.sub("", t)

    # 统一大小写
    t = t.strip()

    return t

def is_noise_candidate(candidate: str) -> bool:
    # 强化噪音筛选：长度、字符集、占位符、标点、乱码等
    if not candidate:
        return True

    s = candidate.strip()

    # 图片/公式/引用直接排除
    if _MD_IMAGE_RE.search(s):
        return True
    if _MATH_LATEX_RE.search(s):
        return True
    if _CITATION_BRACKET_RE.search(s) or _CITATION_AUTHOR_YEAR_RE.search(s):
        return True

    # 占位符直接排除，例如 {{{Table}}}、{{...}}
    if _PLACEHOLDER_RE.search(s) or ("{{{" in s or "}}}" in s) or ("{{" in s or "}}" in s):
        return True

    # 列表项/编号前缀，如 "# 4 performance", "1. xxx", "2) xxx" 等
    if _BULLET_PREFIX_RE.match(s):
        return True

    # 词数限制（不超过 4-5 个词）
    words = [w for w in re.split(r"\s+", s) if w]
    if len(words) > _MAX_TERM_WORDS:
        return True

    # 只允许字母、空格、连字符（禁止数字、下划线、斜杠等）
    if not _ALLOWED_CHARS_RE.match(s):
        return True

    # 中间不允许出现除 - 以外的��点符号
    if _DISALLOWED_PUNCT_RE.search(s):
        return True

    # 全局最低字符数量（去掉空格）
    if len(s.replace(" ", "")) <= _NOISE_MIN_CHAR:
        return True

    # 必须包含英文字母
    if not re.search(r"[A-Za-z]", s):
        return True

    # 可疑的连续重复字母（如 aaaa, qqqq）
    if _REPEAT_LETTERS_RE.search(s):
        return True

    # 可疑的无元音的大写长串（例如无意义缩写/乱码）
    tokens = re.split(r"-", s)
    for tok in tokens:
        t = tok.replace(" ", "")
        if len(t) >= 6 and t.isalpha() and t.upper() == t and not re.search(r"[AEIOU]", t):
            return True

    # 常见噪音词
    if s.lower() in _EXTRA_NOISE:
        return True

    return False

def dedupe_keep_longest(candidates: List[str]) -> List[str]:
    """去重并保留最长的候选词，使用小写形式作为去重 key，但保留原始大小写。"""
    seen_map: Dict[str, str] = {}
    for t in candidates:
        if not t or not t.strip():
            continue
        key = t.strip().lower()
        if key not in seen_map or len(t) > len(seen_map[key]):
            seen_map[key] = t

    return list(seen_map.values())

def score_term(term: str, doc) -> float:
    """
    等价替换：依据你现有规则的“统一化评分系统”。
    评分项保持一致，但结构更清晰、无重复计算。
    """

    t = term.lower().strip()
    if not t:
        return 0.0

    # -------- 预处理缓存 --------
    tokens = [tok.text.lower() for tok in doc]
    text_len = len(tokens)

    # 基础词频
    freq = tokens.count(t)

    # 出现位置（越靠前加成越大）
    try:
        first_idx = tokens.index(t)
        pos_score = (text_len - first_idx) / text_len  # 你原逻辑类似
    except ValueError:
        pos_score = 0.1  # 若未找到，与原代码一样给弱保底

    # 长度加成
    length_score = min(len(t) / 10, 1.0)

    # 惩罚无意义短词
    short_penalty = -0.5 if len(t) <= 2 else 0

    # 词频加成（上限，用于避免爆炸）
    freq_score = min(freq, 5) * 0.3

    # 是否为全字母（你原代码对纯符号词会降权）
    is_alpha = t.isalpha()
    alpha_bonus = 0.2 if is_alpha else -0.2

    # -------- 最终分数组合（顺序与你原逻辑保持一致） --------
    score = (
        freq_score +
        pos_score +
        length_score +
        short_penalty +
        alpha_bonus
    )

    return round(score, 4)
