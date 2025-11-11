from typing import List, TypedDict, Dict, Set
from langgraph.graph import StateGraph, END
import re
import spacy
import json
import time
import logging
import itertools
from functools import wraps
from utils.zhipu_client import client
from utils.db_interface import query_term_translation, save_translation
from utils.web_fetcher import fetch_core_translations
from utils.Get_term import translate_term as translate_term_external
import typing


# 简单的包装/解包辅助（兼容工作流可能传入的三元组包装或直接 dict）
def _unwrap(raw: typing.Any):
    """解包状态：若 state 是三元组 (inner, parent, key) 则返回它，否则若为 dict 则返回 (dict, None, None)。"""
    try:
        if isinstance(raw, tuple) and len(raw) == 3:
            return raw
    except Exception:
        pass
    if isinstance(raw, dict):
        return raw, None, None
    return raw, None, None


def _rewrap(original_raw: typing.Any, parent, key, new_inner: typing.MutableMapping):
    """将 new_inner 包回：若 original_raw 为三元组则返回 (new_inner, parent, key)，否则返回 new_inner 本身。"""
    try:
        if isinstance(original_raw, tuple) and len(original_raw) == 3:
            return (new_inner, parent, key)
    except Exception:
        pass
    return new_inner


# ===================== 1️⃣ 定义状态类型 =====================
class TermState(TypedDict):
    text: str
    candidates: List[str]
    terms: List[str]
    topic: str
    translations: Dict[str, List[str]]  # term -> 候选翻译
    final_translations: Dict[str, str]


# ===================== 2️⃣ 初始化 =====================
nlp = spacy.load("en_core_web_sm")

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 可调参数
_MAX_TERMS_TO_PROCESS = 6  # 目标提取 5-6 个术语（最大候选数）
_LLM_RETRIES = 2
_RETRY_BACKOFF = 1.0
_NOISE_MIN_CHAR = 2

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


# ----------------- 简化版 helpers（已移除缓存与并发，便于验证） -----------------
def _cached_nlp(text: str):
    """直接调用 spaCy（已移除缓存）。"""
    return nlp(text)


def _cached_translate_term_external(term: str):
    """直接调用外部翻译器（无缓存）。返回列表形式以兼容现有代码。"""
    try:
        res = translate_term_external(term) or []
        return list(res)
    except Exception:
        return []


def _cached_fetch_core_translations(term: str):
    """直接调用 fetch_core_translations（无缓存）。"""
    try:
        res = fetch_core_translations(term) or []
        return list(res)
    except Exception:
        return []


def _cached_llm_completion(prompt: str, system: str = "你是术语分类助手") -> str:
    """直接调用 LLM，同步阻塞，不做缓存或线程超时控制（便于验证正确性）。

    注意：如果你需要超时保护，可以在后续恢复线程/超时逻辑。
    """
    global LLM_CALLS
    LLM_CALLS += 1
    try:
        completion = client.chat.completions.create(
            model="glm-4.5",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0,
        )
        return _safe_extract_completion_content(completion)
    except Exception as e:
        logger.warning("LLM direct call failed: %s", e)
        return ""


# Decorator to time node functions and log start/finish with simple state sizes
def timed_node(name: str = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            node_name = name or func.__name__
            logger.info("Node %s START", node_name)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                end = time.perf_counter()
                dur = end - start
                # Try to extract counts from the first arg if it's the workflow state
                try:
                    state = args[0] if args else None
                    inner, parent, key = _unwrap(state) if state is not None else (None, None, None)
                    sd = inner if isinstance(inner, dict) else (state if isinstance(state, dict) else {})
                    cands = sd.get("candidates") if isinstance(sd, dict) else None
                    terms = sd.get("terms") if isinstance(sd, dict) else None
                    trans = sd.get("translations") if isinstance(sd, dict) else None
                    logger.info("Node %s END (%.3fs) candidates=%s terms=%s translations=%s", node_name, dur,
                                (len(cands) if isinstance(cands, (list, set)) else '-' ),
                                (len(terms) if isinstance(terms, (list, set)) else '-' ),
                                (len(trans) if isinstance(trans, dict) else '-' ))
                except Exception:
                    logger.info("Node %s END (%.3fs)", node_name, dur)
        return wrapper
    return decorator


# ===================== 辅助函数：标准化/噪声过滤/去重 =====================
LEADING_ARTICLES = re.compile(r'^(?:a|an|the)\s+', flags=re.I)


def normalize_candidate(text: str) -> str:
    """
    更严格的标准化：
    - 去首尾空白、首冠词、去除 possessive (\'s) 与不必要引号/括号
    - 合并空格、小写
    - 返回空字符串表示被完全移除
    """
    if not text:
        return ""
    s = text.strip()
    # 去掉典型的首冠词
    s = re.sub(r'^(?:a|an|the)\s+', '', s, flags=re.I)
    # 去掉 possessive " 's" 与孤立的单引号、双引号以及包裹的括号
    s = re.sub(r"\'s\b", "", s, flags=re.I)
    s = s.strip(" `\"“”()[]{}")
    # 删除两端多余标点（保留中间的连字符/斜杠）
    s = re.sub(r'^[.:;\-]+|[.:;\-]+$', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = s.lower().strip()
    return s


def is_noise_candidate(candidate: str) -> bool:
    if not candidate:
        return True
    if candidate in _EXTRA_NOISE:
        return True
    # 很短的非字母串
    if len(candidate.replace(" ", "")) <= _NOISE_MIN_CHAR:
        return True
    # 必须包含字母
    if not re.search(r"[a-zA-Z]", candidate):
        return True

    # 使用 spaCy 进一步判断：至少包含一个名词/专有名词
    try:
        doc = _cached_nlp(candidate)
    except Exception:
        doc = nlp(candidate)
    has_content = any(getattr(t, 'pos_', None) in ("NOUN", "PROPN") for t in doc)
    if not has_content:
        return True
    # 避免仅为代词/限定词
    all_noise = all((getattr(t, 'is_stop', False) or getattr(t, 'pos_', None) in ("PRON", "DET", "ADP", "PART", "PUNCT")) for t in doc)
    if all_noise:
        return True
    return False


def dedupe_keep_longest(candidates: List[str]) -> List[str]:
    """保留最长短语，若 A 完整包含 B（词边界）则保留 A，丢弃 B。"""
    uniq = sorted(set(candidates), key=lambda x: (-len(x.split()), x))
    kept: List[str] = []
    for cand in uniq:
        cand_words_pattern = r'\b' + re.escape(cand) + r'\b'
        skip = False
        for k in kept:
            if re.search(cand_words_pattern, k):
                skip = True
                break
        if not skip:
            kept.append(cand)
    # 返回稳定排序（按长度和字母）
    return sorted(kept)


def extract_candidates(state: typing.Any) -> TermState:
    original = state
    inner, parent, key = _unwrap(state)
    state = inner if isinstance(inner, dict) else {"text": ""}
    state_dict: dict = typing.cast(dict, state)

    text = state_dict.get("text", "")
    if not text:
        logger.warning("extract_candidates received state without 'text' key")
        state_dict.setdefault("text", "")
        text = state_dict["text"]

    candidates: Set[str] = set()

    # 1️⃣ spaCy 名词短语（限制数量以提高速度）
    doc = _cached_nlp(text)
    for chunk in itertools.islice(doc.noun_chunks, _MAX_NOUN_CHUNKS):
        chunk_text = getattr(chunk, "text", str(chunk)).strip()
        if chunk_text:  # 不限制长度，但限制总数
            candidates.add(chunk_text)

    # 2️⃣ spaCy 专有名词
    proper_nouns = {getattr(token, "text", str(token)).strip()
                    for token in doc if getattr(token, "pos_", None) == "PROPN"}
    candidates.update(proper_nouns)

    # 3️⃣ 连字符/下划线组合、缩写
    regex_terms = set(re.findall(r"\b[A-Za-z]+(?:[-_/][A-Za-z]+)+\b", text))
    candidates.update(regex_terms)

    # 4️⃣ 标准化 + 去噪（只去掉明显无用短词）
    normalized = []
    for c in candidates:
        c_norm = normalize_candidate(c)
        if not c_norm:
            continue
        if c_norm in _EXTRA_NOISE:
            continue
        # 至少包含一个字母
        if not re.search(r"[a-zA-Z]", c_norm):
            continue
        normalized.append(c_norm)

    # 5️⃣ 去重复，优先保留最长短语
    final_candidates = dedupe_keep_longest(normalized)

    # 使用启发式评分对候选进行排序并优先选取 top N（替代 KeyBERT）
    def score_candidate(cand: str) -> float:
        s = 0.0
        cand_l = cand.lower()
        text_l = text.lower()
        # 出现频次（更频繁优先）
        try:
            occ = text_l.count(cand_l)
        except Exception:
            occ = 0
        s += occ * 2.0
        # 词数（更长短语稍微加分）
        words = len(cand.split())
        s += 0.5 * words
        # 词性评分：尽量复用原始 doc 的字符跨度以避免重复解析
        pos_score = 0
        start_idx = text_l.find(cand_l)
        if start_idx >= 0:
            span = doc.char_span(start_idx, start_idx + len(cand_l), alignment_mode='expand')
            if span is not None:
                for t in span:
                    if getattr(t, 'pos_', None) == 'PROPN':
                        pos_score += 2
                    elif getattr(t, 'pos_', None) == 'NOUN':
                        pos_score += 1
                    elif getattr(t, 'pos_', None) == 'ADJ':
                        pos_score += 0.3
        else:
            # 回退到缓存的 nlp 分析
            try:
                docc = _cached_nlp(cand)
                for t in docc:
                    if getattr(t, 'pos_', None) == 'PROPN':
                        pos_score += 2
                    elif getattr(t, 'pos_', None) == 'NOUN':
                        pos_score += 1
                    elif getattr(t, 'pos_', None) == 'ADJ':
                        pos_score += 0.3
            except Exception:
                pass
        s += pos_score

        # 包含连字符/下划线视为技术短语，加分
        if re.search(r"[-_/]", cand):
            s += 1.0
        # 出现位置：越靠前越好
        idx = text_l.find(cand_l)
        if idx >= 0:
            pos_bonus = max(0.0, 1.0 - (idx / max(1, len(text_l))))
            s += pos_bonus
        # 长度归一化小加分
        s += min(len(cand), 50) / 50.0
        return s

    scored = []
    for c in final_candidates:
        if is_noise_candidate(c):
            continue
        scored.append((score_candidate(c), c))
    # 按分数降序，分数相同时按短语长度降序
    scored.sort(key=lambda x: (-x[0], -len(x[1].split())))
    final_candidates = [c for _, c in scored][:_MAX_TERMS_TO_PROCESS]

    # 6️⃣ 限制数量，避免后续大量 LLM 调用
    if len(final_candidates) > _MAX_TERMS_TO_PROCESS:
        logger.info("候选术语过多(%d)，截断到 %d", len(final_candidates), _MAX_TERMS_TO_PROCESS)
        final_candidates = final_candidates[:_MAX_TERMS_TO_PROCESS]

    state_dict["candidates"] = sorted(final_candidates)
    result = _rewrap(original, parent, key, state_dict)
    return result


# ===================== 4️⃣ 术语筛选（调用 LLM 后再清洗） =====================
@timed_node()
def filter_terms(state: typing.Any) -> TermState:
    original = state
    inner, parent, key = _unwrap(state)
    state = inner if isinstance(inner, dict) else {"candidates": []}
    state_dict: dict = typing.cast(dict, state)

    # quick local fallback: if very few candidates, avoid LLM and use rules
    cands = state_dict.get("candidates", [])
    if not cands:
        state_dict["terms"] = []
        state_dict.setdefault("term_types", {})
        return _rewrap(original, parent, key, state_dict)

    if len(cands) <= 2:
        filtered = []
        for c in cands:
            c_norm = normalize_candidate(c)
            if not c_norm:
                continue
            if is_noise_candidate(c_norm):
                continue
            filtered.append(c_norm)
        state_dict["terms"] = sorted(set(filtered))
        # 简单情况全部视为 term
        state_dict["term_types"] = {t: "term" for t in state_dict["terms"]}
        return _rewrap(original, parent, key, state_dict)

    prompt = """
你是一个专业术语识别助手。下面给出一个候选术语列表（来自文档自动抽取）。请严格从该候选列表中挑选并分类，返回严格的 JSON 对象（仅此输出，绝对不要添加任何解释性文字、注释或换行之外的内容）。输出格式必须是一个 JSON 对象，包含两个键： "term" 和 "proper_noun"，它们对应的值都是字符串数组。例如：{"term":["术语A","术语B"],"proper_noun":["专有名1"]}
约束与规则（务必遵守）：
只从候选列表中选择候选项，且必须按候选列表中的原样文本返回（不要改写候选文本的字面形式）。不要创造新术语或拼写变体；如果候选里存在同义或重复项只保留一次。
分类说明：
"term"：学术/技术术语、理论概念常见的学术/工程术语
"proper_noun"：专有名词、算法名、模型名、库/框架名、数据集名称、公司/组织名、产品名、明确的缩写或首字母缩写
长度限制：所选短语长度（以词为单位）不应超过 4–5 个单词。超过该长度的候选请排除，除非它明显为一个已命名的专有名（仍应放在 proper_noun 中）。
严格排除：不要选择明显的普通词或无意义短语，例如 "data", "set", "vector", "each example", "the method", "this paper" 等。若候选仅是停用词/代词/短泛词，应排除。
数量与去重：每个分类内部去重（不重复返回同一字符串）
输出要求：
必须返回合法可 parse 的 JSON，仅此一行或紧凑 JSON（不允许多行文本/人类说明）。
"""
    try:
        cands_repr = json.dumps(state_dict.get('candidates', []), ensure_ascii=False)
    except Exception:
        cands_repr = str(state_dict.get('candidates', []))
    prompt += "\n候选词列表:\n" + cands_repr + "\n"

    terms = []
    proper_nouns = []
    term_types: Dict[str, str] = {}
    try:
        raw_text = _cached_llm_completion(prompt, system="你是术语分类助手")
        if not raw_text:
            raise ValueError("empty LLM response or timed out")
        parsed = json.loads(raw_text)
        terms = list(set(parsed.get("term", [])))
        proper_nouns = list(set(parsed.get("proper_noun", [])))
    except Exception as e:
        logger.warning("LLM 分类失败或返回非 JSON，降级过滤: %s", e)
        candidates = state_dict.get("candidates", [])
        filtered = []
        for c in candidates:
            if is_noise_candidate(c):
                continue
            dd = _cached_nlp(c)
            if any(getattr(t, 'pos_', None) in ("NOUN", "PROPN") for t in dd):
                filtered.append(c)
        terms = dedupe_keep_longest(filtered)
        proper_nouns = []

    final_terms = []
    for t in set(terms + proper_nouns):
        t_norm = normalize_candidate(t)
        if not t_norm or is_noise_candidate(t_norm):
            continue
        final_terms.append(t_norm)
        if t in proper_nouns:
            term_types[t_norm] = "proper_noun"
        else:
            term_types[t_norm] = "term"

    state_dict["terms"] = sorted(set(final_terms))
    state_dict["term_types"] = term_types
    result = _rewrap(original, parent, key, state_dict)
    return result


# ===================== 5️⃣ 判断抓取方式 =====================
def decide_post_db_method(term: str) -> str:
    term_lower = term.lower()
    simple_words = {"data", "set", "map", "input", "output", "value", "use", "function"}
    if term_lower in simple_words or len(term) <= 3:
        return "dict"
    if len(term.split()) > 1 or re.search(r"[-_/]", term):
        return "llm"
    return "dict"


# ===================== 6️⃣ 获取候选翻译（带缓存与重试） =====================
def _safe_extract_completion_content(completion) -> str:
    """Robustly extract text/content from various completion response shapes.
    Returns empty string when content is missing or blank.
    """
    if not completion:
        return ""
    try:
        # common OpenAI-like shape
        choices = None
        if isinstance(completion, dict):
            choices = completion.get('choices')
        else:
            choices = getattr(completion, 'choices', None)
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get('message') or first.get('text')
            else:
                msg = getattr(first, 'message', None) or getattr(first, 'text', None)
            if not msg:
                return ""
            if isinstance(msg, dict):
                return (msg.get('content') or msg.get('text') or "").strip()
            else:
                return str(getattr(msg, 'content', None) or msg).strip()
        # fallback for direct attribute
        if isinstance(completion, dict):
            return (completion.get('text') or "").strip()
        return (getattr(completion, 'text', '') or '').strip()
    except Exception:
        return ""


def _fetch_core_translations_with_retry(term: str, retries=2, backoff=1.0):
    global DICT_CALLS
    attempt = 0
    while attempt <= retries:
        attempt += 1
        try:
            DICT_CALLS += 1
            # use cached wrapper to avoid repeated network calls
            return list(_cached_fetch_core_translations(term))
        except Exception as e:
            logger.warning("fetch_core_translations attempt %d failed for %s: %s", attempt, term, e)
            if attempt > retries:
                return []
            time.sleep(backoff * attempt)


# ===================== 7️⃣ 翻译节点（并发 + 缓存 + 超时） =====================
@timed_node()
def translate_node(state: typing.Any) -> TermState:
    """Concurrent translation of detected terms with caching and timeouts.
    改为只翻译 state['selected_terms']（如果存在），否则退回到 state['terms']。
    支持批量 LLM 翻译以减少请求次数。
    """
    original = state
    inner, parent, key = _unwrap(state)
    state = inner if isinstance(inner, dict) else {"terms": []}
    state_dict: dict = typing.cast(dict, state)

    translations: Dict[str, List[str]] = {}
    final_translations: Dict[str, str] = {}

    # 优先使用 selected_terms（由 select_top_terms 节点生成）
    terms = state_dict.get("selected_terms") or state_dict.get("terms") or []
    if not terms:
        state_dict["translations"] = translations
        state_dict["final_translations"] = final_translations
        return _rewrap(original, parent, key, state_dict)

    # -----------------------------
    # 🔹 新增：批量调用本地翻译接口（含 LLM 批量翻译）
    # -----------------------------
    try:
        from utils.Get_term import get_translation_candidates_batch
        batch_result = get_translation_candidates_batch(terms)
    except Exception as e:
        logger.warning("批量翻译失败，回退到单项模式: %s", e)
        batch_result = {}

    # -----------------------------
    # 🔹 对每个 term 进行清理、降级回退
    # -----------------------------
    for term in terms:
        candidates = batch_result.get(term, [])

        # 如果批量翻译为空，则尝试旧逻辑（缓存或 web 抓取）
        if not candidates:
            try:
                from utils.Get_term import translate_term as translate_term_external
                candidates = translate_term_external(term)
            except Exception as e:
                logger.warning("单项回退翻译失败 %s: %s", term, e)
                candidates = []

        # clean + dedupe preserving order
        seen = set()
        cleaned = []
        for c in candidates:
            if not c:
                continue
            c_clean = str(c).strip()
            if not c_clean or c_clean in seen:
                continue
            seen.add(c_clean)
            cleaned.append(c_clean)

        translations[term] = cleaned
        if cleaned:
            final_translations[term] = cleaned[0]
        else:
            try:
                local = query_term_translation(term) or []
                if local:
                    translations[term] = local
                    final_translations[term] = local[0]
            except Exception:
                pass

    state_dict["translations"] = translations
    state_dict["final_translations"] = final_translations
    result = _rewrap(original, parent, key, state_dict)
    return result



# ===================== 新增节点：先选出最终要翻译的 top-N 术语 =====================
@timed_node()
def select_top_terms(state: typing.Any) -> TermState:
    """从 state['terms'] 中选出最多 _MAX_TERMS_TO_PROCESS 个高质量术语，输出到 state['selected_terms']。

    优先使用 LLM；失败时使用启发式降级。
    """
    original = state
    inner, parent, key = _unwrap(state)
    state = inner if isinstance(inner, dict) else {"terms": []}
    state_dict: dict = typing.cast(dict, state)

    terms = list(state_dict.get("terms", []) or [])
    topic = state_dict.get("topic", "")

    if not terms:
        state_dict["selected_terms"] = []
        return _rewrap(original, parent, key, state_dict)

    if len(terms) <= _MAX_TERMS_TO_PROCESS:
        state_dict["selected_terms"] = terms
        return _rewrap(original, parent, key, state_dict)

    # 构建 LLM prompt，要求返回 JSON 列表: ["term1","term2",...]
    sel_prompt = f"""
你是术语筛选助手。下面给出若干候选术语，请从中选出最核心的 {_MAX_TERMS_TO_PROCESS} 个术语，按重要性排序并只返回 JSON 数组, 如:["术语1","术语2",...]
主题: {topic}
候选:
"""
    for t in terms:
        sel_prompt += f"\n- {t}"

    raw = None
    chosen_list = None
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            raw = _cached_llm_completion(sel_prompt, system="你是术语筛选助手，严格返回 JSON 列表。")
            if not raw:
                time.sleep(_RETRY_BACKOFF * attempt)
                continue
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                # 归一化并只保留在原始 terms 列表中的项
                chosen_list = []
                term_norm_set = {normalize_candidate(x): x for x in terms}
                for item in parsed:
                    if not isinstance(item, str):
                        continue
                    item_norm = normalize_candidate(item)
                    if item_norm in term_norm_set and item_norm not in chosen_list:
                        chosen_list.append(item_norm)
                if chosen_list:
                    break
        except Exception as _e:
            logger.debug("LLM select_top_terms 失败（尝试 %d）: %s", attempt, _e)
            if raw:
                logger.debug("LLM raw output (select_top_terms): -----\n%s\n-----", raw)
            time.sleep(_RETRY_BACKOFF * attempt)

    if chosen_list is None:
        # 降级：启发式选择，按出现位置和词数排序
        text = state_dict.get("text", "").lower()
        scored = []
        for t in terms:
            score = 0
            score += len(t.split()) * 0.1
            idx = text.find(t.lower())
            if idx >= 0:
                score += max(0.0, 1.0 - (idx / max(1, len(text))))
            scored.append((score, t))
        scored.sort(key=lambda x: -x[0])
        chosen_list = [normalize_candidate(t) for _, t in scored[:_MAX_TERMS_TO_PROCESS]]

    # 保证长度不超过 N
    chosen_list = list(dict.fromkeys(chosen_list))[:_MAX_TERMS_TO_PROCESS]

    state_dict["selected_terms"] = chosen_list
    result = _rewrap(original, parent, key, state_dict)
    return result


# ===================== 重构：只负责从 translations 中选择最终翻译并保存 =====================
@timed_node()
def finalize_translations(state: typing.Any) -> TermState:
    """在已翻译的 translations_map 中为 state['selected_terms']（或 state['terms']）选择最终翻译并保存。
    该节点不再进行大规模 LLM top-N 选择，只负责 pick + save。
    """
    original = state
    inner, parent, key = _unwrap(state)
    state = inner if isinstance(inner, dict) else {"translations": {}}
    state_dict: dict = typing.cast(dict, state)

    translations_map: Dict[str, List[str]] = state_dict.get("translations", {})
    topic = state_dict.get("topic", "")

    terms = list(state_dict.get("selected_terms") or state_dict.get("terms") or list(translations_map.keys()))
    final_translations: Dict[str, str] = {}

    def pick_best_candidate(term: str, candidates: List[str]) -> typing.Optional[str]:
        if not candidates:
            return None
        for c in candidates:
            if re.search(r"[\u4e00-\u9fff]", c):
                return c.strip()
        for c in candidates:
            if c and c.strip():
                return c.strip()
        return None

    for term in terms[:_MAX_TERMS_TO_PROCESS]:
        candidates = translations_map.get(term, [])
        chosen = pick_best_candidate(term, candidates)
        if chosen:
            chosen_norm = chosen.strip()
            if normalize_candidate(term) in _COMMON_GENERIC_WORDS:
                continue
            try:
                if len(term) > 1 and normalize_candidate(term) not in _COMMON_GENERIC_WORDS:
                    save_translation(term, chosen_norm, "term")
            except Exception:
                logger.debug("保存翻译时出错: %s -> %s", term, chosen_norm)
            final_translations[term] = chosen_norm

    state_dict["final_translations"] = final_translations
    result = _rewrap(original, parent, key, state_dict)
    return result


# ===================== 9️⃣ 构建 LangGraph 工作流 =====================
@timed_node()
def build_graph():
    # Use a generic mapping type for the graph's input type to satisfy type checkers
    graph = StateGraph(dict)  # type: ignore
    graph.add_node("extract_candidates", extract_candidates)  # type: ignore
    graph.add_node("filter_terms", filter_terms)  # type: ignore
    # 新增先选择 top-N 的节点
    graph.add_node("select_top_terms", select_top_terms)  # type: ignore
    # 翻译只作用于 selected_terms
    graph.add_node("translate_node", translate_node)  # type: ignore
    # 最终翻译选择器
    graph.add_node("finalize_translations", finalize_translations)  # type: ignore

    graph.set_entry_point("extract_candidates")
    graph.add_edge("extract_candidates", "filter_terms")
    graph.add_edge("filter_terms", "select_top_terms")
    graph.add_edge("select_top_terms", "translate_node")
    graph.add_edge("translate_node", "finalize_translations")
    graph.add_edge("finalize_translations", END)
    return graph.compile()


# ===================== 🔟 手动测试 =====================
if __name__ == "__main__":
    text = "This study proposes a multimodal deep learning framework for semantic segmentation of remote sensing imagery, aiming to address the well-known trade-off between the spatial resolution of panchromatic images and the spectral richness of hyperspectral data. We construct a dual-branch encoder in which one branch focuses on hyperspectral feature extraction and the other on panchromatic spatial enhancement."
    topic = "Multimodal remote sensing image semantic segmentation — emphasizing cross-modal feature fusion, adaptive attention mechanisms, hybrid loss design, and multi-scale supervision to improve segmentation accuracy and generalization across domains"

    state = {
        "text": text,
        "candidates": [],
        "terms": [],
        "topic": topic,
        "translations": {},
        "final_translations": {},
    }

    workflow = build_graph()
    from typing import Any
    result: Any = workflow.invoke(state)  # type: ignore

    print("✅ 候选词数量:", len(result["candidates"]))
    print("✅ 翻译候选:", result.get("translations"))
    print("✅ 最终翻译:", result.get("final_translations"))
    # debug counters
    try:
        print("LLM_CALLS:", LLM_CALLS)
        print("DICT_CALLS:", DICT_CALLS)
    except Exception:
        pass


# ===================== 新增：仅执行术语提取与筛选，不做翻译的工作流 =====================
@timed_node()
def build_graph_terms_only():
    """仅执行术语提取与筛选，不做翻译，便于后置批量翻译。"""
    graph = StateGraph(dict)  # type: ignore
    graph.add_node("extract_candidates", extract_candidates)  # type: ignore
    graph.add_node("filter_terms", filter_terms)  # type: ignore
    graph.add_node("select_top_terms", select_top_terms)  # type: ignore
    graph.set_entry_point("extract_candidates")
    graph.add_edge("extract_candidates", "filter_terms")
    graph.add_edge("filter_terms", "select_top_terms")
    graph.add_edge("select_top_terms", END)
    return graph.compile()

