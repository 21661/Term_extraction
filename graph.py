from typing import List, TypedDict, Dict
from langgraph.graph import StateGraph, END
import spacy
import json
import time
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.zhipu_client import client
from utils.db_interface import query_term_translation, save_translation
from utils.web_fetcher import fetch_core_translations
from utils.Get_term import translate_term as translate_term_external, get_translation_candidates_batch,translate_term
from utils.workflow_adapter import _rewrap,_unwrap
from utils.extract_candidates import extract_candidates
from utils.TimeNode import timed_node
from utils.candidate_tool import normalize_candidate, is_noise_candidate, dedupe_keep_longest, _MAX_TERMS_TO_PROCESS, _LLM_RETRIES, _RETRY_BACKOFF, _COMMON_GENERIC_WORDS
import typing

# ===================== 1️⃣ 定义状态类型 =====================
class TermState(TypedDict):
    text: str
    candidates: List[str]
    terms: List[str]
    topic: str
    translations: Dict[str, List[str]]  # term -> 候选翻译
    final_translations: Dict[str, str]


# ===================== 2️⃣ 初始化 =====================
nlp = spacy.load("en_core_web_trf")

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# ----------------- Helpers (no caching) -----------------
def llm_completion(prompt: str, system: str = "你是术语分类助手") -> str:
    """直接调用 LLM，同步阻塞，不做缓存或超时保护。"""
    try:
        completion = client.chat.completions.create(
            model="glm-4.5-flash",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            temperature=0,
        )
        return _safe_extract_completion_content(completion)
    except Exception as e:
        logger.warning("LLM direct call failed: %s", e)
        return "None"
    # no caching or decorator returned; function ends here
# ===================== 5️⃣ 判断抓取方式 =====================
def decide_post_db_method(term: str) -> str:
    term_lower = term.lower()
    simple_words = {"data", "set", "map", "input", "output", "value", "use", "function"}
    if term_lower in simple_words or len(term) <= 3:
        return "dict"
    if len(term.split()) > 1 or re.search(r"[-_/]", term):
        return "llm"
    return "dict"

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
    topic = state_dict.get("topic") or state_dict.get("summary") or ""
    if not terms:
        state_dict["translations"] = translations
        state_dict["final_translations"] = final_translations
        return _rewrap(original, parent, key, state_dict)

    # -----------------------------
    # 🔹 新增：批量调用本地翻译接口（含 LLM 批量翻译），传入 topic
    # -----------------------------
    try:
        from utils.Get_term import get_translation_candidates_batch as _batch
        batch_result = _batch(terms, topic=topic)
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
                candidates = translate_term_external(term, topic=topic)
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

    保持术语的原始大小写；只在内部比较/去重时使用 normalize_candidate。
    """
    original = state
    inner, parent, key = _unwrap(state)
    state = inner if isinstance(inner, dict) else {"terms": []}
    state_dict: dict = typing.cast(dict, state)

    # 尝试从已有的 terms 获取；若没有，则退回到 candidates
    terms = list(state_dict.get("terms") or [])
    topic = state_dict.get("topic", "")

    # 若 terms 为空，从 candidates 基于原始字符串构建，噪声过滤和去重时用 normalize_candidate
    if not terms:
        candidates = state_dict.get("candidates") or []
        filtered_original: list[str] = []
        for c in candidates:
            if not isinstance(c, str):
                c = str(c)
            if not c.strip():
                continue
            if is_noise_candidate(c):
                continue
            filtered_original.append(c)
        # 使用 normalize_candidate 做 key 去重，但保留第一出现的原始形式
        seen = set()
        deduped: list[str] = []
        for t in filtered_original:
            key_norm = normalize_candidate(t)
            if not key_norm:
                continue
            if key_norm in seen:
                continue
            seen.add(key_norm)
            deduped.append(t)
        terms = deduped

    if not terms:
        state_dict["selected_terms"] = []
        return _rewrap(original, parent, key, state_dict)

    if len(terms) <= _MAX_TERMS_TO_PROCESS:
        state_dict["selected_terms"] = terms
        return _rewrap(original, parent, key, state_dict)

    # 构建 LLM prompt
    sel_prompt = f"""
你是术语筛选助手。下面给出若干候选术语，请从中选出最核心的术语，要求必须是不容易翻译或者多义的，很容易翻译的，基础的务必排除。不超过{_MAX_TERMS_TO_PROCESS}个。按重要性排序并只返回 JSON 数组, 如:["术语1","术语2",...]
主题: {topic}
候选:
"""
    for t in terms:
        sel_prompt += f"\n- {t}"

    raw = None
    chosen_norm_keys: typing.Optional[list[str]] = None
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            raw = llm_completion(sel_prompt, system="你是术语筛选助手，严格返回 JSON 列表。")
            if not raw:
                time.sleep(_RETRY_BACKOFF * attempt)
                continue
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                # 使用 normalize_candidate 匹配 LLM 返回的术语，但最终保留原始形式
                term_norm_to_original: dict[str, str] = {}
                for t in terms:
                    key_norm = normalize_candidate(t)
                    if key_norm and key_norm not in term_norm_to_original:
                        term_norm_to_original[key_norm] = t
                chosen_norm_keys = []
                for item in parsed:
                    if not isinstance(item, str):
                        continue
                    item_norm = normalize_candidate(item)
                    if item_norm in term_norm_to_original and item_norm not in chosen_norm_keys:
                        chosen_norm_keys.append(item_norm)
                if chosen_norm_keys:
                    break
        except Exception as _e:
            logger.debug("LLM select_top_terms 失败（尝试 %d）: %s", attempt, _e)
            if raw:
                logger.debug("LLM raw output (select_top_terms): -----\n%s\n-----", raw)
            time.sleep(_RETRY_BACKOFF * attempt)

    if chosen_norm_keys is None:
        # 启发式降级：这里也保持原始大小写
        text_lower = state_dict.get("text", "").lower()
        scored: list[tuple[float, str]] = []
        for t in terms:
            score = 0.0
            score += len(t.split()) * 0.1
            idx = text_lower.find(t.lower())
            if idx >= 0:
                score += max(0.0, 1.0 - (idx / max(1, len(text_lower))))
            scored.append((score, t))
        scored.sort(key=lambda x: -x[0])
        # 直接取前 N 个原始形式
        selected_terms = [t for _, t in scored[:_MAX_TERMS_TO_PROCESS]]
    else:
        # 根据 chosen_norm_keys 映射回原始形式
        term_norm_to_original: dict[str, str] = {}
        for t in terms:
            key_norm = normalize_candidate(t)
            if key_norm and key_norm not in term_norm_to_original:
                term_norm_to_original[key_norm] = t
        selected_terms = []
        for nk in chosen_norm_keys:
            orig = term_norm_to_original.get(nk)
            if orig and orig not in selected_terms:
                selected_terms.append(orig)

    # 保证长度不超过 N
    selected_terms = selected_terms[:_MAX_TERMS_TO_PROCESS]
    state_dict["selected_terms"] = selected_terms
    return _rewrap(original, parent, key, state_dict)


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
            return "none"
        for c in candidates:
            if re.search(r"[\u4e00-\u9fff]", c):
                return c.strip()
        for c in candidates:
            if c and c.strip():
                return c.strip()
        return "none"

    for term in terms[:_MAX_TERMS_TO_PROCESS]:
        candidates = translations_map.get(term, [])
        chosen = pick_best_candidate(term, candidates)
        if chosen and chosen.lower() != "none":
            chosen_norm = chosen.strip()
            # 使用 .lower() 进行临时比较
            if term.lower() in _COMMON_GENERIC_WORDS:
                continue
            try:
                # 使用 .lower() 进行临时比较
                if len(term) > 1 and term.lower() not in _COMMON_GENERIC_WORDS:
                    save_translation(term, chosen_norm, "term")
            except Exception:
                logger.debug("保存翻译时出错: %s -> %s", term, chosen_norm)
            final_translations[term] = chosen_norm
        else:
            final_translations[term] = "none"

    state_dict["final_translations"] = final_translations
    result = _rewrap(original, parent, key, state_dict)
    return result


# ===================== 9️⃣ 构建 LangGraph 工作流（重构为 main.extract 的批处理流程） =====================
_TERMS_ONLY_WORKFLOW = None


@timed_node()
def _init_extract_state(state: typing.Any) -> dict:
    """初始化 extract 流程所需字段。输入需包含: summary(str), chunks(dict[str,str])"""
    original = state
    inner, parent, key = _unwrap(state)
    state = inner if isinstance(inner, dict) else {}
    sd: dict = typing.cast(dict, state)
    sd.setdefault("summary", sd.get("topic", "") or "")
    sd.setdefault("chunks", {})
    sd.setdefault("per_chunk_results", [])
    sd.setdefault("unique_terms", [])
    sd.setdefault("translations_map", {})
    sd.setdefault("termAnnotations", {})
    sd.setdefault("stats", {})
    sd.setdefault("errors", [])
    return _rewrap(original, parent, key, sd)


@timed_node()
def _terms_only_batch(state: typing.Any) -> dict:
    """对每个 chunk 运行术语提取（不翻译），兼容 main.extract 的 process_chunk_terms_only 行为。"""
    original = state
    inner, parent, key = _unwrap(state)
    sd: dict = typing.cast(dict, inner if isinstance(inner, dict) else {})

    summary = sd.get("summary", "")
    chunks: Dict[str, str] = sd.get("chunks", {}) or {}

    global _TERMS_ONLY_WORKFLOW
    if _TERMS_ONLY_WORKFLOW is None:
        try:
            _TERMS_ONLY_WORKFLOW = build_graph_terms_only()
        except Exception as e:
            logger.exception("Failed to compile terms-only workflow in _terms_only_batch: %s", e)
            _TERMS_ONLY_WORKFLOW = None

    per_chunk_results: List[Dict[str, typing.Any]] = []
    errors: List[Dict[str, str]] = []

    # 使用线程池并发处理每个 chunk
    def _process_one(cid: str, ctext: typing.Any) -> Dict[str, typing.Any]:
        try:
            text = ctext if isinstance(ctext, str) else str(ctext)
            init_state = {
                "text": text,
                "candidates": [],
                "terms": [],
                "topic": summary,
                "translations": {},
                "final_translations": {},
                "term_types": {},
            }
            if _TERMS_ONLY_WORKFLOW is None:
                raise RuntimeError("terms_only_workflow_not_initialized")
            raw_res = _TERMS_ONLY_WORKFLOW.invoke(init_state)
            if isinstance(raw_res, dict):
                result = raw_res
            else:
                result = {}
            selected_terms = result.get("selected_terms") or result.get("terms") or []
            term_types = result.get("term_types", {})
            return {"chunk_id": str(cid), "terms": list(selected_terms), "term_types": dict(term_types)}
        except Exception as e:
            return {"chunk_id": str(cid), "terms": [], "term_types": {}, "error": str(e)}

    if chunks:
        max_workers = min(5, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cid = {executor.submit(_process_one, cid, ctext): str(cid) for cid, ctext in chunks.items()}
            for fut in as_completed(future_to_cid):
                r = fut.result()
                if r.get("error"):
                    errors.append({"chunk_id": r.get("chunk_id"), "error": r.get("error")})
                per_chunk_results.append(r)
    else:
        per_chunk_results = []

    sd["per_chunk_results"] = per_chunk_results
    sd.setdefault("errors", []).extend(errors)
    return _rewrap(original, parent, key, sd)


@timed_node()
def _aggregate_unique_terms(state: typing.Any) -> dict:
    original = state
    inner, parent, key = _unwrap(state)
    sd: dict = typing.cast(dict, inner if isinstance(inner, dict) else {})

    all_terms: List[str] = []
    for r in sd.get("per_chunk_results", []):
        all_terms.extend(r.get("terms", []))
    unique_terms = sorted(set(t for t in all_terms if isinstance(t, str) and t.strip()))
    sd["unique_terms"] = unique_terms
    return _rewrap(original, parent, key, sd)


@timed_node()
def _batch_translate_with_fallback(state: typing.Any) -> dict:
    original = state
    inner, parent, key = _unwrap(state)
    sd: dict = typing.cast(dict, inner if isinstance(inner, dict) else {})

    unique_terms: List[str] = sd.get("unique_terms", [])
    translations_map: Dict[str, List[str]] = {}
    topic = sd.get("summary", "")

    # 批量翻译（使用线程池并行处理分批）
    try:
        if unique_terms:
            batch_size = 50
            batches = [unique_terms[i:i + batch_size] for i in range(0, len(unique_terms), batch_size)]
            max_workers = min(5, len(batches))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(get_translation_candidates_batch, batch, batch_size=batch_size, topic=topic): idx
                    for idx, batch in enumerate(batches)
                }
                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    try:
                        res = fut.result() or {}
                        if isinstance(res, dict):
                            translations_map.update(res)
                        else:
                            logger.warning("Batch %d returned non-dict result, ignored", idx)
                    except Exception as e:
                        logger.warning("Batch translation failed for batch %d: %s", idx, e)
    except Exception as e:
        logger.warning("Batch translation failed, will fallback to single + db: %s", e)
        translations_map = {}

    # 单项回退 + DB 回退
    for t in unique_terms:
        if translations_map.get(t):
            continue
        try:
            single = translate_term_external(t, topic=topic) or []
        except Exception:
            single = []
        if not single:
            try:
                local = query_term_translation(t) or []
                single = local
            except Exception:
                pass
        translations_map[t] = single or []

    sd["translations_map"] = translations_map
    return _rewrap(original, parent, key, sd)
@timed_node()
def _single_translate_concurrent(state: typing.Any) -> dict:
    original = state
    inner, parent, key = _unwrap(state)
    sd: dict = typing.cast(dict, inner if isinstance(inner, dict) else {})

    unique_terms: List[str] = sd.get("unique_terms", [])
    topic = sd.get("summary", "")
    translations_map: Dict[str, List[str]] = {}

    if not unique_terms:
        sd["translations_map"] = translations_map
        return _rewrap(original, parent, key, sd)

    # 可用并发数（你可根据 RPM 调整）
    max_workers = 50

    def worker(term: str):
        """
        单个词翻译任务，自动网络重试 3 次。
        失败或未找到翻译 → 返回空列表。
        """
        retry_delays = [0.3, 0.6, 1.0]  # 渐进延迟

        for attempt in range(3):
            try:
                res = translate_term(term, topic=topic)
                if res:
                    return term, res
                else:
                    # 找不到翻译不是网络错误，不必重试
                    return term, []
            except Exception as e:
                logger.warning(
                    "Translate attempt %d failed for %s: %s",
                    attempt + 1, term, e
                )
                if attempt < 2:
                    time.sleep(retry_delays[attempt])

        # 三次都失败 → 给空列表
        return term, []

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_term = {
                executor.submit(worker, t): t for t in unique_terms
            }

            for fut in as_completed(future_to_term):
                term = future_to_term[fut]
                try:
                    k, v = fut.result()
                    translations_map[k] = v
                except Exception as e:
                    logger.warning("Unexpected translation failure for %s: %s", term, e)
                    translations_map[term] = []

    except Exception as e:
        logger.exception("Concurrent translation failed: %s", e)
        translations_map = {t: [] for t in unique_terms}

    sd["translations_map"] = translations_map
    return _rewrap(original, parent, key, sd)


@timed_node()
def _assemble_annotations(state: typing.Any) -> dict:
    original = state
    inner, parent, key = _unwrap(state)
    sd: dict = typing.cast(dict, inner if isinstance(inner, dict) else {})

    chunks: Dict[str, str] = sd.get("chunks", {}) or {}
    per_chunk_results: List[Dict[str, typing.Any]] = sd.get("per_chunk_results", [])
    translations_map: Dict[str, List[str]] = sd.get("translations_map", {})

    per_chunk_map: Dict[str, Dict[str, typing.Any]] = {str(r.get("chunk_id")): r for r in per_chunk_results if r and r.get("chunk_id") is not None}

    term_annotations: Dict[str, typing.Any] = {}
    translated_count = 0

    def _pick_best(term: str, cands: List[str]) -> str:
        if not cands:
            return "none"
        # 忽略大小写和空格后与原词相同的候选
        term_norm = term.strip().lower()
        for c in cands:
            if not c:
                continue
            c_norm = c.strip()
            # 如果候选词与原词相同（忽略大小写），则跳过
            if c_norm.lower() == term_norm:
                continue
            # 优先返回包含中文的翻译
            if re.search(r"[\u4e00-\u9fff]", c_norm):
                return c_norm
        # 如果没有中文翻译，但有其他不同于原文的翻译
        for c in cands:
            if not c:
                continue
            c_norm = c.strip()
            if c_norm.lower() != term_norm:
                return c_norm
        # 所有候选都与原词相同或为空
        return "none"

    for idx, cid in enumerate(chunks.keys(), start=1):
        r = per_chunk_map.get(str(cid), {"terms": [], "term_types": {}})
        items: List[Dict[str, typing.Any]] = []
        for t in r.get("terms", []):
            # 这里 t 是在 terms_only_workflow 中产生的，已保持原始大小写
            cands = translations_map.get(t, [])
            chosen = _pick_best(t, cands)
            if chosen and chosen.lower() != "none":
                translated_count += 1
            # term 字段直接使用 t，不做 lower 处理
            items.append({"term": t, "translation": chosen})
        term_annotations[str(idx)] = items

    stats = {
        "total_chunks": len(chunks),
        "unique_terms": len(sd.get("unique_terms", [])),
        "translated_terms": translated_count,
    }

    sd["termAnnotations"] = term_annotations
    sd["stats"] = stats
    return _rewrap(original, parent, key, sd)


@timed_node()
def build_graph():
    """构建与 main.extract 等价的批处理图。

    输入状态需要包含：
    - summary: Optional[str]
    - chunks: Dict[str, str]

    输出（写入状态）：
    - termAnnotations: Dict[str, Any]
    - stats: Dict[str, Any]
    - errors: List[Dict[str,str]]（若存在）
    - 以及中间结果：per_chunk_results, unique_terms, translations_map
    """
    graph = StateGraph(dict)  # type: ignore
    graph.add_node("extract_candidates", extract_candidates)  # type: ignore
    # 新增先选择 top-N 的节点
    graph.add_node("select_top_terms", select_top_terms)  # type: ignore

    # 最终翻译选择器
    graph.add_node("finalize_translations", finalize_translations)  # type: ignore
    graph.add_node("init", _init_extract_state)  # type: ignore
    graph.add_node("terms_only_batch", _terms_only_batch)  # type: ignore
    graph.add_node("aggregate_unique_terms", _aggregate_unique_terms)  # type: ignore
    graph.add_node("batch_translate", _single_translate_concurrent)  # type: ignore
    graph.add_node("assemble_annotations", _assemble_annotations)  # type: ignore

    graph.set_entry_point("init")
    graph.add_edge("init", "terms_only_batch")
    graph.add_edge("terms_only_batch", "aggregate_unique_terms")
    graph.add_edge("aggregate_unique_terms", "batch_translate")
    graph.add_edge("batch_translate", "assemble_annotations")
    graph.add_edge("assemble_annotations", END)
    return graph.compile()
# ===================== 新增：仅执行术语提取与筛选，不做翻译的工作流 =====================
@timed_node()
def build_graph_terms_only():
    """仅执行术语提取与筛选，不做翻译，便于后置批量翻译。"""
    graph = StateGraph(dict)  # type: ignore
    graph.add_node("extract_candidates", extract_candidates)  # type: ignore
    # 直接使用 select_top_terms，不再依赖 filter_terms
    graph.add_node("select_top_terms", select_top_terms)  # type: ignore
    graph.set_entry_point("extract_candidates")
    graph.add_edge("extract_candidates", "select_top_terms")
    graph.add_edge("select_top_terms", END)
    return graph.compile()
