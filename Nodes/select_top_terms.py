import json
import typing
import logging
import asyncio
import time
from collections import defaultdict

from utils.TermState import TermState
from utils.TimeNode import timed_node
from utils.workflow_adapter import _rewrap, _unwrap
from utils.LLMClientManager import LLMclientManager
from utils.candidate_tool import (
    normalize_candidate,
    _LLM_RETRIES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= 配置区域 =================

BATCH_SIZE = 45

CONCURRENCY_LIMIT = 16

SEGMENT_TIMEOUT = 20


# ===========================================

def _clean_json_string(raw: str) -> str:
    """极简清洗"""
    if not raw: return ""
    raw = raw.strip()
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("[") and p.endswith("]"):
                return p
    return raw


async def call_supplement_segment(candidate_pool: list[str], reason: str, topic: str) -> list[str]:
    """
    【增量补录 - 严格模式】
    """
    safe_candidates = json.dumps(candidate_pool, ensure_ascii=False)
    safe_reason_snippet = str(reason)[:800]
    if len(str(reason)) > 800:
        safe_reason_snippet += "..."

    prompt = f"""
    【任务】：术语提取二轮补录（严格模式）。
    【主题】：{topic}

    【背景】：上一轮筛选因质量不够高被部分驳回。
    以下是审核员的**负面点评（排雷指南）**：
    -------------------------
    {safe_reason_snippet}
    -------------------------

    【指令】：
    1. 请阅读上述点评，领悟“宁缺毋滥”的审核标准。
    2. 从下方的【新备选池】中，**仅**挑选出 1-5 个**绝对核心**的漏网之鱼。
    3. **标准极高**：如果不确定一个词是否完美，就不要选。我们宁愿要一个空列表，也不要噪音。

    【新备选池】：{safe_candidates}

    【返回格式】：JSON字符串数组，如 ["term1"] 或 []。
    """

    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            completion = await LLMclientManager.achat(
                messages=[
                    {"role": "system", "content": "你是一个极度苛刻的术语审稿人。"},
                    {"role": "user", "content": prompt}
                ],
            )
            raw = completion.content
            clean_raw = _clean_json_string(raw)
            if not clean_raw: continue

            parsed = json.loads(clean_raw)
            if isinstance(parsed, list):
                return [str(p) for p in parsed if isinstance(p, str)]
        except Exception as e:
            logger.debug(f"Supplement attempt {attempt} failed: {e}")
            if attempt < _LLM_RETRIES:
                await asyncio.sleep(0.5)

    return []
async def call_segment(seg_terms: list[str], reason: str, topic: str, sem: asyncio.Semaphore) -> list[str]:
    """
    【首次筛选 - 严格模式】
    """
    safe_terms = json.dumps(seg_terms, ensure_ascii=False)
    safe_reason = str(reason)[:200]

    prompt = f"""
    任务：从候选列表中筛选**Top 15%**的核心领域术语。
    主题：{topic}
    排除标准：{safe_reason} (及人名/地名/动词/常用词/非专业词)。

    【原则】：**宁少勿多，宁缺毋滥**。
    1. 只保留那个领域最不可或缺的专有名词。
    2. 如果该批次全是普通词汇，请直接返回空数组 []。

    候选：{safe_terms}
    返回：JSON 字符串数组。
    """

    async with sem:
        for attempt in range(1, _LLM_RETRIES + 1):
            try:
                completion = await LLMclientManager.achat(
                    messages=[
                        {"role": "system", "content": "JSON Generator. Be extremely strict."},
                        {"role": "user", "content": prompt}
                    ],
                )
                raw = completion.content
                clean_raw = _clean_json_string(raw)
                if not clean_raw: continue

                try:
                    parsed = json.loads(clean_raw)
                except json.JSONDecodeError:
                    parsed = json.loads(clean_raw.replace("'", '"'))

                if isinstance(parsed, list):
                    return [str(p) for p in parsed if isinstance(p, (str, int, float))]

            except Exception as e:
                logger.debug(f"Segment attempt {attempt} error: {e}")
                if attempt < _LLM_RETRIES:
                    await asyncio.sleep(0.5)

    return []
async def call_segment_FAST(seg_terms: list[str], topic: str, sem: asyncio.Semaphore) -> list[str]:
    """
    【首次筛选 - 严格模式】
    """
    safe_terms = json.dumps(seg_terms, ensure_ascii=False)

    prompt = f"""
    任务：从候选列表中筛选**Top 25%**的核心领域术语。
    主题：{topic}
    排除标准： 排除人名/地名/动词/常用词/非主题领域的专业词

    【原则】：**宁少勿多，宁缺毋滥**。
    1. 只保留那个领域最不可或缺的专有名词。
    2. 如果该批次全是普通词汇，请直接返回空数组 []。

    候选：{safe_terms}
    返回：JSON 字符串数组。
    """

    async with sem:
        for attempt in range(1, _LLM_RETRIES + 1):
            try:
                completion = await LLMclientManager.achat(
                    messages=[
                        {"role": "system", "content": "JSON Generator. Be extremely strict."},
                        {"role": "user", "content": prompt}
                    ],
                )
                raw = completion.content
                clean_raw = _clean_json_string(raw)
                if not clean_raw: continue

                try:
                    parsed = json.loads(clean_raw)
                except json.JSONDecodeError:
                    parsed = json.loads(clean_raw.replace("'", '"'))

                if isinstance(parsed, list):
                    return [str(p) for p in parsed if isinstance(p, (str, int, float))]

            except Exception as e:
                logger.debug(f"Segment attempt {attempt} error: {e}")
                if attempt < _LLM_RETRIES:
                    await asyncio.sleep(0.5)

    return []


@timed_node()
async def select_top_terms(state: TermState) -> TermState:
    inner, parent, key = _unwrap(state)
    sd: TermState = inner if isinstance(inner, dict) else TermState()

    candidates = sd.get("candidates") or []
    if not candidates:
        return _pack_result([], {})

    current_selected = sd.get("selected_terms") or []
    reflect_attempts = int(sd.get("reflect_attempts", 0) or 0)

    remove_list = sd.get("reflect_remove_terms") or []
    remove_list = remove_list[0:int(len(current_selected) * 0.1)]  # 防止过大
    reason = sd.get("reflect_reason", "")
    topic = sd.get("summary", "")

    # 1. 建立映射
    norm_to_original = {}
    for t in candidates:
        n = normalize_candidate(t)
        if n:
            if n not in norm_to_original or len(t) > len(norm_to_original[n]):
                norm_to_original[n] = t

    remove_norms = {normalize_candidate(t) for t in remove_list if t}



    if reflect_attempts > 0 and current_selected:
        logger.info(f"Reflect Attempt {reflect_attempts}: Strict Incremental Mode.")

        # 1. 剔除坏词
        cleaned_selected = []
        for t in current_selected:
            if normalize_candidate(t) not in remove_norms:
                cleaned_selected.append(t)
        final_list = cleaned_selected

        return _pack_result(final_list, sd.get("term_to_chunks", {}))

    # =====================================================
    # 分支 B: 首次筛选 (Strict First Pass)
    # =====================================================
    logger.info("First pass selection (Strict Mode).")

    valid_candidates_list = []
    for n, t in norm_to_original.items():
        if n not in remove_norms:
            valid_candidates_list.append(t)

    segments = [valid_candidates_list[i:i + BATCH_SIZE] for i in range(0, len(valid_candidates_list), BATCH_SIZE)]

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [asyncio.create_task(call_segment(seg, reason, topic, sem)) for seg in segments]

    parsed_results = []
    if tasks:
        try:
            async def run_with_timeout(t):
                try:
                    return await asyncio.wait_for(t, timeout=SEGMENT_TIMEOUT)
                except asyncio.TimeoutError:
                    return []
                except Exception:
                    return []

            safe_tasks = [run_with_timeout(t) for t in tasks]
            results = await asyncio.gather(*safe_tasks)
            parsed_results = [r for r in results if r]
        except Exception as e:
            logger.error(f"Async Gather Error: {e}")

    # 聚合
    chosen_norms = set()
    final_selected_list = []

    for seg_res in parsed_results:
        for item in seg_res:
            nk = normalize_candidate(item)
            if nk and nk in norm_to_original and nk not in chosen_norms:
                chosen_norms.add(nk)
                final_selected_list.append(norm_to_original[nk])

    # Fallback (宁少勿多：如果实在太少，只补前15个，不再补50个)
    if len(final_selected_list) < 3:
        logger.info("Selection critical low, triggering conservative fallback.")
        scored_fb = []
        text_lower = str(topic).lower()
        for t in valid_candidates_list:
            score = 0
            if len(t) > 3: score += len(t)
            if t.lower() in text_lower: score += 20  # 强相关
            scored_fb.append((score, t))
        scored_fb.sort(key=lambda x: -x[0])
        # 只取 Top 15
        final_selected_list = [t for _, t in scored_fb[:50]]

    print(f"Async First Pass Selected: {len(final_selected_list)}")

    return _pack_result(final_selected_list, sd.get("term_to_chunks", {}))

@timed_node()
async def select_top_terms_FAST(state: TermState) -> TermState:
    inner, parent, key = _unwrap(state)
    sd: TermState = inner if isinstance(inner, dict) else TermState()

    candidates = sd.get("candidates") or []
    if not candidates:
        return _pack_result([], {})

    topic = sd.get("summary", "")

    # 1. 建立映射
    norm_to_original = {}
    for t in candidates:
        n = normalize_candidate(t)
        if n:
            if n not in norm_to_original or len(t) > len(norm_to_original[n]):
                norm_to_original[n] = t


    logger.info("First pass selection (Strict Mode).")
    valid_candidates_list = []
    for n, t in norm_to_original.items():
        valid_candidates_list.append(t)

    segments = [valid_candidates_list[i:i + BATCH_SIZE] for i in range(0, len(valid_candidates_list), BATCH_SIZE)]

    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [asyncio.create_task(call_segment_FAST(seg,topic, sem)) for seg in segments]

    parsed_results = []
    if tasks:
        try:
            async def run_with_timeout(t):
                try:
                    return await asyncio.wait_for(t, timeout=SEGMENT_TIMEOUT)
                except asyncio.TimeoutError:
                    return []
                except Exception:
                    return []

            safe_tasks = [run_with_timeout(t) for t in tasks]
            results = await asyncio.gather(*safe_tasks)
            parsed_results = [r for r in results if r]
        except Exception as e:
            logger.error(f"Async Gather Error: {e}")

    # 聚合
    chosen_norms = set()
    final_selected_list = []

    for seg_res in parsed_results:
        for item in seg_res:
            nk = normalize_candidate(item)
            if nk and nk in norm_to_original and nk not in chosen_norms:
                chosen_norms.add(nk)
                final_selected_list.append(norm_to_original[nk])

    # Fallback (宁少勿多：如果实在太少，只补前15个，不再补50个)
    if len(final_selected_list) < 3:
        logger.info("Selection critical low, triggering conservative fallback.")
        scored_fb = []
        text_lower = str(topic).lower()
        for t in valid_candidates_list:
            score = 0
            if len(t) > 3: score += len(t)
            if t.lower() in text_lower: score += 20  # 强相关
            scored_fb.append((score, t))
        scored_fb.sort(key=lambda x: -x[0])
        # 只取 Top 15
        final_selected_list = [t for _, t in scored_fb[:15]]

    print(f"Async First Pass Selected: {len(final_selected_list)}")

    return _pack_result(final_selected_list, sd.get("term_to_chunks", {}))

def _pack_result(selected_terms: list, term_to_chunks: dict) -> TermState:
    """打包结果"""
    chunk_terms_map = defaultdict(list)
    for term in selected_terms:
        cids = term_to_chunks.get(term, [])
        for cid in cids:
            chunk_terms_map[cid].append(term)

    chunk_terms_list = [
        {"chunk_id": str(cid), "terms": terms}
        for cid, terms in chunk_terms_map.items()
    ]

    return typing.cast(TermState, {
        "selected_terms": selected_terms,
        "chunk_terms": chunk_terms_list,
    })