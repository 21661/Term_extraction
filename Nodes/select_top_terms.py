import json

from utils.LLMManager import LLMConfigModel, AgentConfigRequestModel, AgentManager
from utils.TermState import TermState
from utils.TimeNode import timed_node
from utils.workflow_adapter import _rewrap,_unwrap
from utils.LLMClientManager import LLMclientManager
from concurrent.futures import ThreadPoolExecutor, as_completed
import typing
import logging
from collections import defaultdict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from utils.candidate_tool import (
    normalize_candidate,
    is_noise_candidate,
    _LLM_RETRIES,
    _RETRY_BACKOFF,
)

# ============================================================
# LLM 子段调用
# ============================================================
def call_segment(seg_terms: list[str], reason: str, topic: str) -> list[str]:
    raw = None
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            prompt = _build_prompt(seg_terms, reason, topic)
            completion = LLMclientManager.chat(
                messages=[
                    {"role": "system", "content": "只返回json列表，不要使用 ```json ``` 或其他格式，不要加解释。"},
                    {"role": "user", "content": prompt}
                ],
            )
            raw = completion.content

            parsed = json.loads(raw)

            # 3. 验证解析结果
            if isinstance(parsed, list):
                return [p for p in parsed if isinstance(p, str)]
            if isinstance(parsed, list):
                return parsed
        except Exception as e:
            logger.debug("LLM segment failed attempt %d: %s", attempt, e)
            if raw:
                logger.debug("LLM raw output: %s", raw)
            time.sleep(_RETRY_BACKOFF * attempt)
    return []


def _build_prompt(segment_terms: list[str], reason: str, topic: str) -> str:
    sel_prompt = f"""
    "role": 你是术语筛选助手。
    "task": 从候选术语中选出核心、专业、不可翻译的术语。
    "constraints": ["人名地名排除","泛词排除","按重要性排序并返回 JSON 列表","优先返回质量好的词"]
    "mistakes": {reason}
    "主题": {topic}
    "候选": {segment_terms}
    """
    return sel_prompt


# ============================================================
# 主筛选逻辑（LLM + 映射）
# ============================================================
@timed_node()
def select_top_terms(state: TermState) -> TermState:
    inner, parent, key = _unwrap(state)
    sd: TermState = inner if isinstance(inner, dict) else TermState()

    remove_list = sd.get("reflect_remove_terms") or []
    candidates = sd.get("candidates") or []
    print(f"Total candidates before selection: {len(candidates)}")

    # 1. 准备排除列表
    remove_norms = {normalize_candidate(rt) for rt in remove_list if normalize_candidate(rt)}
    if len(remove_norms) > len(candidates) // 10:
        remove_norms = list(remove_norms)[: len(candidates) // 10]  # 修正 set 切片问题
        remove_norms = set(remove_norms)

    term_to_chunks = sd.get("term_to_chunks") or {}
    topic = sd.get("summary", "")

    _reasons = sd.get("reflect_reason", []) or []
    reason = " | ".join([str(r) for r in _reasons if isinstance(r, str)]) if isinstance(_reasons, list) else str(
        _reasons)

    # 2. 分段并发调用 LLM

    chunk_size = min(50, len(candidates) // 3 or 1)
    segments = [candidates[i:i + chunk_size] for i in range(0, len(candidates), chunk_size)]
    parsed_results: list[list[str]] = []

    try:
        max_workers = min(16, len(segments))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(call_segment, seg, reason, topic) for seg in segments]
            for fut in as_completed(futures):
                parsed_results.append(fut.result() or [])
    except Exception as e:
        logger.debug("Segments concurrent calling failed: %s", e)
        parsed_results = []

    # 3. 收集 LLM 选中的词
    norm_to_original = {normalize_candidate(t): t for t in candidates if normalize_candidate(t)}
    chosen_norms = []

    for seg in parsed_results:
        if(len(seg) > 8):
            seg = seg[0:8]
        for item in seg:
            nk = normalize_candidate(item)
            # 保持顺序添加，去重
            if nk and nk in norm_to_original and nk not in chosen_norms:
                chosen_norms.append(nk)

    # 4. Fallback 逻辑修正：不要在这里全局截断成 20 个！
    if not chosen_norms:
        print("LLM did not select any terms, applying heuristic fallback.")
        scored = []
        text_lower = sd.get("summary", "").lower()
        for t in candidates:
            score = len(t.split()) * 0.1
            idx = text_lower.find(t.lower())
            if idx >= 0:
                score += max(0, 1 - idx / max(1, len(text_lower)))
            scored.append((score, t))
        scored.sort(key=lambda x: -x[0])

        # 【修改点】如果走 heuristic，取前 50% 或者至少取 100 个，保证每个 chunk 都有词分
        fallback_limit = max(100, int(len(candidates) * 0.5))
        selected_terms = [t for _, t in scored[:fallback_limit]]
    else:
        selected_terms = [
            norm_to_original[nk]
            for nk in chosen_norms if nk in norm_to_original
        ]

    # 过滤掉 remove_list 中的词
    selected_terms = [t for t in selected_terms if normalize_candidate(t) not in remove_norms]
    print(f"Selected terms count after filtering: {len(selected_terms)}")

    # 5. 分发回 Chunk 并执行“每个 Chunk 20 个”的限制
    chunk_terms = defaultdict(list)
    for term in selected_terms:
        for cid in term_to_chunks.get(term, []):
            chunk_terms[cid].append(term)

    chunk_terms = [
        {"chunk_id": str(cid), "terms": terms}
        for cid, terms in chunk_terms.items()]


    print(f"Global selected terms count: {len(selected_terms)}")
    print(f"Chunk terms distribution: {chunk_terms}")

    return typing.cast(TermState, {
        "selected_terms": selected_terms,
        "chunk_terms": chunk_terms,
    })
