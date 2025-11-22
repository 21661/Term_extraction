import json
import logging
import typing
import asyncio
from collections import defaultdict
from typing import List, Any

# 假设你的项目中有这些导入
from utils.LLMManager import LLMConfigModel, AgentConfigRequestModel, AgentManager
from utils.TermState import TermState
from utils.TimeNode import timed_node
from utils.workflow_adapter import _unwrap
from utils.LLMClientManager import LLMclientManager
from utils.candidate_tool import (
    normalize_candidate,
    _LLM_RETRIES,
    _RETRY_BACKOFF,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


BATCH_SIZE = 60  # 建议每批处理 50 个词，既不超 Context 又能利用并发


def _build_prompt(segment_terms: list[str], reason: str, topic: str) -> str:
    safe_terms = json.dumps(segment_terms, ensure_ascii=False)
    sel_prompt = f"""
    "role": 你是术语筛选助手。
    "task": 从候选术语中选出核心、专业、不可翻译的术语。
    "constraints": ["人名地名排除","泛词排除","按重要性排序并返回 JSON 列表","优先返回质量好的词"]
    "mistakes": {reason}
    "主题": {topic}
    "候选": {safe_terms}
    """
    return sel_prompt


# ============================================================
# 异步 LLM 子段调用
# ============================================================
async def call_segment_async(seg_terms: list[str], reason: str, topic: str, batch_idx: int) -> list[str]:
    """
    异步调用 LLM 进行分段筛选
    """
    raw = None
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            prompt = _build_prompt(seg_terms, reason, topic)

            # --- 关键修改：使用 achat ---
            completion = await LLMclientManager.achat(
                messages=[
                    {"role": "system", "content": "只返回json列表，不要使用 ```json ``` 或其他格式，不要加解释。"},
                    {"role": "user", "content": prompt}
                ],
            )

            raw = completion.content

            # 清洗可能存在的 Markdown 标记
            clean_raw = raw.strip()
            if clean_raw.startswith("```json"):
                clean_raw = clean_raw[7:]
            if clean_raw.startswith("```"):
                clean_raw = clean_raw[3:]
            if clean_raw.endswith("```"):
                clean_raw = clean_raw[:-3]

            parsed = json.loads(clean_raw)

            # 验证解析结果
            if isinstance(parsed, list):
                valid_terms = [p for p in parsed if isinstance(p, str)]
                return valid_terms

        except Exception as e:
            logger.warning(f"[Batch-{batch_idx}] LLM attempt {attempt} failed: {e}")
            if raw:
                logger.debug(f"[Batch-{batch_idx}] Raw output: {raw}")
            # --- 关键修改：异步等待 ---
            await asyncio.sleep(_RETRY_BACKOFF * attempt)

    logger.error(f"[Batch-{batch_idx}] All retries failed.")
    return []


# ============================================================
# 主筛选逻辑（异步）
# ============================================================
@timed_node()
async def select_top_terms(state: TermState) -> TermState:
    inner, parent, key = _unwrap(state)
    sd: TermState = inner if isinstance(inner, dict) else TermState()

    remove_list = sd.get("reflect_remove_terms") or []
    candidates = sd.get("candidates") or []
    print(f"Total candidates before selection: {len(candidates)}")

    # 1. 准备排除列表 (CPU计算，保持同步)
    remove_norms = {normalize_candidate(rt) for rt in remove_list if normalize_candidate(rt)}
    # 限制排除列表大小，避免过度计算
    limit_exclude = max(10, len(candidates) // 10)
    if len(remove_norms) > limit_exclude:
        remove_norms = set(list(remove_norms)[:limit_exclude])

    term_to_chunks = sd.get("term_to_chunks") or {}
    topic = sd.get("summary", "")

    _reasons = sd.get("reflect_reason", []) or []
    reason = " | ".join([str(r) for r in _reasons if isinstance(r, str)]) if isinstance(_reasons, list) else str(
        _reasons)

    # 2. 准备分段 (Batching)
    # 使用切片生成 segments
    segments = [candidates[i:i + BATCH_SIZE] for i in range(0, len(candidates), BATCH_SIZE)]

    logger.info(f"Splitting {len(candidates)} candidates into {len(segments)} batches (size={BATCH_SIZE}).")

    # 3. 并发调用 LLM (Async Gather)
    tasks = []
    for i, seg in enumerate(segments):
        tasks.append(call_segment_async(seg, reason, topic, i))

    # --- 关键修改：并发等待所有结果 ---
    results_lists = await asyncio.gather(*tasks)

    # 4. 收集结果
    norm_to_original = {normalize_candidate(t): t for t in candidates if normalize_candidate(t)}
    chosen_norms = set()  # 使用 set 加速查找
    chosen_norms_list = []  # 保持顺序

    for seg_res in results_lists:
        if not seg_res: continue
        for item in seg_res:
            nk = normalize_candidate(item)
            if nk and nk in norm_to_original:
                if nk not in chosen_norms:
                    chosen_norms.add(nk)
                    chosen_norms_list.append(nk)

    # 5. Fallback 逻辑 (如果 LLM 没选出什么词)
    if not chosen_norms_list:
        logger.warning("LLM did not select any terms, applying heuristic fallback.")
        scored = []
        text_lower = sd.get("summary", "").lower()
        for t in candidates:
            # 简单的启发式打分：单词长度 * 0.1 + 是否在摘要中出现
            score = len(t.split()) * 0.1
            idx = text_lower.find(t.lower())
            if idx >= 0:
                score += max(0, 1 - idx / max(1, len(text_lower)))
            scored.append((score, t))
        scored.sort(key=lambda x: -x[0])

        # Fallback 策略：取前 50% 或至少 100 个
        fallback_limit = max(100, int(len(candidates) * 0.5))
        selected_terms = [t for _, t in scored[:fallback_limit]]
    else:
        selected_terms = [
            norm_to_original[nk]
            for nk in chosen_norms_list
            if nk in norm_to_original
        ]

    # 6. 再次过滤 (排除 remove_list)
    final_terms = []
    for t in selected_terms:
        if normalize_candidate(t) not in remove_norms:
            final_terms.append(t)

    print(f"Selected terms count after filtering: {len(final_terms)}")

    # 7. 分发回 Chunk (CPU计算)
    # 这里需要将选中的词映射回它们来源的 chunk_id
    chunk_terms_map = defaultdict(list)
    for term in final_terms:
        # 一个词可能出现在多个 chunk 中
        for cid in term_to_chunks.get(term, []):
            chunk_terms_map[cid].append(term)

    # 转换为 List[Dict] 格式
    chunk_terms_list = [
        {"chunk_id": str(cid), "terms": terms}
        for cid, terms in chunk_terms_map.items()
    ]

    print(f"Global selected terms count: {len(final_terms)}")
    # print(f"Chunk terms distribution: {chunk_terms_list}") # 日志可能太长，按需开启

    return typing.cast(TermState, {
        "selected_terms": final_terms,
        "chunk_terms": chunk_terms_list,
    })