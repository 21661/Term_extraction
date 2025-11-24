import json
import logging
import asyncio
import typing
from typing import List, Dict, Any
from Nodes.select_top_terms import _pack_result

from utils.LLMClientManager import LLMclientManager
from utils.TermState import TermState
from utils.TimeNode import timed_node
from utils.workflow_adapter import _unwrap
from utils.candidate_tool import _LLM_RETRIES, _RETRY_BACKOFF, normalize_candidate

logger = logging.getLogger(__name__)
BATCH_SIZE = 100
async def process_batch(batch_terms: List[str], topic: str, batch_index: int,prompt:str) -> Dict[str, Any]:
    """
    异步处理单个批次，包含重试逻辑
    """
    raw = None
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            # 使用异步接口 achat
            # 建议开启 reasoning=False 以获得更快的速度，除非你需要强推理
            completion = await LLMclientManager.achat(
                messages=[
                    {"role": "system", "content": "你必须返回纯JSON"},
                    {"role": "user", "content": prompt},
                ],

            )

            raw = completion.content

            clean_raw = raw.strip()
            if clean_raw.startswith("```json"):
                clean_raw = clean_raw[7:]
            if clean_raw.startswith("```"):
                clean_raw = clean_raw[3:]
            if clean_raw.endswith("```"):
                clean_raw = clean_raw[:-3]

            parsed = json.loads(clean_raw)

            # 简单的结构验证
            if "pass" not in parsed:
                parsed["pass"] = False
            if "remove_terms" not in parsed:
                parsed["remove_terms"] = []

            return parsed  # 成功返回

        except Exception as e:
            logger.warning(f"[Batch-{batch_index}] Attempt {attempt} failed: {e}")
            await asyncio.sleep(_RETRY_BACKOFF * attempt)

    # 如果所有重试都失败，为了不阻塞流程，默认这一批次不通过但也不删除任何词（或者你可以选择删除所有词以求稳）
    logger.error(f"[Batch-{batch_index}] Failed after retries. Raw: {raw}")
    return {"pass": False, "reason": "LLM processing failed", "remove_terms": []}
def build_remove_prompt_batch(topic: str, batch_terms: List[str]) -> str:
    safe_batch_terms = json.dumps(batch_terms, ensure_ascii=False)
    # 提示词保持你原有的逻辑
    prompt = "\n".join([
        f"""
        你现在是术语筛选审查器。你的任务是遵循筛选标准筛选要筛选的词汇表。你必须只返回 JSON，不允许任何注释、解释、自然语言、前缀、后缀。

        【主题】：{topic}
        【要筛选的词汇表】：{safe_batch_terms}

        筛选标准：
        1. 不可以是人名或地名
        2. 无意义的人称代词连词或者和主题无关的初级词汇（如"you","box"）
        3. 不需要翻译的词语（如“ppt","json")
        4. 拼写错误极大以至于无法正常翻译的词语（可以被识别出的微小错误可以忽略）

        - "reason": 对每一个词语给出简洁明确的理由，无论是否认为通过筛选。
        - "remove_terms": 所有需要删除的词语

        参照以下 JSON 格式返回（注意必须使用双引号）：
        """,
        "{",
        "  \"reason\": \"xxx\",",
        "  \"remove_terms\": []",
        "}",
    ])
    return prompt

@timed_node()
async def remove_sync_node(state: TermState) -> TermState:
    """
    异步极速移除节点。
    """
    inner, parent, key = _unwrap(state)
    sd: TermState = typing.cast(TermState, inner if isinstance(inner, dict) else TermState())

    # 获取数据
    topic = sd.get("summary", "")
    target_terms = sd.get("selected_terms", []) or []
    term_to_chunks = sd.get("term_to_chunks", {})

    logger.info(f"[remove] Start check. Count: {len(target_terms)}")

    if not target_terms:
        return _pack_result([], term_to_chunks)

    # 1. 分批 (加大 Batch)
    chunks = [target_terms[i:i + BATCH_SIZE] for i in range(0, len(target_terms), BATCH_SIZE)]

    # 2. 异步并发执行 (Gather)
    # 直接调用 process_batch_negative_selection
    tasks = []
    for i, chunk in enumerate(chunks):
        tasks.append(process_batch(chunk, topic, i,build_remove_prompt_batch(topic, chunk)))

    # 等待结果 (List[List[str]])
    results = await asyncio.gather(*tasks)

    # 3. 聚合要删除的词
    aggregated_remove_terms = []
    for res in results:
        if res:
            aggregated_remove_terms.extend(res)

    # 4. 执行过滤
    if not aggregated_remove_terms:
        logger.info("[remove] No terms removed. Perfect!")
        return _pack_result(target_terms, term_to_chunks)

    remove_norms = {normalize_candidate(t) for t in aggregated_remove_terms if t}
    cleaned_selected = []
    removed_count = 0

    for t in target_terms:
        norm = normalize_candidate(t)
        if norm and norm not in remove_norms:
            cleaned_selected.append(t)
        else:
            removed_count += 1

    logger.info(f"[remove] Done. {len(target_terms)} -> {len(cleaned_selected)} (Removed: {removed_count})")

    return _pack_result(cleaned_selected, term_to_chunks)