import json
import logging
import asyncio
import typing
from typing import List, Dict, Any

# 假设你的项目中有这些导入
from utils.LLMClientManager import LLMclientManager
from utils.TermState import TermState
from utils.TimeNode import timed_node
from utils.workflow_adapter import _unwrap
from utils.candidate_tool import _LLM_RETRIES, _RETRY_BACKOFF

logger = logging.getLogger(__name__)

# 配置参数
BATCH_SIZE = 60  # 每批处理30个词，平衡并发数和Token长度


# ------------------------
# 构建 prompt (保持不变，逻辑复用)
# ------------------------
def build_reflect_prompt_batch(topic: str, batch_terms: List[str]) -> str:
    safe_batch_terms = json.dumps(batch_terms, ensure_ascii=False)
    # 提示词保持你原有的逻辑
    prompt = "\n".join([
        f"""
    你现在是术语筛选审查器。你的任务是遵循筛选标准筛选要筛选的词汇表。你必须只返回 JSON，不允许任何注释、解释、自然语言、前缀、后缀。
    请谨慎的筛选词语，只排除错误明显的词语。
    remove_terms 中违反标准1的词语应排在最前面，依此类推。

    【主题】：{topic}
    【要筛选的词汇表】：{safe_batch_terms}

    筛选标准：
    1. 不可以是人名或地名
    2. 无意义的人称代词连词或者和主题无关的初级词汇（如"you","box"）
    3. 不需要翻译的词语（如“ppt","json")
    4. 拼写错误极大以至于无法正常翻译的词语（可以被识别出的微小错误可以忽略）

    - "pass": 如果所有词语都合格，则为 true，否则为 false
    - "reason": 对每一个词语给出简洁明确的理由，无论是否认为通过筛选,reason返回形式应该为字符串
    - "remove_terms": 所有需要删除的词语

    参照以下 JSON 格式返回（注意必须使用双引号）：
    """,
        "{",
        "    \"pass\": true,",
        "  \"reason\": \"xxx\",",
        "  \"remove_terms\": []",
        "}",
    ])
    return prompt


# ------------------------
# 异步单个批次处理
# ------------------------
async def process_batch(batch_terms: List[str], topic: str, batch_index: int) -> Dict[str, Any]:
    """
    异步处理单个批次，包含重试逻辑
    """
    raw = None
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            prompt = build_reflect_prompt_batch(topic, batch_terms)

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


# ------------------------
# 优化的 Reflect 节点 (Async)
# ------------------------
@timed_node()
async def reflect_sync_node(state: TermState, maxRetry: int = 1) -> TermState:
    """
    异步版本的反思节点。
    并发执行检查，最后一次性写入 State。
    """
    # 1. 解包 State
    inner, parent, key = _unwrap(state)
    sd: TermState = typing.cast(TermState, inner if isinstance(inner, dict) else TermState())

    # 2. 检查重试次数
    reflect_attempts = int(sd.get("reflect_attempts", 0) or 0) + 1
    if reflect_attempts > maxRetry:
        logger.info("[reflect] Max retries reached, skipping check.")
        return typing.cast(TermState, {
            "reflect_attempts": reflect_attempts,
            "reflected": True,  # 强制通过，避免无限循环
            "reflect_reason": ["max retry pass"],
            "reflect_remove_terms": [],
        })

    # 3. 获取数据
    topic = sd.get("summary", "")
    # 注意：根据你的逻辑，第一次可能检查 selected_terms，后续可能需要检查其他字段？
    # 这里假设始终检查 selected_terms
    target_terms = sd.get("selected_terms", [])

    logger.info(f"[reflect] Start concurrent check. Terms count: {len(target_terms)}, Attempt: {reflect_attempts}")

    if not target_terms:
        return typing.cast(TermState, {
            "reflect_attempts": reflect_attempts,
            "reflected": True,
            "reflect_reason": ["no terms"],
            "reflect_remove_terms": [],
        })

    # 4. 准备分批并发任务
    tasks = []
    # 将列表切片
    chunks = [target_terms[i:i + BATCH_SIZE] for i in range(0, len(target_terms), BATCH_SIZE)]

    for i, chunk in enumerate(chunks):
        tasks.append(process_batch(chunk, topic, i))

    # 5. 并发执行并等待所有结果
    # asyncio.gather 会并发运行所有任务
    results = await asyncio.gather(*tasks)

    # 6. 聚合结果 (Aggregation)
    # 只要有一个批次不通过 (pass=False)，整体就不通过
    # 将所有批次发现的 remove_terms 合并

    final_pass = True
    aggregated_reasons = []
    aggregated_remove_terms = []

    for res in results:
        if not res.get("pass", True):
            final_pass = False

        # 收集理由
        r_reason = res.get("reason", "")
        if r_reason and r_reason != "无":
            if isinstance(r_reason, list):
                aggregated_reasons.extend(r_reason)
            else:
                aggregated_reasons.append(r_reason)

        # 收集要删除的词
        r_remove = res.get("remove_terms", [])
        if r_remove:
            aggregated_remove_terms.extend(r_remove)

    # 去重 remove_terms
    aggregated_remove_terms = list(set(aggregated_remove_terms))

    logger.info(f"[reflect] Done. Pass: {final_pass}, Terms to remove: {len(aggregated_remove_terms)}")

    # 7. 返回最终的一次性 State 更新
    # 这样不会在节点内多次写入 State，符合你的要求
    return typing.cast(TermState, {
        "reflect_attempts": reflect_attempts,
        "reflected": final_pass,
        "reflect_reason": "; ".join(aggregated_reasons)[:500],  # 截断一下避免日志太长
        "reflect_remove_terms": aggregated_remove_terms,
    })


# 路由函数保持不变，它只读取状态
def route_after_reflect(state: TermState) -> str:
    attempts = int(state.get("reflect_attempts", 0) or 0)
    is_pass = bool(state.get("reflected", False))

    if not is_pass:
        # 如果没通过，且还有机会重试
        if attempts < 2:
            logger.info("[reflect-route] FAIL -> retry select_top_terms")
            return "retry"
        else:
            logger.info("[reflect-route] FAIL but max attempts used -> proceed")
            return "proceed"

    logger.info("[reflect-route] PASS -> proceed")
    return "proceed"