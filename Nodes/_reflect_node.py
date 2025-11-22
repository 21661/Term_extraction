import json
import logging
import time
import typing
from typing import List, Dict, Any
from utils.LLMClientManager import LLMclientManager
from utils.TermState import TermState
from utils.TimeNode import timed_node
from utils.workflow_adapter import _unwrap
from utils.candidate_tool import _LLM_RETRIES, _RETRY_BACKOFF

logger = logging.getLogger(__name__)



# ------------------------
# 构建 prompt（安全转义）
# ------------------------
def build_reflect_prompt_batch(topic: str, batch_terms: List[str]) -> str:
    safe_batch_terms = json.dumps(batch_terms, ensure_ascii=False)
    print("safe_batch_terms",safe_batch_terms)
    prompt = "\n".join([f"""
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
    logger.warning(f"prompt: {prompt}")
    return prompt


# ------------------------
# 调用 LLM
# ------------------------
def call_segment(selected_terms: List[str],  topic: str) -> Dict[str, Any]:
    raw = None
    parsed = None
    for attempt in range(1, _LLM_RETRIES + 1):
        try:
            prompt = build_reflect_prompt_batch(topic, selected_terms)
            completion = LLMclientManager.chat(
                messages=[
                    {"role": "system", "content": "你必须返回纯JSON"},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = completion.content
            print(f"__________raw{raw}")
            if not completion:
                logger.info("No completion received from LLM.")
            if not raw:
                time.sleep(_RETRY_BACKOFF * attempt)
                continue

            parsed = json.loads(raw)
            return parsed  # 成功返回
        except Exception as e:
            print("__________exception:", e)
            logger.debug("LLM segment failed attempt %d: %s", attempt, e)
            if raw:
                logger.debug("LLM raw output: %s", raw)
            time.sleep(_RETRY_BACKOFF * attempt)

    # 全部失败返回 fallback
    return parsed if parsed else {"pass": False, "reason": "LLM JSON 无法解析", "remove_terms": []}

# ------------------------
# Reflect 节点
# ------------------------
@timed_node()
def reflect_sync_node(state: TermState, maxRetry: int = 1) -> TermState:
    inner, parent, key = _unwrap(state)
    sd: TermState = typing.cast(TermState, inner if isinstance(inner, dict) else TermState())
    reflect_attempts = int(sd.get("reflect_attempts", 0) or 0) + 1

    if reflect_attempts > maxRetry:
        return typing.cast(TermState,{
            "reflect_attempts": reflect_attempts,
            "reflected": True,
            "reflect_reason": ["max retry pass"],
            "reflect_remove_terms": [],
        })

    topic = sd.get("summary", "")
    candidates = sd.get("candidates", [])
    selected_terms = sd.get("selected_terms", [])
    logger.warning("[reflect] topic=%s, candidates=%s, selected_terms=%s", topic, candidates, selected_terms)

    if not candidates and not selected_terms:
        return typing.cast(TermState,{
            "reflect_attempts": reflect_attempts,
            "reflected": True,
            "reflect_reason": ["no candidates"],
            "reflect_remove_terms": [],
        })

    parsed = call_segment(selected_terms, topic)

    pass_flag = bool(parsed.get("pass", False))
    reason = parsed.get("reason", "无")
    if(isinstance(reason, list)):
        reason = "; ".join(reason)
    remove_terms = parsed.get("remove_terms", [])
    print(pass_flag, reason, remove_terms)
    return typing.cast(TermState,
                       {"reflect_attempts": reflect_attempts, "reflected": pass_flag, "reflect_reason": reason,
                        "reflect_remove_terms": remove_terms, })

# ------------------------
# 路由逻辑
# ------------------------
def route_after_reflect(state: TermState) -> str:
    attempts = int(state.get("reflect_attempts", 0) or 0)
    is_pass = bool(state.get("reflected", False))
    if not is_pass:
        if attempts < 2:
            logger.info("[reflect-route] FAIL → retry select_top_terms (attempts=%d)", attempts)
            return "retry"
        else:
            logger.info("[reflect-route] FAIL but max attempts used → proceed to aggregate")
            return "proceed"
    logger.info("[reflect-route] PASS → proceed to aggregate")
    return "proceed"
