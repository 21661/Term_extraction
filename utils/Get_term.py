# utils/Get_term.py
import json
import re
import logging
import asyncio
from typing import List, Dict, Optional, Any

# 假设 DB 查询很快，如果是远程数据库，建议也改为异步
from utils.db_interface import query_term_translation
from utils.LLMClientManager import LLMclientManager

logger = logging.getLogger(__name__)


# ==============================================================================
# 基础工具函数
# ==============================================================================

def _clean_text(text: str) -> str:
    """清洗 LLM 返回的 Markdown 标记"""
    if not text: return ""
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return text.strip()


def _parse_json_safe(text: str, default_type=list) -> Any:
    """安全的 JSON 解析"""
    try:
        return json.loads(_clean_text(text))
    except json.JSONDecodeError:
        return default_type()


# ==============================================================================
# 核心异步函数
# ==============================================================================

async def translate_term_async(term: str, topic: str = "", model: str = "tencent/Hunyuan-MT-7B") -> List[str]:
    """
    【单词高并发模式】
    优先查本地库，没有则调用 LLM (MT模型)。
    """
    # 1. 查本地库 (同步操作，速度快)
    local = query_term_translation(term)
    if local:
        return list(local)

    # 2. 构造极简 Prompt
    prompt = f"""Target: Translate term to Chinese.
Context: {topic}
Term: "{term}"
Output: JSON list of translations only. No explanations."""

    # 3. 异步调用 LLM
    try:
        # ✅ 修改点：删除了 reasoning=False 参数
        response = await LLMclientManager.achat(
            messages=[
                {"role": "system", "content": "You are a translator. Return JSON list."},
                {"role": "user", "content": prompt}
            ],
            model=model,  # 指定 MT 模型
        )

        parsed = _parse_json_safe(response.content, list)
        if isinstance(parsed, list):
            # 简单的后处理，去重
            return list(set([str(p) for p in parsed if p]))

    except Exception as e:
        logger.warning(f"Async translate '{term}' failed: {e}")

    return []


async def translate_batch_async(terms: List[str], topic: str = "") -> Dict[str, List[str]]:
    """
    【批量并发模式】
    优先查本地库，剩余的打包发给 LLM (通用模型)。
    """
    if not terms:
        return {}

    results: Dict[str, List[str]] = {}
    missing_terms: List[str] = []

    # 1. 先查本地库
    for term in terms:
        local = query_term_translation(term)
        if local:
            results[term] = list(local)
        else:
            missing_terms.append(term)

    if not missing_terms:
        return results

    # 2. 构造批量 Prompt
    safe_terms = json.dumps(missing_terms, ensure_ascii=False)
    prompt = f"""Context: {topic}
Source Terms: {safe_terms}
Task: Translate terms to Chinese.
Format: JSON object {{ "term": ["trans1", "trans2"] }}."""

    # 3. 异步调用 LLM
    try:
        response = await LLMclientManager.achat(
            messages=[
                {"role": "system", "content": "Return JSON object only."},
                {"role": "user", "content": prompt}
            ],
        )

        parsed = _parse_json_safe(response.content, dict)

        # 4. 合并结果
        if isinstance(parsed, dict):
            for t, trans_list in parsed.items():
                if isinstance(trans_list, list):
                    results[t] = trans_list
                elif isinstance(trans_list, str):
                    results[t] = [trans_list]

    except Exception as e:
        logger.warning(f"Async batch translate failed: {e}")

    # 5. 兜底：没查到的填空列表
    for t in missing_terms:
        if t not in results:
            results[t] = []

    return results