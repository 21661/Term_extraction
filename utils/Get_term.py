# utils/Get_term.py
# CLI helper for querying/translating terms (uses utils package imports)
import json
from typing import List, Dict, Optional
from utils.db_interface import query_term_translation
# from utils.web_fetcher import fetch_core_translations
from utils.LLMClientManager import LLMclientManager
import logging
import re
import time

logger = logging.getLogger(__name__)

def translate_term_Dict(term_en):
    # 单DIct翻译
    # 1. 本地库查询
    local = query_term_translation(term_en)
    if local:
        print(f"本地库找到翻译：{local}")
        return local
    else:
        print("本地库未找到翻译")
    return []

def translate_terms_Dict(terms: List[str]) -> Dict[str, List[str]]:
    result = {}
    for term in terms:
        result[term] = translate_term_Dict(term)
    return result

def _normalize_term(term: str) -> str:
    return term.strip().lower()

def _decide_translation_method(term: str) -> str:
    term_norm = _normalize_term(term)
    if len(term_norm) <= 2:
        return "dict"
    if len(term.split()) > 1 or re.search(r"[-_\/]", term):
        return "llm"
    return "dict"


# ---------------- LLM 调用：按 topic 返回单一最佳释义 ----------------

def _fetch_llm_translation(term: str, topic: Optional[str] = None, retries: int = 4, backoff: float = 1.0) -> List[str]:
    """
    使用 LangChain 的 ChatOpenAI 获取术语翻译。
    """
    sys_prompt = "把下面术语翻译成中文，仅返回中文翻译。如果得不到翻译，请返回None。"
    user_prompt = f"{term}\n"

    for attempt in range(1, retries + 1):
        try:
            # ⭐ 不再 get_client_by_model
            resp = LLMclientManager.chat(   # 自动根据 model 匹配配置
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="tencent/Hunyuan-MT-7B",
            )

            # ⭐ LangChain 的返回结构
            raw_text = (resp.content or "").strip()
            if not raw_text:
                continue

            # 去除 "term:翻译"
            if ":" in raw_text:
                raw_text = raw_text.split(":", 1)[-1].strip()

            parts = [t.strip() for t in re.split(r"[\n,，、;；]", raw_text) if t.strip()]
            if not parts and raw_text:
                parts = [raw_text]

            return [parts[0]] if parts else []

        except Exception as e:
            logger.warning(
                "LLM 翻译失败 (attempt %d) for term %s: %s",
                attempt, term, e
            )
            time.sleep(backoff * attempt)

    return []


def _fetch_llm_translation_batch(terms: List[str], topic: Optional[str] = None, retries: int = 2, backoff: float = 1.0) -> Dict[str, List[str]]:
    if not terms:
        return {}

    sys_prompt = (
        "你是一个专业的中英文翻译助手。"
        "你必须返回一个 JSON 对象，其中的键是原始英文术语，值是对应的中文翻译。"
    )

    input_data = {"terms": terms}
    Input_json = json.dumps(input_data, ensure_ascii=False)
    user_prompt = f"""
        翻译对象：“{Input_json}”
        任务描述：
        "请将翻译对象列表中的每个术语翻译成最贴切的中文。"
        "返回一个 JSON 对象，其中的键是原始英文术语，值是对应的中文翻译。"
        """

    for attempt in range(1, retries + 1):
        try:
            completion = LLMclientManager.chat(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

            raw_text = (completion.content or "").strip()
            if not raw_text:
                continue

            try:
                json_match = re.search(r"```json\n({.*})\n```", raw_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = raw_text

                parsed_json = json.loads(json_str)
                result: Dict[str, List[str]] = {}

                for term, translation in parsed_json.items():
                    if isinstance(translation, str) and translation.strip():
                        result[term] = [translation.strip()]

                for t in terms:
                    result.setdefault(t, [])
                return result
            except json.JSONDecodeError:
                logger.warning("LLM 批量翻译返回的 JSON 无效: %s", raw_text)
                continue

        except Exception as e:
            logger.warning("LLM 批量翻译失败 attempt %d: %s", attempt, e)
            time.sleep(backoff * attempt)

    return {t: [] for t in terms}

def _clean_translation_text(text: str) -> str:
    return re.sub(r'[\(（][^)）]*[\)）]', '', text).strip()

def _get_translation_candidates(term: str, topic: Optional[str] = None) -> List[str]:
    # 1️⃣ 本地库
    local = query_term_translation(term)
    if local:
        logger.info("本地库找到翻译: %s -> %s", term, local)
        return [_clean_translation_text(t) for t in local]

    # 2️⃣ LLM
    candidates = _fetch_llm_translation(term, topic=topic)

    seen = set()
    cleaned: List[str] = []
    for c in candidates:
        c_clean = _clean_translation_text(c.strip())
        if c_clean and c_clean not in seen:
            seen.add(c_clean)
            cleaned.append(c_clean)

    return cleaned

def get_translation_candidates_batch(terms: List[str], batch_size: int = 50, topic: Optional[str] = None) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    llm_terms: List[str] = []

    # Step 1: 本地库
    for term in terms:
        local = query_term_translation(term)
        if local:
            result[term] = [local[0]]
        else:
            llm_terms.append(term)

    # Step 2: LLM
    if llm_terms:
        for i in range(0, len(llm_terms), batch_size):
            batch = llm_terms[i:i + batch_size]
            llm_results = _fetch_llm_translation_batch(batch, topic=topic)
            for term, candidates in llm_results.items():
                result[term] = candidates or []

    # Step 3: fallback → 原文
    for term, candidates in result.items():
        cleaned = []
        seen = set()
        for c in (candidates or []):
            c2 = c.strip()
            if c2 and c2 not in seen:
                seen.add(c2)
                cleaned.append(c2)

        if cleaned:
            result[term] = cleaned[:1]
        else:
            result[term] = [term]

    return result

def translate_term(term_en: str, topic: Optional[str] = None) -> List[str]:
    """
    翻译单个词。
    1. 本地库查不到 → 必须继续查 LLM
    2. LLM 也查不到 → 返回英文原文
    """
    # 本地库
    local = query_term_translation(term_en)
    if local:
        return [local[0]]
    # LLM
    candidates = _get_translation_candidates(term_en, topic=topic)
    # fallback
    if not candidates:
        logger.info("未找到 %s 的任何翻译，返回英文原文。", term_en)
        return [term_en]

    return candidates[:1]
