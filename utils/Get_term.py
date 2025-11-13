# utils/Get_term.py
# CLI helper for querying/translating terms (uses utils package imports)
from typing import List, Dict
from utils.db_interface import query_term_translation, save_translation
#from utils.web_fetcher import fetch_core_translations
from utils.zhipu_client import client
import logging
import re
import time

logger = logging.getLogger(__name__)

def select_candidate(candidates):
    """人工选择候选翻译"""
    if not candidates:
        return None
    print("抓取到候选翻译：")
    for idx, c in enumerate(candidates, 1):
        print(f"{idx}. {c}")
    choice = input("请选择翻译序号(默认第一个)或直接回车: ")
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    except Exception:
        pass
    return candidates[0]

def translate_term_Dict(term_en):
    # 单DIct翻译
    # 1. 本地库查询
    local = query_term_translation(term_en)
    if local:
        print(f"本地库找到翻译：{local}")
        return local
    else:
        print("本地库未找到翻译")

    # # 2. 网络抓取
    # candidates = fetch_core_translations(term_en)
    # if not candidates:
    #     print("网络抓取未找到翻译")
    #     return []
    #
    # # 3. 人工确认
    # term_cn = select_candidate(candidates)
    # if term_cn:
    #     save_translation(term_en, term_cn)
    #     print(f"已保存翻译：{term_cn}")
    #     return [term_cn]

    return []

def translate_terms_Dict(terms: List[str]) -> Dict[str, List[str]]:
    result = {}
    for term in terms:
        result[term] = translate_term_Dict(term)
    return result

def _normalize_term(term: str) -> str:
    return term.strip().lower()

def _decide_translation_method(term: str) -> str:
    """自动判断使用字典还是 LLM"""
    term_norm = _normalize_term(term)

    if len(term_norm) <= 2:
        return "dict"
    if len(term.split()) > 1 or re.search(r"[-_/]", term):
        return "llm"
    return "dict"

def _fetch_llm_translation(term: str, retries=2, backoff=1.0) -> List[str]:
    """使用 LLM 获取翻译候选"""
    prompt = f'请将下列英文学术/技术术语翻译成中文，保持专业性：\n"{term}"\n请返回中文列表，用逗号或换行分隔。'
    for attempt in range(1, retries + 1):
        try:
            completion = client.chat.completions.create(
                model="glm-4-FlashX-250414",
                messages=[
                    {"role": "system", "content": "你是专业术语翻译助手"},
                    {"role": "user", "content": prompt},
                ],
                # thinking={
                #     "type": "disabled",
                # },
                temperature=0.2,
            )

            raw_text = _extract_completion_text(completion)
            if raw_text:
                candidates = [t.strip() for t in re.split(r"[,\n，、；;]", raw_text) if t.strip()]
                return candidates

        except Exception as e:
            logger.warning("LLM 翻译失败 (attempt %d) for term %s: %s", attempt, term, e)
            time.sleep(backoff * attempt)
    return []

def _fetch_llm_translation_batch(terms: list[str], retries=2, backoff=1.0) -> dict[str, list[str]]:
    """
    批量调用 LLM 翻译多个术语
    返回 dict: {term: [候选翻译列表]}
    """
    if not terms:
        return {}

    prompt = "请将下列英文学术/技术术语翻译成中文，保持专业性，每个术语给出一个或多个翻译候选，用换行分隔，格式为 term: 翻译1, 翻译2\n\n"
    prompt += "\n".join(terms)

    for attempt in range(1, retries + 1):
        try:
            completion = client.chat.completions.create(
                model="glm-4-FlashX-250414",
                messages=[
                    {"role": "system", "content": "你是专业术语翻译助手"},
                    {"role": "user", "content": prompt},
                ],
                # thinking={
                #     "type": "disabled",
                # },
                temperature=0.2,
            )
            raw_text = _extract_completion_text(completion)
            if not raw_text:
                continue

            result: dict[str, list[str]] = {}
            for line in raw_text.splitlines():
                if ":" in line:
                    term, translations = line.split(":", 1)
                    term = term.strip()
                    candidates = [t.strip() for t in re.split(r"[,\n，、；;]", translations) if t.strip()]
                    result[term] = candidates
            return result
        except Exception as e:
            logger.warning("LLM 批量翻译失败 (attempt %d): %s", attempt, e)
            time.sleep(backoff * attempt)
    return {term: [] for term in terms}

def _get_translation_candidates(term: str) -> List[str]:
    # 完善的翻译模块
    # 1️⃣ 尝试本地数据库
    local = query_term_translation(term)
    if local:
        logger.info("本地库找到翻译: %s -> %s", term, local)
        return local

    # 2️⃣ 根据方法选择
    method = _decide_translation_method(term)
    candidates = []

    candidates = _fetch_llm_translation(term)
    # 去重
    seen = set()
    cleaned: List[str] = []
    for c in candidates:
        c_clean = c.strip()
        if c_clean and c_clean not in seen:
            seen.add(c_clean)
            cleaned.append(c_clean)
    return cleaned

def get_translation_candidates_batch(terms: list[str], batch_size: int = 50) -> dict[str, list[str]]:
    """
    批量获取翻译候选（带分批机制）。
    1️⃣ 先查本地库
    2️⃣ 收集需要 LLM 批量翻译的词
    3️⃣ 分批调用 LLM，避免请求过大
    """
    result: dict[str, list[str]] = {}
    llm_terms: list[str] = []

    # Step 1: 查询本地翻译
    for term in terms:
        local = query_term_translation(term)
        if local:
            result[term] = local
        else:
            llm_terms.append(term)

    # Step 2: 分批调用 LLM
    if llm_terms:
        for i in range(0, len(llm_terms), batch_size):
            batch = llm_terms[i:i + batch_size]
            llm_results = _fetch_llm_translation_batch(batch)
            for term, candidates in llm_results.items():
                result[term] = candidates

    # Step 3: 清理去重
    for term, candidates in result.items():
        cleaned: list[str] = []
        seen: set[str] = set()
        for c in candidates:
            c_clean = c.strip()
            if c_clean and c_clean not in seen:
                seen.add(c_clean)
                cleaned.append(c_clean)
        result[term] = cleaned

    return result

# ------------------ 单术语翻译 ------------------
def translate_term(term_en: str) -> List[str]:
    candidates = _get_translation_candidates(term_en)
    if not candidates:
        print(f"最终未找到 {term_en} 的翻译")
        return []
    return candidates

def _extract_completion_text(completion) -> str:
    """从不同 shape 的 completion 中抽取文本内容"""
    try:
        if isinstance(completion, dict):
            def _find_text(obj):
                if isinstance(obj, dict):
                    for k in ("content", "text", "message"):
                        if k in obj and isinstance(obj[k], str) and obj[k].strip():
                            return obj[k].strip()
                    for v in obj.values():
                        res = _find_text(v)
                        if res:
                            return res
                elif isinstance(obj, list):
                    for item in obj:
                        res = _find_text(item)
                        if res:
                            return res
                return None

            res = _find_text(completion)
            return res or ""

        choices = getattr(completion, 'choices', None)
        if choices and len(choices) > 0:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get('message') or first.get('text') or {}
                if isinstance(msg, dict):
                    return (msg.get('content') or msg.get('text') or "") .strip()
                return str(msg).strip()

            msg = getattr(first, 'message', None) or getattr(first, 'text', None) or None
            if isinstance(msg, dict):
                return (msg.get('content') or msg.get('text') or "").strip()
            if msg is not None:
                try:
                    s = str(getattr(msg, 'content', None) or getattr(msg, 'text', None)).strip()
                    if s:
                        return s
                except Exception:
                    pass

            for attr in ('content', 'text', 'delta'):
                v = getattr(first, attr, None)
                if isinstance(v, str) and v.strip():
                    return v.strip()
                if v is not None:
                    try:
                        s = str(v).strip()
                        if s:
                            return s
                    except Exception:
                        pass

        top = getattr(completion, 'text', None)
        if isinstance(top, str) and top.strip():
            return top.strip()

        return ""
    except Exception as err:
        logger.debug("_extract_completion_text error: %s", err)
        try:
            return str(completion)
        except Exception:
            return ""

