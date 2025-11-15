# utils/Get_term.py
# CLI helper for querying/translating terms (uses utils package imports)
import json
from typing import List, Dict, Optional
from utils.db_interface import query_term_translation
# from utils.web_fetcher import fetch_core_translations
from utils.zhipu_client import client,DJ_client
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
    """
    自动判断使用字典还是 LLM
    """
    term_norm = _normalize_term(term)

    if len(term_norm) <= 2:
        return "dict"
    if len(term.split()) > 1 or re.search(r"[-_\/]", term):
        return "llm"
    return "dict"


def _extract_completion_text(completion) -> str:
    """
    从不同 shape 的 completion 中抽取文本内容
    """
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


# ---------------- LLM 调用：按 topic 返回单一最佳释义 ----------------

def _fetch_llm_translation(term: str, topic: Optional[str] = None, retries: int = 2, backoff: float = 1.0) -> List[str]:
    """
    使用 LLM 获取单个术语的“最佳且唯一”中文翻译。
    会结合 topic 作为上下文，强制要求只返回一个最贴切的中文词/短语。
    返回: 单元素列表以兼容旧接口，若失败则返回空列表。
    """

    sys_prompt = "把下面术语翻译成中文，仅返回中文翻译。如果得不到翻译，请返回None。"
    user_prompt = (
        f"{term}\n"
    )

    for attempt in range(1, retries + 1):
        try:
            completion = DJ_client.chat.completions.create(
                model="tencent/Hunyuan-MT-7B",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            raw_text = _extract_completion_text(completion).strip()
            if not raw_text:
                continue
            # 规范化：若模型误返回 "term: 翻译" 或包含多项，用分隔符切分取首项
            if ":" in raw_text:
                raw_text = raw_text.split(":", 1)[-1].strip()
            parts = [t.strip() for t in re.split(r"[\n,，、;；]", raw_text) if t.strip()]
            if not parts and raw_text:
                parts = [raw_text]
            # 仅返回一个
            return [parts[0]] if parts else []
        except Exception as e:
            logger.warning("LLM 翻译失败 (attempt %d) for term %s: %s", attempt, term, e)
            time.sleep(backoff * attempt)
    return []


def _fetch_llm_translation_batch(terms: List[str], topic: Optional[str] = None, retries: int = 2, backoff: float = 1.0) -> Dict[str, List[str]]:
    """
    批量调用 LLM 翻译多个词语。使用 JSON 输入和输出来提高稳定性。
    结合 topic，以“每个词只返回一个最贴切的译名”。
    返回: {term: [唯一译名]}，失败则为空列表。
    """
    if not terms:
        return {}

    sys_prompt = (
        "你会收到一个包含'terms'字段的 JSON 对象。"
        "请将'terms'列表中的每个术语翻译成最贴切的中文。"
        "你必须返回一个 JSON 对象，其中的键是原始的英文术语，值是对应的中文翻译。"
        "如果某个术语无法翻译，请在返回的 JSON 中省略该术语。"
        "示例输入: {\"terms\": [\"Transformer encoder\", \"ResNet\"]}"
        "示例输出: {\"Transformer encoder\": \"变换器编码器\", \"ResNet\": \"残差网络\"}"
    )

    input_data = {
        "terms": terms
    }
    user_prompt = json.dumps(input_data, ensure_ascii=False)

    for attempt in range(1, retries + 1):
        try:
            completion = DJ_client.chat.completions.create(
                model="tencent/Hunyuan-MT-7B",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                # 确保模型开启 JSON 模式（如果 API 支持）
                # response_format={"type": "json_object"},
            )
            raw_text = _extract_completion_text(completion)
            if not raw_text:
                continue

            # 尝试从文本中提取 JSON
            try:
                # 模型可能返回被 markdown 包裹的 JSON
                json_match = re.search(r"```json\n({.*})\n```", raw_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = raw_text

                parsed_json = json.loads(json_str)

                result: Dict[str, List[str]] = {}
                # 将模型的 {term: "翻译"} 格式转换为 {term: ["翻译"]}
                for term, translation in parsed_json.items():
                    if isinstance(translation, str) and translation.strip():
                        result[term] = [translation.strip()]

                # 为所有输入的 term 保证有 key，未翻译的则为空列表
                for t in terms:
                    result.setdefault(t, [])
                return result

            except json.JSONDecodeError as json_e:
                logger.warning("LLM 批量翻译返回的 JSON 无效 (attempt %d): %s. Raw text: %s", attempt, json_e, raw_text)
                continue # 继续重试

        except Exception as e:
            logger.warning("LLM 批量翻译失败 (attempt %d): %s", attempt, e)
            time.sleep(backoff * attempt)

    # 如果所有尝试都失败，返回一个所有词都为空列表的字典
    return {t: [] for t in terms}


def _clean_translation_text(text: str) -> str:
    """Removes content within parentheses (both full-width and half-width)."""
    return re.sub(r'[\(（][^)）]*[\)）]', '', text).strip()


def _get_translation_candidates(term: str, topic: Optional[str] = None) -> List[str]:
    # 完善的翻译模块
    # 1️⃣ 尝试本地数据库
    local = query_term_translation(term)
    if local:
        logger.info("本地库找到翻译: %s -> %s", term, local)
        # 若本地已有多个，仍按旧逻辑返回列表（不截断）
        return [_clean_translation_text(t) for t in local]

    # 2️⃣ LLM 单项（结合 topic，只返回一个最佳译名）
    candidates = _fetch_llm_translation(term, topic=topic)

    # 去重和清理
    seen = set()
    cleaned: List[str] = []
    for c in candidates:
        c_clean = _clean_translation_text(c.strip())
        if c_clean and c_clean not in seen:
            seen.add(c_clean)
            cleaned.append(c_clean)
    return cleaned


def get_translation_candidates_batch(terms: List[str], batch_size: int = 50, topic: Optional[str] = None) -> Dict[str, Optional[List[str]]]:
    """
    批量获取翻译候选（带分批机制）。
    现在会结合 topic，并且每个词只返回一个最佳译名（以单元素列表形式返回）。
    流程：
    1️⃣ 先查本地库
    2️⃣ 收集需要 LLM 批量翻译的词
    3️⃣ 分批调用 LLM
    4️⃣ 清理去重
    """
    result: Dict[str, Optional[List[str]]] = {}
    llm_terms: List[str] = []

    # Step 1: 查询本地翻译
    for term in terms:
        local = query_term_translation(term)
        if local:
            result[term] = local
        else:
            llm_terms.append(term)

    # Step 2: 分批调用 LLM（每项只返回一个最佳译名）
    if llm_terms:
        for i in range(0, len(llm_terms), batch_size):
            batch = llm_terms[i:i + batch_size]
            llm_results = _fetch_llm_translation_batch(batch, topic=topic)
            for term, candidates in llm_results.items():
                result[term] = candidates

    # Step 3: 清理去重（尽管每个通常只有一个，但仍保证稳健性）
    for term, candidates in result.items():
        cleaned: List[str] = []
        seen: set[str] = set()
        if candidates:
            for c in candidates:
                c_clean = c.strip()
                if c_clean and c_clean not in seen:
                    seen.add(c_clean)
                    cleaned.append(c_clean)
        result[term] = cleaned[:1] if cleaned else None

    # 为在 llm_terms 中但未在 result 中的术语设置 None
    for term in llm_terms:
        if term not in result:
            result[term] = None
    return result


# ------------------ 单术语翻译 ------------------

def translate_term(term_en: str, topic: Optional[str] = None) -> Optional[List[str]]:
    """
    翻译单个词，结合 topic，只返回一个最佳译名（以单元素列表返回）。
    如果找不到翻译，则返回 None。
    """
    candidates = _get_translation_candidates(term_en, topic=topic)
    if not candidates:
        print(f"最终未找到 {term_en} 的翻译")
        return None
    # 只保留一个最佳
    return candidates[:1]
