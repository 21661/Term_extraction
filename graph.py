from typing import List, Dict
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END
import spacy, json, time, logging, re
from utils.LLMClientManager import LLMclientManager
from utils.Get_term import (
    translate_term, get_translation_candidates_batch,
)
from utils.workflow_adapter import _unwrap
from utils.TimeNode import timed_node

import typing
from utils.TermState import TermState
from Nodes._reflect_node import reflect_sync_node,route_after_reflect
from Nodes.select_top_terms import select_top_terms
from Nodes._terms_only_batch import _terms_only_batch

# ===================== 2️⃣ 初始化 =====================
nlp = spacy.load("en_core_web_trf")

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== 9️⃣ 构建 LangGraph 工作流（重构为 main.extract 的批处理流程） =====================
_TERMS_ONLY_WORKFLOW = None


@timed_node()
def _init_extract_state(state: TermState) -> TermState:
    original: TermState | tuple | dict = state
    inner, parent, key = _unwrap(state)
    sd: TermState = typing.cast(TermState, inner if isinstance(inner, dict) else TermState())
    # 仅初始化缺失的键，避免重复写入
    updates: TermState = {}
    if "summary" not in sd:
        updates["summary"] = sd.get("summary", "") or ""
    return typing.cast(TermState, updates)


@timed_node()
def _aggregate_unique_terms(state: TermState) -> TermState:
    inner, parent, key = _unwrap(state)
    sd: TermState = typing.cast(TermState, inner if isinstance(inner, dict) else TermState())

    # --- 修改开始 ---
    # 不要读取 sd["selected_terms"]，因为它可能被并行覆盖了
    # 从结构化的 chunk_terms 中提取所有出现过的术语
    chunk_terms_data = sd.get("chunk_terms", [])
    all_extracted_terms = []

    if chunk_terms_data:
        for item in chunk_terms_data:
            # item 是 {'chunk_id': '...', 'terms': [...]}
            terms = item.get("terms", [])
            if isinstance(terms, list):
                all_extracted_terms.extend(terms)

    # 去重
    unique_terms = sorted(set(t for t in all_extracted_terms if isinstance(t, str) and t.strip()))
    # --- 修改结束 ---

    logger.info("Aggregated %d unique terms from chunks for translation.", len(unique_terms))
    return typing.cast(TermState, {"unique_terms": unique_terms})


@timed_node()
def _single_translate_concurrent(state: TermState) -> TermState:
    original: TermState | tuple | dict = state
    inner, parent, key = _unwrap(state)
    sd: TermState = typing.cast(TermState, inner if isinstance(inner, dict) else TermState())

    unique_terms: List[str] = sd.get("unique_terms", [])
    topic = sd.get("summary", "")
    translations_map: Dict[str, List[str]] = {}

    if not unique_terms:
        return typing.cast(TermState, {"translations_map": translations_map})

    # 1. 检查是否存在高并发的 MT 模型
    target_mt_model = "tencent/Hunyuan-MT-7B"
    has_mt_model = LLMclientManager.check_model_exists(target_mt_model)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    if has_mt_model:
        # =====================================================
        # 分支 A: 使用 MT 模型 (高并发单词模式)
        # 适用场景：专用于翻译的模型，支持超高并发，单次请求延迟极低
        # =====================================================
        logger.info(f"检测到 MT 模型 {target_mt_model}，启用【单词高并发】模式。")
        max_workers = 100

        def worker_single(term: str):
            # 这里沿用你之前的单词重试逻辑
            retry_delays = [0.3, 0.6, 1.0]
            for attempt in range(3):
                try:
                    res = translate_term(term, topic=topic)
                    if res:
                        return term, res
                    if attempt < 2:
                        time.sleep(retry_delays[attempt])
                except Exception as e:
                    logger.warning(f"Translate attempt {attempt + 1} failed for {term}: {e}")
                    if attempt < 2:
                        time.sleep(retry_delays[attempt])
            return term, []

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_term = {executor.submit(worker_single, t): t for t in unique_terms}
                for fut in as_completed(future_to_term):
                    term = future_to_term[fut]
                    try:
                        k, v = fut.result()
                        translations_map[k] = v
                    except Exception as e:
                        logger.warning(f"Single translation failure for {term}: {e}")
                        translations_map[term] = []
        except Exception as e:
            logger.exception("Concurrent single translation failed: %s", e)

    else:
        # =====================================================
        # 分支 B: 使用通用模型 (分块并发模式)
        # 适用场景：GLM/GPT等通用模型，RPM有限，但Context Window较大，适合打包处理
        # =====================================================
        logger.info("未找到 MT 模型，启用【分块并发批量】模式。")

        # 配置参数：通用模型并发不宜过高，避免 429
        max_workers = 8  # 并发线程数
        batch_size = 15  # 每个线程处理的词数

        # 辅助函数：将列表分块
        def chunk_list(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        # 创建分块任务列表
        term_chunks = list(chunk_list(unique_terms, batch_size))
        logger.info(
            f"共 {len(unique_terms)} 个术语，分为 {len(term_chunks)} 个批次 (Batch Size: {batch_size})，并发数: {max_workers}")

        def worker_batch(terms_chunk: List[str]):
            # 直接调用你 Get_term.py 里的批量函数
            # 该函数内部会处理本地库查询和 LLM JSON解析
            try:
                # 注意：这里调用 get_translation_candidates_batch
                # 即使传入 batch_size，因为我们切分后的 chunk 长度本身就等于 batch_size，
                # 所以该函数内部通常只会发一次 LLM 请求
                return get_translation_candidates_batch(terms_chunk, batch_size=batch_size, topic=topic)
            except Exception as e:
                logger.error(f"Batch worker failed: {e}")
                return {t: [] for t in terms_chunk}

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交分块任务
                futures = [executor.submit(worker_batch, chunk) for chunk in term_chunks]

                for fut in as_completed(futures):
                    try:
                        batch_result = fut.result()  # 返回的是 Dict[str, List[str]]
                        # 将批量结果合并到主字典中
                        translations_map.update(batch_result)
                    except Exception as e:
                        logger.exception("Unexpected error in batch translation future")

        except Exception as e:
            logger.exception("Concurrent batch translation failed: %s", e)
            # 兜底：确保所有词都有键
            for t in unique_terms:
                if t not in translations_map:
                    translations_map[t] = []

    # 最终检查，防止漏词
    missing_count = 0
    for t in unique_terms:
        if t not in translations_map:
            translations_map[t] = []
            missing_count += 1

    print(f"Completed translations_map. Total: {len(translations_map)}, Missing filled: {missing_count}")
    return typing.cast(TermState, {"translations_map": translations_map})

@timed_node()
def _assemble_annotations(state: TermState) -> TermState:
    import re
    original = state
    inner, parent, key = _unwrap(state)
    sd: TermState = typing.cast(TermState, inner if isinstance(inner, dict) else TermState())

    per_chunk_results = sd.get("chunk_terms", [])
    translations_map: Dict[str, List[str]] = sd.get("translations_map", {})
    term_annotations: Dict[str, typing.Any] = {}

    def lookup_candidates(t: str):
        key_raw = t
        key_lower = t.lower().strip()
        # 简单的归一化查找
        return (
                translations_map.get(key_raw) or
                translations_map.get(key_lower) or
                []
        )

    def _pick_best(term: str, cands: List[str]) -> str:
        if not cands:
            return term  # 如果没有翻译，返回原词

        term_norm = term.strip().lower()
        filtered = [c.strip() for c in cands if c and c.strip()]

        # 1. 优先找中文 (Unicode范围)
        for c in filtered:
            if re.search(r"[\u4e00-\u9fff]", c):
                return c

        # 2. 找非中文但内容不同的 (比如全称扩展)
        for c in filtered:
            if c.lower() != term_norm:
                return c

        # 3. 如果只有相同的词（比如 Docling -> Docling），也返回它
        if filtered:
            return filtered[0]

        return term

    for chunk_item in per_chunk_results:
        cid = chunk_item.get("chunk_id")
        terms = chunk_item.get("terms", [])

        items = []
        for t in terms:
            cands = lookup_candidates(t)
            if cands is None:
                continue
            chosen = _pick_best(t, cands)

            # --- 修改核心 ---
            # 移除 if chosen != t 的判断
            # 只要这个词被选中了，无论是否有翻译变化，都应该输出
            items.append({"term": t, "translation": chosen})
            # ---------------

        term_annotations[str(cid)] = items


    print(f"Assembled term_annotations: {term_annotations}")
    return typing.cast(TermState, {"term_annotations": term_annotations})  # 注意这里 key 修正为 term_annotations 保持一致



@timed_node()
def build_graph() -> Runnable:
    graph: StateGraph[TermState] = StateGraph(TermState)

    graph.add_node("init", _init_extract_state)
    graph.add_node("terms_only_batch", _terms_only_batch)
    graph.add_node("select_top_terms", select_top_terms)
    graph.add_node("reflect_terms", reflect_sync_node)
    graph.add_node("aggregate_unique_terms", _aggregate_unique_terms)

    graph.add_node("batch_translate", _single_translate_concurrent)

    graph.add_node("assemble_annotations", _assemble_annotations)


    graph.set_entry_point("init")

    graph.add_edge("init", "terms_only_batch")

    graph.add_edge("terms_only_batch", "select_top_terms")

    graph.add_edge("select_top_terms", "reflect_terms")

    graph.add_conditional_edges(
        "reflect_terms",
        route_after_reflect,
        {
            "retry": "select_top_terms",       # 仍然只回到 select_top_terms
            "proceed": "aggregate_unique_terms",
        },
    )

    graph.add_edge("aggregate_unique_terms", "batch_translate")

    graph.add_edge("batch_translate", "assemble_annotations")

    graph.add_edge("assemble_annotations", END)

    return graph.compile()
