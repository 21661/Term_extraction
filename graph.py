import asyncio
from typing import List, Dict
from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END
import spacy, json, time, logging, re
from utils.LLMClientManager import LLMclientManager
from utils.Get_term import (
     translate_batch_async, translate_term_async,
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
async def _single_translate_concurrent(state: TermState) -> TermState:
    """
    翻译节点（极致性能版）
    """
    inner, parent, key = _unwrap(state)
    sd: TermState = typing.cast(TermState, inner if isinstance(inner, dict) else TermState())

    unique_terms: List[str] = sd.get("unique_terms", [])
    topic = sd.get("summary", "")
    translations_map: Dict[str, List[str]] = {}

    if not unique_terms:
        return typing.cast(TermState, {"translations_map": translations_map})

    # 检查是否有 MT 模型
    target_mt_model = "tencent/Hunyuan-MT-7B"
    has_mt_model = LLMclientManager.check_model_exists(target_mt_model)

    if has_mt_model:
        # =====================================================
        # 策略 A: 单词高并发 (MT 模型)
        # =====================================================
        logger.info(f"🚀 启用 MT 高并发模式 ({target_mt_model})")

        # 信号量：控制同时飞在天上的请求数，防止 API 限流
        # 建议根据你的 API 额度调整，100 是个激进但高效的值
        semaphore = asyncio.Semaphore(100)

        async def worker(term):
            async with semaphore:
                # 失败自动重试 2 次
                for _ in range(2):
                    res = await translate_term_async(term, topic, target_mt_model)
                    if res: return term, res
                    # 稍微退避一下
                    # await asyncio.sleep(0.1)
                return term, []

        # 创建任务并发执行
        tasks = [worker(t) for t in unique_terms]
        results = await asyncio.gather(*tasks)

        for term, res in results:
            translations_map[term] = res

    else:
        # =====================================================
        # 策略 B: 批量分块 (通用模型)
        # =====================================================
        logger.info("📦 启用通用模型批量模式")

        batch_size = 20  # 通用模型一次处理 20 个词比较稳
        max_concurrency = 10  # 控制并发数

        semaphore = asyncio.Semaphore(max_concurrency)

        # 切分列表
        chunks = [unique_terms[i:i + batch_size] for i in range(0, len(unique_terms), batch_size)]

        async def worker_batch(chunk):
            async with semaphore:
                for _ in range(2):  # 简单重试
                    res = await translate_batch_async(chunk, topic)
                    if res: return res
                return {}

        tasks = [worker_batch(c) for c in chunks]
        results = await asyncio.gather(*tasks)

        for batch_map in results:
            if batch_map:
                translations_map.update(batch_map)

    # 兜底检查
    missing = 0
    for t in unique_terms:
        if t not in translations_map:
            translations_map[t] = []
            missing += 1

    logger.info(f"翻译完成。总数: {len(translations_map)}, 补全空缺: {missing}")
    return typing.cast(TermState, {"translations_map": translations_map})


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
        return (
                translations_map.get(key_raw) or
                translations_map.get(key_lower) or
                []
        )

    # --- 修改点 1：_pick_best 逻辑修正 ---
    def _pick_best(term: str, cands: List[str]) -> typing.Optional[str]:
        # 只有当完全没有候选词时，才视为“失败”，返回 None
        if not cands:
            return None

        filtered = [c.strip() for c in cands if c and c.strip()]
        if not filtered:
            return None

        term_norm = term.strip().lower()

        # 优先策略不变：先找中文
        for c in filtered:
            if re.search(r"[\u4e00-\u9fff]", c):
                return c

        # 其次：找和原文不一样的（比如全称扩展）
        for c in filtered:
            if c.lower() != term_norm:
                return c

        # 【关键修正】：如果只剩下和原文一样的词（例如 AVL -> AVL），直接返回它
        # 只要翻译表里有它，就说明它是有效结果
        return filtered[0]

    for chunk_item in per_chunk_results:
        cid = chunk_item.get("chunk_id")
        terms = chunk_item.get("terms", [])

        items = []
        for t in terms:
            cands = lookup_candidates(t)

            # 这里的 cands 如果是 []，_pick_best 会返回 None
            chosen = _pick_best(t, cands)

            # --- 修改点 2：只过滤 None ---
            if chosen is None:
                # 说明翻译表里根本没这个词（或者值是空的），跳过
                continue

            items.append({"term": t, "translation": chosen})
            # ---------------------------


            term_annotations[str(cid)] = items

    print(f"Assembled term_annotations: {term_annotations}")
    return typing.cast(TermState, {"term_annotations": term_annotations})



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
