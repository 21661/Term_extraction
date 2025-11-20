from utils.TimeNode  import timed_node
from utils.TermState import TermState
from utils.workflow_adapter import _rewrap,_unwrap
import typing
from utils.extract_candidates  import extract_candidates
from Nodes.select_top_terms import select_top_terms
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from utils.candidate_tool import (
    normalize_candidate,
    is_noise_candidate,
    dedupe_keep_longest,
)
@timed_node()
def _terms_only_batch(state: TermState) -> TermState:
    """
    并发处理 chunk，提取术语 → 清洗 → 去噪 → select_top_terms
    """
    original: TermState | tuple | dict = state
    inner, parent, key = _unwrap(state)
    sd: TermState = typing.cast(TermState, inner if isinstance(inner, dict) else TermState())

    chunks: dict[str, str] = sd.get("chunks", {}) or {}
    summary = sd.get("summary", "") or ""

    per_chunk_results = []
    candidates = set(sd.get("candidates", []))
    errors = []

    def process_chunk(cid: str, text: str) -> dict:
        try:
            init_state = {"text": text, "topic": summary}

            # 1) 原始提取
            s1 = extract_candidates(init_state)

            # 2) 清洗 + 过滤
            raw_terms = s1.get("candidates", []) or []
            cleaned = []
            for t in raw_terms:
                nt = normalize_candidate(t)
                if nt and not is_noise_candidate(nt):
                    cleaned.append(nt)

            cleaned = dedupe_keep_longest(cleaned)
            s1["candidates"] = cleaned

            # 3) select_top_terms
            inner_state = s1.get("inner") or s1
            if "terms" not in inner_state:
                inner_state["terms"] = cleaned

            s2 = select_top_terms(inner_state)
            terms = s2.get("selected_terms") or []

            return {"chunk_id": str(cid), "terms": terms}

        except Exception as e:
            return {"chunk_id": str(cid), "terms": [], "error": str(e)}

    # 并发执行
    if chunks:
        max_workers = min(4, len(chunks))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_cid = {
                executor.submit(process_chunk, cid, ctext): cid
                for cid, ctext in chunks.items()
            }
            for fut in as_completed(future_to_cid):
                res = fut.result()
                per_chunk_results.append(res)
                candidates.update(res.get("terms", []))

    term_to_chunks = defaultdict(list)
    for r in per_chunk_results:
        cid = r["chunk_id"]
        for t in r.get("terms", []):
            term_to_chunks[t].append(cid)
    term_to_chunks = dict(term_to_chunks)
    updates: TermState = {
        "chunks": chunks,
        "candidates": list(candidates),
        "term_to_chunks": term_to_chunks,
    }

    return typing.cast(TermState, updates)

