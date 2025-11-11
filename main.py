# main.py (simplified unified batch translation workflow)
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from concurrent.futures import ThreadPoolExecutor, as_completed
from graph import build_graph_terms_only
from typing import List, Dict, Any
import logging
import time
import re
from utils.Get_term import get_translation_candidates_batch, translate_term as translate_term_external
from utils.db_interface import query_term_translation

app = FastAPI(title="Term Extraction API")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize only the terms-only workflow (extraction without translation)
try:
    WORKFLOW_TERMS_ONLY = build_graph_terms_only()
except Exception as e:
    logger.exception("Failed to build terms-only graph at startup: %s", e)
    WORKFLOW_TERMS_ONLY = None


def process_chunk_terms_only(chunk: Dict[str, Any], summary: str) -> Dict[str, Any]:
    """运行术语提取（不翻译），返回 {chunk_id, terms, term_types, error?}"""
    if not isinstance(chunk, dict) or len(chunk) != 1:
        return {"chunk_id": None, "terms": [], "term_types": {}, "error": "malformed_chunk"}
    if WORKFLOW_TERMS_ONLY is None:
        return {"chunk_id": next(iter(chunk.keys())), "terms": [], "term_types": {}, "error": "terms_only_workflow_not_initialized"}

    chunk_id, chunk_text = next(iter(chunk.items()))
    text = chunk_text if isinstance(chunk_text, str) else str(chunk_text)

    init_state = {
        "text": text,
        "candidates": [],
        "terms": [],
        "topic": summary,
        "translations": {},
        "final_translations": {},
        "term_types": {},
    }
    try:
        result = WORKFLOW_TERMS_ONLY.invoke(init_state)
        selected_terms = result.get("selected_terms") or result.get("terms") or []
        term_types = result.get("term_types", {})
        return {"chunk_id": str(chunk_id), "terms": list(selected_terms), "term_types": dict(term_types)}
    except Exception as e:
        logger.exception("terms-only workflow error: %s", e)
        return {"chunk_id": str(chunk_id), "terms": [], "term_types": {}, "error": str(e)}


def pick_best_translation(cands: List[str]) -> str:
    for c in cands:
        if re.search(r"[\u4e00-\u9fff]", c):
            return c.strip()
    for c in cands:
        if c.strip():
            return c.strip()
    return ""


@app.get("/")
async def root():
    return {"status": "ok", "terms_only_initialized": WORKFLOW_TERMS_ONLY is not None}


@app.post("/extract")
async def extract(request: Request):
    """
    统一翻译模式：
    1) 并行处理每段，提取术语（不翻译）
    2) 汇总全部唯一术语，一次性批量翻译
    3) 将翻译结果按每段术语映射回去并返回注释

    输入 JSON:
    {
      "summary": "...",  # 主题/上下文（可选提升术语筛选质量）
      "chunks": [ {"1": "段落1"}, {"2": "段落2"}, ... ]
    }
    输出字段 termAnnotations 与之前保持一致。
    """
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid json: {e}")

    summary = payload.get("summary", "")
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list):
        raise HTTPException(status_code=400, detail="chunks must be a list of {id: text} objects")

    # 1) 并行术语提取
    per_chunk_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    start_ts = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_chunk_terms_only, chunk, summary) for chunk in chunks]
        for f in as_completed(futures):
            r = f.result()
            if not r:
                continue
            if r.get("error"):
                errors.append({"chunk_id": r.get("chunk_id"), "error": r.get("error")})
            per_chunk_results.append(r)
    logger.info("Terms-only extraction done for %d chunks in %.2fs", len(chunks), time.time() - start_ts)

    # 2) 汇总唯一术语
    all_terms: List[str] = []
    for r in per_chunk_results:
        all_terms.extend(r.get("terms", []))
    unique_terms = sorted(set(t for t in all_terms if isinstance(t, str) and t.strip()))

    # 3) 批量翻译
    translations_map: Dict[str, List[str]] = {}
    try:
        translations_map = get_translation_candidates_batch(unique_terms) or {}
    except Exception as e:
        logger.warning("Batch translation failed, will fallback to single + db: %s", e)
        translations_map = {}

    # 单项回退 + DB 回退
    for t in unique_terms:
        if translations_map.get(t):
            continue
        try:
            single = translate_term_external(t) or []
        except Exception:
            single = []
        if not single:
            try:
                local = query_term_translation(t) or []
                single = local
            except Exception:
                pass
        translations_map[t] = single or []

    # 4) 按分段组装结果
    term_annotations: List[Dict[str, Any]] = []
    for r in per_chunk_results:
        cid = r.get("chunk_id")
        term_types = r.get("term_types", {}) or {}
        final_translations: Dict[str, str] = {}
        for t in r.get("terms", []):
            cands = translations_map.get(t, [])
            chosen = pick_best_translation(cands)
            if chosen:
                final_translations[t] = chosen
        term_list = []
        pn_list = []
        for term, trans in final_translations.items():
            ttype = term_types.get(term, "term")
            entry = {term: trans}
            if ttype == "proper_noun":
                pn_list.append(entry)
            else:
                term_list.append(entry)
        term_annotations.append({cid: [{"term": term_list}, {"proper_noun": pn_list}]})

    resp = {"termAnnotations": term_annotations}
    if errors:
        resp["errors"] = errors
    resp["stats"] = {
        "total_chunks": len(chunks),
        "unique_terms": len(unique_terms),
        "translated_terms": sum(len(x[list(x.keys())[0]][0]["term"]) + len(x[list(x.keys())[0]][1]["proper_noun"]) for x in term_annotations) if term_annotations else 0
    }
    return JSONResponse(resp)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
