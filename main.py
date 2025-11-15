# main.py (simplified unified batch translation workflow via build_graph)
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import uvicorn
import logging
from utils.db_interface import ensure_db_initialized
from graph import build_graph

app = FastAPI(title="Term Extraction API")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize the unified workflow (extraction + aggregation + translation)
try:
    try:
        ensure_db_initialized()
    except Exception as db_e:
        logger.warning("Database initialization failed at startup: %s", db_e)
    WORKFLOW = build_graph()
except Exception as e:
    logger.exception("Failed to build unified graph at startup: %s", e)
    WORKFLOW = None


@app.get("/")
async def root():
    return {"status": "ok", "workflow_initialized": WORKFLOW is not None}


class ExtractPayload(BaseModel):
    summary: Optional[str] = Field("", description="主题/上下文，可选")
    chunks: Dict[str, str] = Field(..., description="对象形式的段落集合：{id: text}")


@app.post("/extract")
async def extract(payload: ExtractPayload):
    """
    使用 graph.build_graph 构建的统一流程：
    - 输入：summary + chunks
    - 自动执行每段术语提取、汇总去重、批量翻译与组装注释
    - 输出：termAnnotations（按需简化为只返回注释对象）
    """
    if WORKFLOW is None:
        return JSONResponse({"termAnnotations": {}})

    init_state: Dict[str, Any] = {
        "summary": payload.summary or "",
        "chunks": payload.chunks or {},
    }

    try:
        result: Dict[str, Any] = WORKFLOW.invoke(init_state)  # type: ignore
    except Exception as e:
        logger.exception("Unified graph invocation failed: %s", e)
        return JSONResponse({"term_annotations": {}})

    resp: Dict[str, Any] = {
        "term_annotations": result.get("termAnnotations", {}),
    }
    return JSONResponse(resp)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8007, reload=False)
