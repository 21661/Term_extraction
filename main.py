# main.py (simplified unified batch translation workflow via build_graph)
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import uvicorn
import logging
from utils.db_interface import ensure_db_initialized
from graph import build_graph
from utils.LLMManager import AgentManager, AgentConfigRequestModel, LLMConfigModel, AgentConfigResponseModel
from utils.LLMClientManager import LLMclientManager
import os

app = FastAPI(title="Term Extraction API")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Global workflow object
WORKFLOW = None

# Initialize LLM configurations at startup (default / fallback)
try:
    default_llms = [
        LLMConfigModel(
            name="zhipu",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            api_key="3a1caa7e1196474b9e93ecca12e4ee93.mxHhWISC0MO7tE7p",
            model_name="GLM-4-Flash-250414",
        ),
        LLMConfigModel(
            name="siliconflow",
            base_url="https://api.siliconflow.cn/v1/",
            api_key="sk-zzzponivncchgzwbyuyrawqvniijligbyorwkuultoyddvqz",
            model_name="tencent/Hunyuan-MT-7B",
        ),
    ]
    default_config = AgentConfigRequestModel(llms=default_llms)
    AgentManager().update_agent(default_config)
    logger.info("LLM AgentManager initialized with default configuration.")
except Exception as llm_e:
    logger.error("Failed to initialize LLM AgentManager: %s", llm_e)

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
def extract_sync(payload: ExtractPayload):
    """
    同步版本：直接调用同步 Graph。
    """
    if WORKFLOW is None:
        return JSONResponse({"termAnnotations": {}})

    init_state: Dict[str, Any] = {
        "summary": payload.summary or "",
        "chunks": payload.chunks or {},
    }

    try:
        result: Dict[str, Any] = WORKFLOW.invoke(init_state)  # 同步调用
    except Exception as e:
        logger.exception("Unified graph invocation failed: %s", e)
        return JSONResponse({"term_annotations": {}})

    resp: Dict[str, Any] = {
        "term_annotations": result.get("term_annotations", {}),
    }
    return JSONResponse(resp)


# --- New dynamic configuration endpoints ---
class UpdateConfigRequest(AgentConfigRequestModel):
    pass

class UpdateConfigResponse(BaseModel):
    status: str
    applied_llm_count: int
    workflow_rebuilt: bool
    error: Optional[str] = None
    current_models: Optional[Dict[str, str]] = None  # name -> model_name

@app.post("/llms/config", response_model=UpdateConfigResponse)
async def update_llm_config(req: UpdateConfigRequest):
    global WORKFLOW
    try:
        # Update agent config
        AgentManager().update_agent(req)
        # Reset client manager cache so new config will be lazily reinitialized
        LLMclientManager.reset_clients()

        # Attempt to rebuild workflow (optional: if it depends on LLM config)
        rebuilt = False
        try:
            WORKFLOW = build_graph()
            rebuilt = WORKFLOW is not None
        except Exception as e:
            logger.warning("Workflow rebuild failed after config update: %s", e)

        # Prepare model mapping snapshot
        config = AgentManager().get_config()
        model_map = {c.name: c.model_name for c in (config.llms if config else [])}

        return UpdateConfigResponse(
            status="ok",
            applied_llm_count=len(req.llms),
            workflow_rebuilt=rebuilt,
            current_models=model_map
        )
    except Exception as e:
        logger.exception("Failed to update LLM config: %s", e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/llms/config", response_model=AgentConfigResponseModel)
async def get_llm_config():
    config = AgentManager().get_config()
    if not config:
        raise HTTPException(status_code=404, detail="No LLM configuration loaded")
    return config


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8007, reload=False)
