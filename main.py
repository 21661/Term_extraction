# main.py (simplified unified batch translation workflow via build_graph)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging
from utils.db_interface import ensure_db_initialized
from graph import build_graph
from utils.LLMManager import AgentManager, AgentConfigRequestModel, LLMConfigModel, AgentConfigResponseModel
from utils.LLMClientManager import LLMclientManager
import uuid
import time
from enum import Enum
from fastapi import BackgroundTasks, HTTPException
from typing import Optional, Dict, Any

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


class TaskStatus(str, Enum):
    PENDING = "pending"  # 排队中
    PROCESSING = "processing"  # 处理中
    COMPLETED = "completed"  # 完成
    FAILED = "failed"  # 失败


# 全局任务存储 {task_id: dict}
# 注意：服务重启后数据会丢失。生产环境建议使用 Redis。
TASK_STORE: Dict[str, Dict[str, Any]] = {}


# ==========================================
# 2. 响应模型定义
# ==========================================

class TaskSubmitResponse(BaseModel):
    task_id: str
    message: str
    status: TaskStatus


class TaskResultResponse(BaseModel):
    task_id: str  # 保持接口契约不变
    status: TaskStatus
    submitted_at: float
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ==========================================
# 3. 后台处理逻辑
# ==========================================

async def process_workflow_task(task_id: str, init_state: Dict[str, Any]):
    """
    后台运行 Workflow (异步模式)
    """
    try:
        TASK_STORE[task_id]["status"] = TaskStatus.PROCESSING

        if WORKFLOW is None:
            raise RuntimeError("Workflow graph is not initialized.")

        # === 2. 使用 await ainvoke ===
        # 这会真正利用异步特性，并发执行你刚才优化的 reflect 节点
        graph_result = await WORKFLOW.ainvoke(init_state)
        # ============================

        term_annotations = graph_result.get("term_annotations", {})

        TASK_STORE[task_id]["result"] = {"term_annotations": term_annotations}
        TASK_STORE[task_id]["status"] = TaskStatus.COMPLETED
        TASK_STORE[task_id]["completed_at"] = time.time()
        logger.info(f"Task {task_id} completed successfully.")

    except Exception as e:
        logger.exception(f"Task {task_id} failed: {e}")
        TASK_STORE[task_id]["error"] = str(e)
        TASK_STORE[task_id]["status"] = TaskStatus.FAILED
        TASK_STORE[task_id]["completed_at"] = time.time()


# ==========================================
# 4. 接口定义
# ==========================================

@app.post("/extract", response_model=TaskSubmitResponse)
async def extract_async(payload: ExtractPayload, background_tasks: BackgroundTasks):
    """
    [异步接口] 提交提取任务。
    返回 task_id，客户端需通过 GET /extract/{task_id} 轮询结果。
    """
    if WORKFLOW is None:
        raise HTTPException(status_code=503, detail="Workflow system not initialized")

    # 1. 生成任务 ID
    task_id = str(uuid.uuid4())

    # 2. 准备 Workflow 初始状态
    init_state: Dict[str, Any] = {
        "summary": payload.summary or "",
        "chunks": payload.chunks or {},
    }

    # 3. 初始化任务记录
    TASK_STORE[task_id] = {
        "id": task_id,
        "status": TaskStatus.PENDING,
        "submitted_at": time.time(),
        "result": None,
        "error": None
    }

    # 4. 将任务加入 FastAPI 后台队列 (立即响应，后台执行)
    background_tasks.add_task(process_workflow_task, task_id, init_state)

    logger.info(f"Task submitted: {task_id}")

    return {
        "task_id": task_id,
        "message": "Task submitted. Poll results at /extract/{task_id}",
        "status": TaskStatus.PENDING
    }





@app.get("/extract/{task_id}", response_model=TaskResultResponse)
async def get_extract_result(task_id: str):
    """
    [轮询接口] 获取任务执行状态和结果。
    """
    task = TASK_STORE.get(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f"Task ID '{task_id}' not found")

    # === 修复点 ===
    # TASK_STORE 中存储的是 {"id": "...", ...}
    # 但 Response Model 期望 {"task_id": "...", ...}
    # 所以我们需要构造一个新的字典来适配
    return {
        "task_id": task.get("id"),  # 将存储中的 'id' 映射给 'task_id'
        "status": task.get("status"),
        "submitted_at": task.get("submitted_at"),
        "completed_at": task.get("completed_at"),
        "result": task.get("result"),
        "error": task.get("error")
    }


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
