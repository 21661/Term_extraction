# pydantic模板
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PostedTask(BaseModel):
    target_lang: Optional[str] = Field("中文", description="目标语言，默认中文")
    strategy: Optional[str] = Field("normal", description="翻译策略，默认 normal，可选fast、normal、thinking")
    client_request_id : str = ""

class termAnnotations(BaseModel):
    """术语注解模型"""
    term_annotations: Optional[dict[str, List[dict[str, str]]]] = None

class DocumentResult(BaseModel):
    """文档处理结果模型"""
    task_id: str = Field(...,description="任务ID")
    status: str = Field(...,description="状态，success/processing/pending/error")
    error : Optional[str] = Field(None,description="错误信息")
    original_markdown: Optional[dict[str, str]] = Field(None,description="原始Markdown内容")
    translated_markdown: Optional[dict[str, str]] = Field(None,description="翻译后的Markdown内容")
    term_annotations: Optional[dict[str, List[dict[str, str]]]] = Field(None,description="术语表")
    client_request_id: str = None

class GeneralResponse(BaseModel):
    """通用处理结果模型"""
    status: str = Field(...,description="状态，success/processing/pending/error")
    task_id: Optional[str] = Field(None,description="任务ID")
    error: Optional[str] = Field(None,description="错误信息")

class LLMConfig(BaseModel):
    """单个 LLM 的配置项"""

    name: str = Field(..., min_length=1, description="唯一名称，用于引用该 LLM")
    base_url: str = Field(..., description="该 LLM 的基础 API 地址")
    api_key: str = Field(..., description="访问该 LLM 的 API Key")
    model_name: str = Field(..., min_length=1, description="要调用的具体模型名")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度，范围 0~2")
    extra_config: Optional[Dict[str, Any]] = Field(default=None, description="额外的运行时配置，如 max_concurrency / rpm 等")
    extra_body: Dict[str, Any] = Field(default_factory=dict, description="额外透传给模型的请求体")

class SafeLLMConfig(BaseModel):
    """单个 LLM 的配置项，没有key """

    name: str = Field(..., min_length=1, description="唯一名称，用于引用该 LLM")
    base_url: str = Field(..., description="该 LLM 的基础 API 地址")
    model_name: str = Field(..., min_length=1, description="要调用的具体模型名")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="温度，范围 0~2")
    # max_concurrency: int = Field(1, ge=0, le=1000, description="最大并发，>=0")
    # rpm: int | None = Field(None, ge=0, description="每分钟并发数")
    extra_config: Optional[Dict[str, Any]] = Field(default=None,
                                                   description="额外的运行时配置，如 max_concurrency / rpm 等")
    extra_body: Dict[str, Any] = Field(default_factory=dict, description="额外透传给模型的请求体")

class ChangeLLMsConfigRequest(BaseModel):
    """更改LLM配置模型"""
    task_type: str = Field("change_llms_config")
    llms: List[LLMConfig] = Field(..., min_items=1)

class LLMsConfigResponse(BaseModel):
    """返回LLM配置模型"""
    llms: List[SafeLLMConfig] = Field(..., min_items=1)

class QueueStatusModel(BaseModel):
    total_tasks: int
    pending_count: int
    processing_count: int
    finished_count: int
    error_count: int
    async_tasks: int
    sync_tasks: int
    current_task: Optional[str] = None
    is_running: bool

class RecentTaskModel(BaseModel):
    task_id: str
    status: str
    is_async: bool
    created_at: str
    finished_at: Optional[str] = None

class StatisticsModel(BaseModel):
    total_tasks: int
    status_counts: Dict[str, int]
    async_tasks: int
    sync_tasks: int
    recent_tasks: List[RecentTaskModel]
    database_size: int
    is_running: bool
    current_task: Optional[str] = None