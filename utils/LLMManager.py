import json
from typing import List, Dict, Any, Optional, Union
from langchain_openai import ChatOpenAI

# 2. 使用 Pydantic 定义数据模型
from pydantic import BaseModel, Field, HttpUrl, ValidationError

class LLMConfigModel(BaseModel):
    """定义单个LLM的配置模型"""
    name: str
    base_url: str
    api_key: str
    model_name: str
    temperature: float = Field(default=0.1, description="模型的温度，控制创造性")
    # 将并发与速率限制移入 extra_config
    extra_config: Optional[Dict[str, Any]] = Field(default=None, description="额外的运行时配置，如 max_concurrency / rpm 等")
    extra_body: Optional[Dict[str, Any]] = Field(default=None, description="厂商自定义参数")


class AgentConfigRequestModel(BaseModel):
    """POST请求的模型"""
    task_type: str = Field(default="change_llms_config")
    llms: List[LLMConfigModel]

class AgentConfigResponseModel(BaseModel):
    """GET请求返回的模型"""
    llms: List[LLMConfigModel]

class StatusResponseModel(BaseModel):
    """POST请求返回的模型"""
    status: str

# 3. Agent 管理器 - 负责状态管理
class AgentManager:
    """管理当前活跃的Agent实例及其配置"""
    _instance = None
    _current_config: Optional[AgentConfigRequestModel] = None
    _max_retries: int = 2

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentManager, cls).__new__(cls)
        return cls._instance

    def update_agent(self, config: AgentConfigRequestModel, max_retries=2):
        """根据新配置更新Agent实例"""
        self._current_config = config
        print("AgentManager: Agent已成功更新。")

    def get_config(self) -> Optional[AgentConfigResponseModel]:
        """获取当前的配置"""
        if self._current_config is None:
            return None

        # 创建 AgentConfigResponseModel 实例，只包含 llms 字段
        return AgentConfigResponseModel(llms=self._current_config.llms)

    def get_config_dict(self) -> dict | None:
        if self._current_config is None:
            return None

        self.dump = AgentConfigResponseModel(llms=self._current_config.llms).model_dump()
        return self.dump


    def get_agent_dict(self, max_retries: int = 2) -> dict:
        agent_llm_configs = []
        for llm_config in self._current_config.llms:
            llm_instance = ChatOpenAI(
                api_key=llm_config.api_key,
                base_url=llm_config.base_url,
                model_name=llm_config.model_name,
                temperature=llm_config.temperature,
                extra_body=llm_config.extra_body
            )
            # 从 extra_config 中读取并发和速率限制
            extra_config = llm_config.extra_config or {}
            agent_llm_configs.append((llm_instance, extra_config))

        return {
            "llm_configs": agent_llm_configs,
            "max_retries": max_retries
        }