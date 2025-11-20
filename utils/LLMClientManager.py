from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Dict
from utils.LLMManager import AgentManager
import logging

logger = logging.getLogger(__name__)

class LLMClientManager:
    _instance = None
    _clients: Dict[str, ChatOpenAI] = {}
    _async_clients: Dict[str, ChatOpenAI] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.agent_manager = AgentManager()
        return cls._instance

    def _initialize_clients(self):
        if self._clients:
            return

        config = self.agent_manager.get_config()
        if not config or not config.llms:
            logger.warning("AgentManager 中没有 LLM 配置")
            return

        for llm_config in config.llms:
            try:
                # 修复：将 extra_body 传入 ChatOpenAI
                # 注意：LangChain 的 ChatOpenAI 接收 model_kwargs 或 extra_body
                client = ChatOpenAI(
                    model=llm_config.model_name,
                    api_key=llm_config.api_key,
                    base_url=llm_config.base_url,
                    temperature=llm_config.temperature or 0.2,
                    extra_body=llm_config.extra_body,  # <--- 添加这一行
                    # 如果你需要将 extra_config 中的并发限制应用到客户端，
                    # 需在此处处理，或者在调用 chat() 时外部控制
                )

                self._clients[llm_config.name] = client
                self._async_clients[llm_config.name] = client

            except Exception as e:
                logger.error(f"初始化 LangChain ChatOpenAI 失败: {e}")

    def chat(self, client_name: str, model: str, messages: list):
        if not self._clients:
            self._initialize_clients()

        client = self._clients.get(client_name)
        if not client:
            raise RuntimeError(f"找不到客户端 {client_name}")

        # 转换 openai 风格消息
        lc_msgs = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role == "user":
                lc_msgs.append(HumanMessage(content=content))
            else:
                lc_msgs.append(AIMessage(content=content))

        return client.invoke(lc_msgs)

    async def achat(self, client_name: str, model: str, messages: list):
        if not self._async_clients:
            self._initialize_clients()

        client = self._async_clients.get(client_name)
        if not client:
            raise RuntimeError(f"找不到 async 客户端 {client_name}")

        lc_msgs = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role == "user":
                lc_msgs.append(HumanMessage(content=content))
            else:
                lc_msgs.append(AIMessage(content=content))

        return await client.ainvoke(lc_msgs)

LLMclientManager = LLMClientManager()
