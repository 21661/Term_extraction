import random
import logging
import threading
from typing import Dict, List
from contextlib import contextmanager

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.LLMManager import AgentManager

logger = logging.getLogger(__name__)


class LLMClientManager:
    _instance = None

    # 存储实际的客户端对象: { "config_id_name": ClientInstance }
    _clients: Dict[str, ChatOpenAI] = {}
    _async_clients: Dict[str, ChatOpenAI] = {}

    # 核心索引：模型名 -> 配置ID列表
    # 例: { "MiniMax-M2": ["account_1", "account_2"], "GLM-4": ["zhipu_main"] }
    _model_index: Dict[str, List[str]] = {}

    # 并发控制
    _concurrency_limits: Dict[str, int] = {}
    _active_counts: Dict[str, int] = {}

    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.agent_manager = AgentManager()
        return cls._instance

    def reset_clients(self):
        with self._lock:
            self._clients.clear()
            self._async_clients.clear()
            self._model_index.clear()  # 清空索引
            self._concurrency_limits.clear()
            self._active_counts.clear()
            logger.info("LLMClientManager: 状态已重置")

    def _initialize_clients(self):
        """根据配置初始化，并建立 [模型名 -> 配置ID] 的索引"""
        if self._clients: return

        config = self.agent_manager.get_config()
        if not config or not config.llms:
            return

        with self._lock:
            if self._clients: return

            for llm_config in config.llms:
                try:
                    # 1. 唯一标识符 (用户配置的 name, 如 "zhipu-1")
                    client_id = llm_config.name
                    # 2. 实际模型名 (如 "GLM-4.5-Flash")
                    real_model_name = llm_config.model_name

                    # 初始化 LangChain 客户端
                    client = ChatOpenAI(
                        model=real_model_name,
                        api_key=llm_config.api_key,
                        base_url=llm_config.base_url,
                        temperature=llm_config.temperature or 0.2,
                        extra_body=llm_config.extra_body,
                    )

                    self._clients[client_id] = client
                    self._async_clients[client_id] = client

                    # --- 核心：构建索引 ---
                    if real_model_name not in self._model_index:
                        self._model_index[real_model_name] = []
                    self._model_index[real_model_name].append(client_id)
                    # -------------------

                    # 初始化并发限制
                    max_concurrency = 1
                    if llm_config.extra_config:
                        max_concurrency = llm_config.extra_config.get("max_concurrency", 1)
                    self._concurrency_limits[client_id] = int(max_concurrency)
                    self._active_counts[client_id] = 0

                except Exception as e:
                    logger.error(f"初始化失败 [{llm_config.name}]: {e}")

    def check_model_exists(self, model_name: str) -> bool:
        """
        检查是否存在指定 model_name 的配置 (精确匹配)。
        :param model_name: 模型名称，如 "GLM-4.5-Flash"
        :return: True / False
        """
        # 1. 确保系统已初始化 (因为是懒加载)
        if not self._clients:
            self._initialize_clients()

        with self._lock:
            # _model_index 的 Key 就是 JSON 中的 model_name
            return model_name in self._model_index
    def _get_available_client_id(self, target_model: str = None) -> str:
        """
        根据传入的模型名，查找可用的 client_id
        """
        if not self._clients:
            self._initialize_clients()

        with self._lock:
            candidates = []

            # 1. 确定候选池
            if target_model:
                # 如果指定了模型名，只从该模型的配置中找
                candidates = self._model_index.get(target_model, [])
                if not candidates:
                    # 如果没找到该模型，抛出明确错误
                    available_models = list(self._model_index.keys())
                    raise ValueError(f"未找到模型 '{target_model}' 的配置。当前可用模型: {available_models}")
            else:
                # 如果没指定模型，从所有配置中随机 (混用模式)
                candidates = list(self._clients.keys())

            # 2. 筛选未满载的 ID
            valid_candidates = []
            for cid in candidates:
                limit = self._concurrency_limits.get(cid, 1)
                current = self._active_counts.get(cid, 0)
                if current < limit:
                    valid_candidates.append(cid)

            if not valid_candidates:
                msg = f"模型 '{target_model}'" if target_model else "系统"
                raise RuntimeError(f"{msg} 所有实例均已满载 (Busy)，请稍后重试。")

            # 3. 随机负载均衡
            selected_id = random.choice(valid_candidates)

            # 4. 增加计数
            self._active_counts[selected_id] += 1
            return selected_id

    def _release_client_id(self, client_id: str):
        with self._lock:
            if client_id in self._active_counts and self._active_counts[client_id] > 0:
                self._active_counts[client_id] -= 1

    @contextmanager
    def _acquire_client_context(self, model: str = None, specific_client_id: str = None):
        """
        上下文管理器：负责获取资源 -> Yield Client -> 自动释放
        """
        selected_id = None

        # 优先级 1: 强制指定具体配置ID (调试用)
        if specific_client_id:
            if not self._clients: self._initialize_clients()
            selected_id = specific_client_id
            with self._lock:
                self._active_counts[selected_id] = self._active_counts.get(selected_id, 0) + 1

        # 优先级 2: 指定模型名 (常用，自动负载均衡)
        # 优先级 3: 什么都不传 (随机)
        else:
            selected_id = self._get_available_client_id(target_model=model)

        try:
            client = self._clients.get(selected_id)
            if not client:
                raise RuntimeError(f"内部错误: 找不到 ID 为 '{selected_id}' 的客户端实例")
            yield client
        finally:
            self._release_client_id(selected_id)

    def _convert_messages(self, messages: list):
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
        return lc_msgs

    # ==========================================
    # Public API
    # ==========================================

    def chat(self, messages: list, model: str = None, client_name: str = None):
        """
        同步对话接口
        :param model: (推荐) 目标模型名称，如 "GLM-4.5-Flash"。系统会自动在配置了该模型的所有Key中负载均衡。
        :param client_name: (可选) 强制指定某个配置的 name。
        """
        lc_msgs = self._convert_messages(messages)
        # 将参数传给 Context Manager 处理选择逻辑
        with self._acquire_client_context(model=model, specific_client_id=client_name) as client:
            return client.invoke(lc_msgs)

    async def achat(self, messages: list, model: str = None, client_name: str = None):
        """
        异步对话接口
        """
        if not self._async_clients:
            self._initialize_clients()

        lc_msgs = self._convert_messages(messages)

        # 手动管理生命周期
        selected_id = None
        if client_name:
            selected_id = client_name
            with self._lock:
                self._active_counts[selected_id] = self._active_counts.get(selected_id, 0) + 1
        else:
            # 这里传入 model，根据模型名选择 ID
            selected_id = self._get_available_client_id(target_model=model)

        try:
            client = self._async_clients.get(selected_id)
            if not client:
                raise RuntimeError(f"Client {selected_id} not found")
            return await client.ainvoke(lc_msgs)
        finally:
            self._release_client_id(selected_id)


LLMclientManager = LLMClientManager()