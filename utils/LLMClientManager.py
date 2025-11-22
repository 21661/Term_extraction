import random
import logging
import threading
import asyncio
from typing import Dict, List, Set
from contextlib import contextmanager

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from utils.LLMManager import AgentManager

logger = logging.getLogger(__name__)


class LLMClientManager:
    _instance = None

    # 存储实际的客户端对象
    _clients: Dict[str, ChatOpenAI] = {}
    _async_clients: Dict[str, ChatOpenAI] = {}

    # 核心索引：模型名 -> 配置ID列表
    _model_index: Dict[str, List[str]] = {}

    # MT 模型 ID 集合 (用于随机选择时排除)
    _mt_client_ids: Set[str] = set()

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
            self._model_index.clear()
            self._concurrency_limits.clear()
            self._active_counts.clear()
            self._mt_client_ids.clear()
            logger.info("LLMClientManager: 状态已重置")

    def _initialize_clients(self):
        if self._clients: return

        config = self.agent_manager.get_config()
        if not config or not config.llms:
            return

        with self._lock:
            if self._clients: return

            for llm_config in config.llms:
                try:
                    client_id = llm_config.name
                    real_model_name = llm_config.model_name

                    # 设置 max_retries=0，禁止底层自动重试，交由上层换号逻辑处理
                    client = ChatOpenAI(
                        model=real_model_name,
                        api_key=llm_config.api_key,
                        base_url=llm_config.base_url,
                        temperature=llm_config.temperature or 0.2,
                        extra_body=llm_config.extra_body,
                        max_retries=0,
                        request_timeout=60
                    )

                    self._clients[client_id] = client
                    self._async_clients[client_id] = client

                    if real_model_name not in self._model_index:
                        self._model_index[real_model_name] = []
                    self._model_index[real_model_name].append(client_id)

                    # 识别 MT 模型
                    extra_body = llm_config.extra_body or {}
                    if extra_body.get("type") == "MT":
                        self._mt_client_ids.add(client_id)

                    # 并发限制
                    max_concurrency = 1
                    if llm_config.extra_config:
                        max_concurrency = llm_config.extra_config.get("max_concurrency", 1)
                    self._concurrency_limits[client_id] = int(max_concurrency)
                    self._active_counts[client_id] = 0

                except Exception as e:
                    logger.error(f"初始化失败 [{llm_config.name}]: {e}")

    def check_model_exists(self, model_name: str) -> bool:
        if not self._clients: self._initialize_clients()
        with self._lock:
            return model_name in self._model_index

    def _get_available_client_id(self, target_model: str = None, exclude_ids: List[str] = None) -> str:
        if not self._clients: self._initialize_clients()

        with self._lock:
            candidates = []

            if target_model:
                # 指定模型：允许选中 MT (只要显式指定)
                candidates = self._model_index.get(target_model, [])
                if not candidates:
                    available = list(self._model_index.keys())
                    raise ValueError(f"未找到模型 '{target_model}'。可用: {available}")
            else:
                # 未指定模型：排除 MT
                candidates = [
                    cid for cid in self._clients.keys()
                    if cid not in self._mt_client_ids
                ]

            # 排除刚才失败的 ID
            if exclude_ids:
                filtered = [c for c in candidates if c not in exclude_ids]
                if filtered:
                    candidates = filtered

            valid_candidates = []
            for cid in candidates:
                limit = self._concurrency_limits.get(cid, 1)
                current = self._active_counts.get(cid, 0)
                if current < limit:
                    valid_candidates.append(cid)

            if not valid_candidates:
                msg = f"模型 '{target_model}'" if target_model else "通用模型池"
                raise RuntimeError(f"{msg} 所有实例均已满载，请稍后重试。")

            selected_id = random.choice(valid_candidates)
            self._active_counts[selected_id] += 1
            return selected_id

    def _release_client_id(self, client_id: str):
        with self._lock:
            if client_id in self._active_counts and self._active_counts[client_id] > 0:
                self._active_counts[client_id] -= 1

    @contextmanager
    def _acquire_client_context(self, model: str = None, specific_client_id: str = None, exclude_ids: List[str] = None):
        selected_id = None

        if specific_client_id:
            if not self._clients: self._initialize_clients()
            selected_id = specific_client_id
            with self._lock:
                self._active_counts[selected_id] = self._active_counts.get(selected_id, 0) + 1
        else:
            selected_id = self._get_available_client_id(target_model=model, exclude_ids=exclude_ids)

        try:
            client = self._clients.get(selected_id)
            if not client:
                raise RuntimeError(f"Client {selected_id} not found")
            yield client, selected_id
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
    # Public API (无 reasoning 参数)
    # ==========================================

    def chat(self, messages: list, model: str = None, client_name: str = None, max_retries: int = 3):
        """
        同步对话接口
        """
        if not self._clients: self._initialize_clients()
        lc_msgs = self._convert_messages(messages)

        failed_ids = []
        for attempt in range(max_retries + 1):
            current_id = None
            try:
                with self._acquire_client_context(model=model, specific_client_id=client_name,
                                                  exclude_ids=failed_ids) as (client, cid):
                    current_id = cid
                    return client.invoke(lc_msgs)
            except Exception as e:
                if current_id: failed_ids.append(current_id)
                logger.warning(f"Chat重试 {attempt + 1}/{max_retries + 1} [ID:{current_id}]: {e}")
                if client_name or attempt == max_retries: raise e

    async def achat(self, messages: list, model: str = None, client_name: str = None, max_retries: int = 3):
        """
        异步对话接口
        """
        if not self._async_clients: self._initialize_clients()
        lc_msgs = self._convert_messages(messages)

        failed_ids = []
        for attempt in range(max_retries + 1):
            current_id = None
            try:
                # 1. 获取 ID
                if client_name:
                    current_id = client_name
                    with self._lock:
                        self._active_counts[current_id] = self._active_counts.get(current_id, 0) + 1
                else:
                    current_id = self._get_available_client_id(target_model=model, exclude_ids=failed_ids)

                # 2. 获取 Client
                client = self._async_clients.get(current_id)
                if not client: raise RuntimeError(f"Client {current_id} not found")

                # 3. 执行
                res = await client.ainvoke(lc_msgs)
                self._release_client_id(current_id)
                return res

            except Exception as e:
                if current_id:
                    self._release_client_id(current_id)
                    failed_ids.append(current_id)

                logger.warning(f"AChat重试 {attempt + 1}/{max_retries + 1} [ID:{current_id}]: {e}")
                if client_name or attempt == max_retries: raise e
                await asyncio.sleep(0.5)


LLMclientManager = LLMClientManager()