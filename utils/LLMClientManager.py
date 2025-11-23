import random
import logging
import threading
import asyncio
import time
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

    # --- 新增: RPM 限制数据结构 ---
    _rpm_limits: Dict[str, int] = {}  # ClientID -> RPM限制值
    _request_timestamps: Dict[str, List[float]] = {}  # ClientID -> 请求时间戳历史

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
            # 重置 RPM 相关数据
            self._rpm_limits.clear()
            self._request_timestamps.clear()
            logger.info("LLMClientManager: 状态已重置")

    def _initialize_clients(self):
        """
        初始化客户端，解析配置中的并发和RPM限制
        """
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

                    # 1. 创建 LangChain 客户端
                    # 设置 max_retries=0，禁止底层自动重试，交由上层换号逻辑处理
                    client = ChatOpenAI(
                        model=real_model_name,
                        api_key=llm_config.api_key,
                        base_url=llm_config.base_url,
                        temperature=llm_config.temperature or 0.2,
                        extra_body=llm_config.extra_body,
                        max_retries=0,
                    )

                    self._clients[client_id] = client
                    self._async_clients[client_id] = client

                    if real_model_name not in self._model_index:
                        self._model_index[real_model_name] = []
                    self._model_index[real_model_name].append(client_id)

                    # 2. 识别 MT 模型 (修复点：同时检查 extra_config 和 extra_body)
                    is_mt = False
                    # 检查 extra_config (如你的 Hunyuan 配置)
                    if llm_config.extra_config and llm_config.extra_config.get("type") == "MT":
                        is_mt = True

                    if is_mt:
                        self._mt_client_ids.add(client_id)

                    # 3. 解析限制参数 (修复点：处理 None 值防止 int() 报错)
                    max_concurrency = 1
                    rpm_limit = 0  # 0 表示不限速

                    if llm_config.extra_config:
                        # 安全获取并发限制
                        mc_val = llm_config.extra_config.get("max_concurrency")
                        if mc_val is not None:
                            max_concurrency = int(mc_val)

                        # 安全获取 RPM 限制
                        rpm_val = llm_config.extra_config.get("rpm")
                        if rpm_val is not None:
                            rpm_limit = int(rpm_val)

                    # 4. 存入状态
                    self._concurrency_limits[client_id] = max_concurrency
                    self._active_counts[client_id] = 0

                    self._rpm_limits[client_id] = rpm_limit
                    self._request_timestamps[client_id] = []

                except Exception as e:
                    logger.error(f"初始化失败 [{llm_config.name}]: {e}")

    def check_model_exists(self, model_name: str) -> bool:
        if not self._clients: self._initialize_clients()
        with self._lock:
            return model_name in self._model_index

    def _get_available_client_id(self, target_model: str = None, exclude_ids: List[str] = None) -> str:
        """
        选择一个可用的 Client ID。
        逻辑：
        1. 筛选符合模型要求的 ID。
        2. 排除 MT 模型（除非指定）。
        3. 排除 exclude_ids。
        4. 检查并发限制 (max_concurrency)。
        """
        if not self._clients: self._initialize_clients()

        with self._lock:
            candidates = []

            if target_model:
                # 指定模型：从索引中取
                candidates = self._model_index.get(target_model, [])
                if not candidates:
                    available = list(self._model_index.keys())
                    raise ValueError(f"未找到模型 '{target_model}'。可用: {available}")
            else:
                # 未指定模型：使用所有非 MT 模型
                candidates = [
                    cid for cid in self._clients.keys()
                    if cid not in self._mt_client_ids
                ]

            # 排除刚才失败的 ID
            if exclude_ids:
                filtered = [c for c in candidates if c not in exclude_ids]
                if filtered:
                    candidates = filtered

            # 检查并发限制
            valid_candidates = []
            for cid in candidates:
                limit = self._concurrency_limits.get(cid, 1)
                current = self._active_counts.get(cid, 0)
                if current < limit:
                    valid_candidates.append(cid)

            if not valid_candidates:
                msg = f"模型 '{target_model}'" if target_model else "通用模型池"
                raise RuntimeError(f"{msg} 所有实例均已满载(并发限制)，请稍后重试。")

            # 随机选择一个
            selected_id = random.choice(valid_candidates)
            self._active_counts[selected_id] += 1
            return selected_id

    def _release_client_id(self, client_id: str):
        with self._lock:
            if client_id in self._active_counts and self._active_counts[client_id] > 0:
                self._active_counts[client_id] -= 1

    # --- RPM 计算核心逻辑 ---
    def _consume_rpm_token(self, client_id: str) -> float:
        """
        检查 RPM 限制并消耗一个令牌。
        返回需要等待的时间(秒)。
        """
        rpm = self._rpm_limits.get(client_id, 0)
        # 如果 rpm 为 0 或负数，表示不限制
        if rpm <= 0:
            return 0.0

        with self._lock:
            history = self._request_timestamps.get(client_id, [])
            now = time.time()

            # 1. 清理超过 60 秒的历史记录
            # history 是有序递增的
            while history and now - history[0] > 60:
                history.pop(0)

            wait_time = 0.0

            # 2. 如果当前窗口内请求数已达上限
            if len(history) >= rpm:
                # 最早的一次请求时间
                oldest_request_time = history[0]
                # 计算该请求过期还需要多久
                # 例如：最早请求是 10:00:00，现在是 10:00:55，限制是 60s
                # wait_time = 60 - (55 - 0) = 5秒
                wait_time = 60 - (now - oldest_request_time)
                if wait_time < 0: wait_time = 0

            # 3. 记录本次请求的时间
            # 如果需要等待，我们记录的是“未来执行的时间点”，确保后续请求排在它后面
            expected_exec_time = now + wait_time
            history.append(expected_exec_time)

            # 更新状态
            self._request_timestamps[client_id] = history

            return wait_time

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
    # Public API
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
                # 获取 Client 上下文（已包含并发+1）
                with self._acquire_client_context(model=model, specific_client_id=client_name,
                                                  exclude_ids=failed_ids) as (client, cid):
                    current_id = cid

                    # --- RPM 限制检查 ---
                    wait_time = self._consume_rpm_token(cid)
                    if wait_time > 0:
                        logger.info(f"Client {cid} RPM限制触发，等待 {wait_time:.2f}s")
                        time.sleep(wait_time)
                    # ------------------

                    return client.invoke(lc_msgs)

            except Exception as e:
                if current_id: failed_ids.append(current_id)
                logger.warning(f"Chat重试 {attempt + 1}/{max_retries + 1} [ID:{current_id}]: {e}")
                # 如果指定了 client_name 或者重试耗尽，抛出异常
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
                # 1. 获取 ID 并增加并发计数
                if client_name:
                    current_id = client_name
                    with self._lock:
                        self._active_counts[current_id] = self._active_counts.get(current_id, 0) + 1
                else:
                    current_id = self._get_available_client_id(target_model=model, exclude_ids=failed_ids)

                # 2. 获取 Client 对象
                client = self._async_clients.get(current_id)
                if not client:
                    self._release_client_id(current_id)
                    raise RuntimeError(f"Client {current_id} not found")

                try:
                    # --- RPM 限制检查 (异步) ---
                    wait_time = self._consume_rpm_token(current_id)
                    if wait_time > 0:
                        logger.info(f"Client {current_id} RPM限制触发，异步等待 {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
                    # -------------------------

                    # 3. 执行请求
                    res = await client.ainvoke(lc_msgs)
                    return res
                finally:
                    # 确保无论成功失败，都释放并发计数
                    self._release_client_id(current_id)

            except Exception as e:
                if current_id:
                    if current_id not in failed_ids:
                        failed_ids.append(current_id)

                logger.warning(f"AChat重试 {attempt + 1}/{max_retries + 1} [ID:{current_id}]: {e}")
                if client_name or attempt == max_retries: raise e

                # 失败后简短避让
                await asyncio.sleep(0.5)


LLMclientManager = LLMClientManager()