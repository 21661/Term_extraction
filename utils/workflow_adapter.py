from typing import Any, MutableMapping, Tuple, Optional


def unwrap_state(raw: Any) -> Tuple[MutableMapping, Optional[Any], Optional[Any]]:
    """
    将 workflow 可能传入的状态格式标准化为 (inner, parent, key)。

    语义兼容：
    - 如果 raw 是 (inner, parent, key) 三元组，按原样拆开并返回。
    - 如果 raw 是 dict，则返回 (raw, None, None)。
    - 其他类型（如 None 或字符串等）视为空状态，返回 ({"text": ""}, None, None)。

    返回：
        (inner_dict, parent, key)
    """
    # 三元组路径：常见框架传入 (inner, parent, key)
    if isinstance(raw, tuple) and len(raw) == 3:
        inner, parent, key = raw
        # 保证 inner 至少为 dict-like
        if isinstance(inner, dict):
            return inner, parent, key
        # 如果 inner 不是 dict，但是 None/其他，仍返回原样以保留语义
        return inner, parent, key

    # 直接 dict：直接返回
    if isinstance(raw, dict):
        return raw, None, None

    # 其它情况：统一降级为空状态 dict，避免调用方频繁检查 None
    return {"text": ""}, None, None


def rewrap_state(original_raw: Any, parent: Any, key: Any, new_inner: MutableMapping) -> Any:
    """
    将处理后的 new_inner 重新包装为与 original_raw 相同的结构。

    行为：
    - 如果 original_raw 最初是三元组，返回 (new_inner, parent, key)
    - 否则直接返回 new_inner

    这保证替换实现后外部调用代码可以继续像之前一样使用返回值。
    """
    if isinstance(original_raw, tuple) and len(original_raw) == 3:
        return (new_inner, parent, key)
    return new_inner


# 向后兼容旧名（若你的库或其他文件直接 import _unwrap/_rewrap）
def _unwrap(raw: Any) -> Tuple[MutableMapping, Optional[Any], Optional[Any]]:
    """兼容旧名：委托给 unwrap_state"""
    return unwrap_state(raw)


def _rewrap(original_raw: Any, parent: Any, key: Any, new_inner: MutableMapping) -> Any:
    """兼容旧名：委托给 rewrap_state"""
    return rewrap_state(original_raw, parent, key, new_inner)