from functools import wraps
import logging
import time
import asyncio
import inspect
from utils.workflow_adapter import _unwrap

logger = logging.getLogger(__name__)

def timed_node(name: str = None):
    """
    Decorator that times a node function. Works for both sync and async functions.
    Usage:
      @timed_node()
      def foo(...): ...

      @timed_node
      async def bar(...): ...
    """
    # Support bare decorator usage: @timed_node
    if callable(name) and not isinstance(name, str):
        func = name
        node_name = func.__name__
        return _make_timed_wrapper(func, node_name)

    def decorator(func):
        node_name = name or func.__name__
        return _make_timed_wrapper(func, node_name)

    return decorator


def _make_timed_wrapper(func, node_name: str):
    if inspect.iscoroutinefunction(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger.info("Node %s START", node_name)
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                _log_end(node_name, start, args)
        return async_wrapper
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info("Node %s START", node_name)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                _log_end(node_name, start, args)
        return wrapper


def _log_end(node_name: str, start: float, args):
    end = time.perf_counter()
    dur = end - start
    try:
        state = args[0] if args else None
        inner, parent, key = _unwrap(state) if state is not None else (None, None, None)
        sd = inner if isinstance(inner, dict) else (state if isinstance(state, dict) else {})
        cands = sd.get("candidates") if isinstance(sd, dict) else None
        terms = sd.get("terms") if isinstance(sd, dict) else None
        trans = sd.get("translations") if isinstance(sd, dict) else None
        logger.info(
            "Node %s END (%.3fs) candidates=%s terms=%s translations=%s",
            node_name,
            dur,
            (len(cands) if isinstance(cands, (list, set)) else '-'),
            (len(terms) if isinstance(terms, (list, set)) else '-'),
            (len(trans) if isinstance(trans, dict) else '-')
        )
    except Exception:
        logger.info("Node %s END (%.3fs)", node_name, dur)
