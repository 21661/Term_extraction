from functools import wraps
import logging
import time
from utils.workflow_adapter import _unwrap

logger = logging.getLogger(__name__)
def timed_node(name: str = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            node_name = name or func.__name__
            logger.info("Node %s START", node_name)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                end = time.perf_counter()
                dur = end - start
                # Try to extract counts from the first arg if it's the workflow state
                try:
                    state = args[0] if args else None
                    inner, parent, key = _unwrap(state) if state is not None else (None, None, None)
                    sd = inner if isinstance(inner, dict) else (state if isinstance(state, dict) else {})
                    cands = sd.get("candidates") if isinstance(sd, dict) else None
                    terms = sd.get("terms") if isinstance(sd, dict) else None
                    trans = sd.get("translations") if isinstance(sd, dict) else None
                    logger.info("Node %s END (%.3fs) candidates=%s terms=%s translations=%s", node_name, dur,
                                (len(cands) if isinstance(cands, (list, set)) else '-' ),
                                (len(terms) if isinstance(terms, (list, set)) else '-' ),
                                (len(trans) if isinstance(trans, dict) else '-' ))
                except Exception:
                    logger.info("Node %s END (%.3fs)", node_name, dur)
        return wrapper
    return decorator