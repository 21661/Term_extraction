from typing import TypedDict, List, Dict, Any
from typing_extensions import Annotated
from langgraph.graph import add_messages


class TermState(TypedDict, total=False):
    candidates: List[str] # 初步筛选得到的候选词，_terms_only_batch

    reflected: bool
    reflect_reason: Annotated[List[str], add_messages]
    reflect_attempts: int
    reflect_remove_terms: List[str]

    selected_terms: List[str] #LLM筛选后的术语 _select_top_terms
    summary: str

    chunks: Dict[str, str]
    chunk_terms: Dict[str, List[str]] # 初步筛选得到的候选词，_terms_only_batch ，后续LLM筛选后会更新，_select_top_terms
    term_to_chunks: Dict[str, List[str]] #初步筛选保留的映射关系，让术语能映射回各个 chunk，_terms_only_batch

    unique_terms: List[str]
    translations_map: Dict[str, List[str]] #翻译得到的map
    term_annotations: Dict[str, Any]


