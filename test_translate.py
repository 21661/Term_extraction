import logging
from utils.Get_term import translate_term

logging.basicConfig(level=logging.DEBUG)

terms = [
    "cross-modal adaptive fusion module",
    "modified multi-head self-attention mechanism",
    "multi-scale supervision strategy",
]

for t in terms:
    print('\n=== Translating:', t)
    cands = translate_term(t)
    print('Result:', cands)

