# zhipu_client.py
from openai import OpenAI
import os
import logging

logger = logging.getLogger(__name__)

API_KEY = os.environ.get('ZHIPU_API_KEY', "3a1caa7e1196474b9e93ecca12e4ee93.mxHhWISC0MO7tE7p")
BASE_URL = os.environ.get('ZHIPU_BASE_URL', "https://open.bigmodel.cn/api/paas/v4/")

if 'ZHIPU_API_KEY' not in os.environ:
    logger.warning("Using embedded default API key in utils/zhipu_client.py; set ZHIPU_API_KEY env var to override")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)
