# web_fetcher.py
import requests
from bs4 import BeautifulSoup
import time
import logging

logger = logging.getLogger(__name__)


def fetch_core_translations(term_en, retries=2, backoff=1.0, timeout=5):
    """
    抓取有道词典核心中文翻译
    只返回简明翻译，去掉词性、例句、文章内容等
    """
    url = f"https://dict.youdao.com/w/{term_en}/"
    headers = {"User-Agent": "Mozilla/5.0"}
    attempt = 0
    while attempt <= retries:
        attempt += 1
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code != 200:
                logger.warning("fetch_core_translations got status %d for %s", r.status_code, term_en)
                return []

            soup = BeautifulSoup(r.text, "html.parser")
            translations = []

            # 基本翻译 <ul><li>
            for li in soup.select(".trans-container ul li"):
                text = li.get_text(strip=True)
                if not text:
                    continue
                # 去掉词性标注 n. v. 等
                if '.' in text[:3]:
                    text = text.split('.', 1)[-1].strip()
                # 分号拆开多译
                for t in text.split('；'):
                    t = t.strip()
                    # 只保留中文词条
                    if t and all('\u4e00' <= ch <= '\u9fff' or ch in '·•' for ch in t):
                        translations.append(t)

            # 去重
            return list(dict.fromkeys(translations))
        except Exception as e:
            logger.warning("抓取失败 (attempt %d) for %s: %s", attempt, term_en, e)
            if attempt > retries:
                return []
            time.sleep(backoff * attempt)
    return []
