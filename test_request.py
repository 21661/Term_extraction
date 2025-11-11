import requests
import json
import sys
from pathlib import Path

# ========== 配置部分 ==========
# 你可以手动写，也可以通过命令行参数传入
json_path = Path(r"E:\Term_extraction\Source\Test1.json")
url = "http://127.0.0.1:8000/extract"

# ========== 读取 JSON 文件 ==========
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ========== 发送 POST 请求 ==========
response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    data=json.dumps(data),
    timeout=120
)

# ========== 打印返回结果 ==========
try:
    print(response.json())
except Exception:
    print("Response text:", response.text)
