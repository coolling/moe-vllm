# chat.py
import os
import requests
import json

# === 强制清除所有代理相关环境变量 ===
proxy_keys = [
    'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
    'ALL_PROXY', 'all_proxy', 'no_proxy', 'NO_PROXY'
]
for key in proxy_keys:
    os.environ.pop(key, None)

# 显式设置 NO_PROXY（对 localhost 生效）
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

# === 请求参数 ===
url = "http://localhost:8000/v1/chat/completions"
payload = {
    "model": "/mnt/nvme0/home/chenyunling/models/Isotonic/smol_llama-4x220M-MoE",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 100,
    "temperature": 0,
    "chat_template": "{% for message in messages %}{% if message['role'] == 'user' %}<s>[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>{% endif %}{% endfor %}"
}

try:
    # 关键：显式禁用代理
    response = requests.post(
        url,
        json=payload,
        timeout=3000,
        proxies={"http": None, "https": None}  # 👈 强制 bypass 代理
    )
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

except requests.exceptions.RequestException as e:
    print("❌ 请求失败:", e)
    if e.response is not None:
        print("状态码:", e.response.status_code)
        print("响应:", repr(e.response.text))
    else:
        print("无响应 —— 极可能是代理干扰或连接被重置")