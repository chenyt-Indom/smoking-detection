"""连接星辰大模型的最小探针：先列模型，再试一次对话。

用法
----
    # 1) 把平台给的 AppKey 放进环境变量（不要写进代码）
    set TELEAI_APPKEY=你的AppKey          # Windows cmd
    $env:TELEAI_APPKEY="你的AppKey"       # PowerShell

    # 2) 列模型
    python scripts/probe_teleai.py

    # 3) 指定模型试对话
    python scripts/probe_teleai.py --model 某个模型名

    # 4) 顺带试一次图片输入（不加 --image 就不测多模态）
    python scripts/probe_teleai.py --model 多模态模型名 --image 某张图.jpg

为什么用 requests 而不是 openai SDK
-----------------------------------
这个接口是 **OpenAI 兼容**的（实测未带 key 时返回
`{"error":{"message":"ApiKey不正确...","code":"api_key_not_found"}}`，
标准 OpenAI 错误结构），所以手写 requests 完全够用，少装一个依赖。

⚠️ 一个坑：官方文档的 openai SDK 示例把 base_url 写成
`https://ai.ctaigw.cn/v1/models`，那是**错的** —— SDK 会在 base_url 后面
再拼 `/models`，变成 `/v1/models/models`。base_url 只能是 `https://ai.ctaigw.cn/v1`。

⚠️ 另一个坑：请求头必须带 User-Agent，不然可能被安全护栏直接拦掉，
返回的东西跟鉴权失败长得一模一样，会误判成 key 错了。
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import requests

# 算力归属中国大陆的模型；全球节点是 https://ai.ctaigw.com/{region}/v1（amer/eu/as）
BASE_URL = os.environ.get("TELEAI_BASE_URL", "https://ai.ctaigw.cn/v1")

HEADERS_COMMON = {
    # 这两个是必须的。User-Agent 缺了会被护栏拦，且报错很像鉴权失败
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Content-Type": "application/json",
    "Accept": "*/*",
}


def _headers(appkey: str) -> dict:
    return {**HEADERS_COMMON, "Authorization": f"Bearer {appkey}"}


def list_models(appkey: str) -> list[dict]:
    """GET /v1/models —— 先确认 key 能用，再看有哪些模型可选。"""
    url = f"{BASE_URL}/models"
    print(f"→ GET {url}")
    r = requests.get(url, headers=_headers(appkey), timeout=20)
    print(f"  HTTP {r.status_code}")
    if r.status_code != 200:
        print(f"  返回：{r.text[:800]}")
        return []
    data = r.json().get("data", [])
    print(f"  共 {len(data)} 个模型：")
    for m in data:
        mid = m.get("id", "?")
        print(f"    · {mid}")
    return data


def chat(appkey: str, model: str, image: str | None = None) -> None:
    """POST /v1/chat/completions —— 纯文本；给了 --image 就走多模态消息体。"""
    url = f"{BASE_URL}/chat/completions"

    content = "用一句话说明你是什么模型。"
    if image:
        with open(image, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(image)[1].lstrip(".").lower() or "jpeg"
        # 多模态走 OpenAI 标准的 image_url + data URI 形式
        content = [
            {"type": "text", "text": "这张图里有什么？一句话说明。"},
            {"type": "image_url",
             "image_url": {"url": f"data:image/{ext};base64,{b64}"}},
        ]

    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": 300,
    }
    print(f"\n→ POST {url}  model={model}  多模态={'是' if image else '否'}")
    r = requests.post(url, headers=_headers(appkey), data=json.dumps(body), timeout=60)
    print(f"  HTTP {r.status_code}")
    if r.status_code != 200:
        print(f"  返回：{r.text[:1200]}")
        return
    d = r.json()
    try:
        print(f"  回复：{d['choices'][0]['message']['content']}")
    except (KeyError, IndexError):
        print(f"  结构不认识，原样打印：{json.dumps(d, ensure_ascii=False)[:1200]}")
    usage = d.get("usage")
    if usage:
        print(f"  用量：{usage}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--appkey", default=os.environ.get("TELEAI_APPKEY", ""),
                    help="平台 AppKey；也可以用环境变量 TELEAI_APPKEY")
    ap.add_argument("--model", default="", help="模型名；不给就只列模型")
    ap.add_argument("--image", default="", help="图片路径，给了就测多模态")
    args = ap.parse_args()

    if not args.appkey:
        print("没有 AppKey。先设置环境变量 TELEAI_APPKEY，或用 --appkey 传入。")
        print("取法：天翼云控制台 → 星辰TokenHub运营服务平台 → 系统管理 →"
              " APP Key 管理 → 创建服务组 → 绑定服务 → 获取 AppKey")
        return 2

    models = list_models(args.appkey)
    if args.model:
        chat(args.appkey, args.model, args.image or None)
    elif models:
        print("\n（用 --model 指定上面任一个模型名，就能试一次对话）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
