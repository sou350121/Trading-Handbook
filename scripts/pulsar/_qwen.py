#!/usr/bin/env python3
"""Reusable qwen caller via DashScope CodingPlan OpenAI-compatible endpoint.
Thinking mode ON. Key from BAILIAN_CODING_PLAN_API_KEY env or ~/.qwen/settings.json.
"""
import os, json, time, urllib.request, urllib.error, pathlib

BASE = "https://coding.dashscope.aliyuncs.com/v1"
MODEL = os.environ.get("QWEN_MODEL", "qwen3.5-plus")

def _key():
    k = os.environ.get("BAILIAN_CODING_PLAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if k:
        return k
    p = pathlib.Path.home() / ".qwen" / "settings.json"
    if p.exists():
        s = json.loads(p.read_text())
        return s.get("env", {}).get("BAILIAN_CODING_PLAN_API_KEY")
    raise RuntimeError("no qwen API key found")

def ask(prompt, system=None, thinking=True, temperature=0.3, max_tokens=8000,
        json_mode=False, retries=4, timeout=300, model=None):
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    payload = {
        "model": model or MODEL,
        "messages": msgs,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "extra_body": {"enable_thinking": bool(thinking)},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    data = json.dumps(payload).encode("utf-8")
    key = _key()
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                f"{BASE}/chat/completions", data=data,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"}, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, KeyError) as e:
            last = e
            code = getattr(e, "code", "n/a")
            try:
                err_body = e.read().decode("utf-8")[:300]
            except Exception:
                err_body = ""
            print(f"  WARN qwen attempt {i+1} code={code}: {e} {err_body}")
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"qwen failed after {retries}: {last}")

if __name__ == "__main__":
    import sys
    print(ask(sys.argv[1] if len(sys.argv)>1 else "用一句话自我介绍"))
