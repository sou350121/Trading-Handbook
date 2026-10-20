"""DeepSeek only. No fallback; never log credentials or HTTP response bodies."""
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

MODEL = os.environ.get('TRADING_DEEPSEEK_MODEL', 'deepseek-flash')


def _key():
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if key:
        return key
    path = Path(os.environ.get('TRADING_DEEPSEEK_ENV', '/home/admin/.clawdbot/.env'))
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.startswith('DEEPSEEK_API_KEY='):
                key = line.split('=', 1)[1].strip().strip('\"\'')
                if key:
                    return key
    raise RuntimeError('deepseek_credential_missing')


def ask(prompt, system=None, thinking=True, temperature=0.3, max_tokens=8000,
        json_mode=False, retries=4, timeout=300, model=None):
    selected = model or MODEL
    if selected not in {'deepseek-flash', 'deepseek-v4-pro'}:
        raise ValueError('unsupported_deepseek_model')
    messages = ([{'role': 'system', 'content': system}] if system else [])
    messages.append({'role': 'user', 'content': prompt})
    payload = {'model': selected, 'messages': messages, 'temperature': temperature,
               'max_tokens': max(max_tokens, 16384) if thinking else max_tokens,
               'thinking': {'type': 'enabled' if thinking else 'disabled'}}
    if json_mode:
        payload['response_format'] = {'type': 'json_object'}
    encoded = json.dumps(payload).encode('utf-8')
    key = _key()
    deadline = time.monotonic() + timeout
    error = 'request_failed'
    for attempt in range(max(1, retries)):
        remaining = deadline - time.monotonic()
        if remaining < 1:
            break
        request = urllib.request.Request('https://api.deepseek.com/chat/completions', data=encoded,
            headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request, timeout=remaining) as response:
                obj = json.load(response)
            choice = obj['choices'][0]
            content = choice['message'].get('content', '')
            if choice.get('finish_reason') != 'stop':
                raise RuntimeError('deepseek_incomplete_output')
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError('deepseek_empty_output')
            if json_mode:
                json.loads(content)
            return content
        except urllib.error.HTTPError as exc:
            error = 'http_' + str(exc.code)
            if exc.code not in (429, 500, 502, 503, 504):
                raise RuntimeError('deepseek_' + error) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            error = 'network_error'
        if attempt + 1 < retries:
            time.sleep(min(3 * (attempt + 1), max(0, deadline - time.monotonic())))
    raise RuntimeError('deepseek_' + error) from None
