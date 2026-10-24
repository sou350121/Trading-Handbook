"""Best-effort, content-free receipts for individual DeepSeek HTTP attempts."""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import sys
import time
import uuid

_MODELS = frozenset(('deepseek-flash', 'deepseek-v4-pro', 'deepseek-v4-flash',
                     'deepseek-v4.1-flash', 'deepseek-v4.1-pro'))
_STOPS = frozenset(('stop', 'length', 'content_filter', 'tool_calls',
                    'insufficient_system_resource', 'end_turn', 'max_tokens',
                    'tool_use', 'stop_sequence'))
_MAX_INTEGER = 9007199254740991


def new_call_id():
    try:
        return uuid.uuid4().hex
    except Exception:
        return None


def response_status(response):
    try:
        return response.status
    except Exception:
        return None


def _known(value, allowed):
    return value if type(value) is str and value in allowed else None


def _counter(value):
    return value if type(value) is int and 0 <= value <= _MAX_INTEGER else None


def _emit(call_id, component, model, attempt, started, state):
    # Keep every metadata lookup and sink operation inside this boundary.
    try:
        if call_id is None or started is None:
            return
        obj = state['response']
        obj = obj if type(obj) is dict else {}
        usage = obj.get('usage')
        usage = usage if type(usage) is dict else {}
        details = usage.get('completion_tokens_details')
        details = details if type(details) is dict else {}
        stop = obj.get('stop_reason')
        choices = obj.get('choices')
        if type(choices) is list and choices and type(choices[0]) is dict:
            stop = choices[0].get('finish_reason')
        status = state['http_status']
        row = {
            'schema': 1,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'call_id': call_id,
            'component': component,
            'operation': 'chat',
            'requested_model': _known(model, _MODELS),
            'served_model': _known(obj.get('model'), _MODELS),
            'status': state['status'],
            'attempt': attempt,
            'duration_ms': max(0, (time.monotonic_ns() - started) // 1_000_000),
            'http_status': status if type(status) is int and 100 <= status <= 599 else None,
            'stop_reason': _known(stop, _STOPS),
            'usage': {
                'input_tokens': _counter(usage.get('prompt_tokens', usage.get('input_tokens'))),
                'output_tokens': _counter(usage.get('completion_tokens', usage.get('output_tokens'))),
                'total_tokens': _counter(usage.get('total_tokens')),
                'cache_hit_tokens': _counter(usage.get('prompt_cache_hit_tokens', usage.get('cache_read_input_tokens'))),
                'cache_miss_tokens': _counter(usage.get('prompt_cache_miss_tokens')),
                'reasoning_tokens': _counter(details.get('reasoning_tokens')),
                'cache_creation_tokens': _counter(usage.get('cache_creation_input_tokens')),
            },
        }
        line = 'DEEPSEEK_CALL ' + json.dumps(row, separators=(',', ':'), ensure_ascii=True) + '\n'
        if len(line.encode('utf-8')) <= 2048:
            sys.stderr.write(line)
    except Exception:
        pass


@contextmanager
def call_attempt(call_id, component, model, attempt):
    state = {'status': 'transport_error', 'http_status': None, 'response': None}
    try:
        started = time.monotonic_ns()
    except Exception:
        started = None
    try:
        yield state
    except BaseException as error:
        if not isinstance(error, Exception):
            state['status'] = 'cancelled'
        # Logging must not replace an exception already leaving the attempt.
        try:
            _emit(call_id, component, model, attempt, started, state)
        except BaseException:
            pass
        raise
    else:
        _emit(call_id, component, model, attempt, started, state)
