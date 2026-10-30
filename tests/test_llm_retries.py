from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from makemarket.llm import NIM_BASE_URL, ChatClient, LLMConfig, Throttle


class RequestError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


def client_for(tmp_path, monkeypatch, outcomes, base_url=NIM_BASE_URL):
    # Exercise transport/cache behavior without requiring the optional SDK.
    client = ChatClient.__new__(ChatClient)
    client.cfg = LLMConfig(model="test-model", base_url=base_url)
    client.cache_dir = tmp_path
    client.throttle = Throttle(1)
    client.max_retries = 3
    create = Mock(side_effect=outcomes)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr("makemarket.llm.time.sleep", lambda _: None)
    return client, create


def response():
    message = SimpleNamespace(content='{"bid": 1}', model_extra={})
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                           usage=SimpleNamespace(completion_tokens=4, prompt_tokens=8),
                           model="test-model", system_fingerprint="deployment-a")


@pytest.mark.parametrize("status", [404, 408, 409, 429, 500, 504, None])
def test_transient_failure_preserves_request_and_caches_success(tmp_path, monkeypatch, status):
    client, create = client_for(tmp_path, monkeypatch, [RequestError(status), response()])
    messages = [{"role": "user", "content": "quote"}]
    first = client.chat(messages)
    assert create.call_count == 2
    assert create.call_args_list[0] == create.call_args_list[1]
    assert first["cached"] is False
    replay = client.chat(messages)
    assert replay == {**first, "cached": True}
    assert create.call_count == 2


def test_persistent_nim_404_stops_at_retry_limit(tmp_path, monkeypatch):
    client, create = client_for(tmp_path, monkeypatch, [RequestError(404)] * 3)
    with pytest.raises(RuntimeError, match="giving up after 3 attempts"):
        client.chat([])
    assert create.call_count == 3
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status,base_url", [
    (400, NIM_BASE_URL), (401, NIM_BASE_URL), (403, NIM_BASE_URL), (410, NIM_BASE_URL),
    (404, "https://example.com/v1"),
])
def test_permanent_client_error_is_not_retried(tmp_path, monkeypatch, status, base_url):
    client, create = client_for(tmp_path, monkeypatch, [RequestError(status)], base_url)
    with pytest.raises(RuntimeError, match="RequestError"):
        client.chat([])
    assert create.call_count == 1
    assert not list(tmp_path.iterdir())
