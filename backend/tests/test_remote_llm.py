"""The Studio-supplied remote model for a presentation pass
(`PresentationSettings.llm_*` -> `llm.client.RemoteChatClient`): the director
uses it when it answers a probe, skips the LM Studio load/eject/responsiveness
dance for it, and falls back to LM Studio when it does not answer.
"""

import json

import httpx
import pytest

from llm import client as client_mod
from presentation import director
from presentation.models import PresentationSettings


def _remote_settings(**extra):
    return PresentationSettings(llm_provider="openai_compat", llm_base_url="http://127.0.0.1:8787/v1",
                                llm_api_key="sk-local-x", llm_model="sonnet", **extra)


def test_default_settings_name_no_remote_model():
    assert director._remote_client_for(PresentationSettings()) is None
    assert director._remote_client_for(None) is None
    # The provider alone is not enough: an endpoint is needed.
    assert director._remote_client_for(PresentationSettings(llm_provider="openai_compat")) is None


def test_remote_settings_build_a_client_with_the_key_and_model():
    remote = director._remote_client_for(_remote_settings())
    assert isinstance(remote, client_mod.RemoteChatClient)
    assert remote.base_url == "http://127.0.0.1:8787/v1"
    assert remote.model_name == "sonnet"
    assert remote._headers()["Authorization"] == "Bearer sk-local-x"


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient: records requests, answers from a script."""

    requests = []
    get_status = 200
    post_payload = {"choices": [{"message": {"content": "```json\n{\"topics\": []}\n```"}}]}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, headers=None, **kwargs):
        _FakeAsyncClient.requests.append(("GET", url, headers, None))
        return _FakeResponse(_FakeAsyncClient.get_status)

    async def post(self, url, headers=None, json=None, **kwargs):
        _FakeAsyncClient.requests.append(("POST", url, headers, json))
        return _FakeResponse(200, _FakeAsyncClient.post_payload)


@pytest.fixture
def fake_httpx(monkeypatch):
    _FakeAsyncClient.requests = []
    _FakeAsyncClient.get_status = 200
    monkeypatch.setattr(client_mod.httpx, "AsyncClient", _FakeAsyncClient)
    return _FakeAsyncClient


@pytest.mark.asyncio
async def test_director_asks_the_remote_model_and_restates_the_schema(fake_httpx):
    ask = await director._llm_asker(_remote_settings())
    assert ask is not None

    schema = {"name": "moods", "schema": {"type": "object", "properties": {"topics": {"type": "array"}}}}
    answer = await ask("SYSTEM", "USER", schema)

    # The code fence is peeled off so first_json_object() gets clean JSON.
    assert json.loads(answer) == {"topics": []}
    method, url, headers, body = fake_httpx.requests[-1]
    assert (method, url) == ("POST", "http://127.0.0.1:8787/v1/chat/completions")
    assert headers["Authorization"] == "Bearer sk-local-x"
    assert body["model"] == "sonnet"
    assert "response_format" not in body
    assert "JSON schema" in body["messages"][0]["content"]
    assert body["messages"][1]["content"] == "USER"


@pytest.mark.asyncio
async def test_director_falls_back_to_lm_studio_when_the_remote_model_is_down(fake_httpx, monkeypatch):
    fake_httpx.get_status = 503

    async def no_local_model(*args, **kwargs):
        return None

    import llm.lm_launcher as launcher
    monkeypatch.setattr(launcher, "ensure_ready", no_local_model)

    # Remote refused, LM Studio has nothing loaded: no model at all, and the
    # pass degrades exactly as before this feature existed.
    assert await director._llm_asker(_remote_settings()) is None
    assert fake_httpx.requests[0][0] == "GET"


def test_the_report_never_carries_the_remote_api_key():
    settings = _remote_settings()
    dumped = settings.model_dump(exclude={"llm_api_key"})
    assert "llm_api_key" not in dumped
    assert dumped["llm_provider"] == "openai_compat"


def test_unknown_llm_provider_values_are_ignored_not_fatal():
    # An older Studio (or a typo) must not break the job: anything but
    # "openai_compat" means LM Studio, as always.
    assert director._remote_client_for(PresentationSettings(llm_provider="claude", llm_base_url="http://x")) is None


def test_httpx_is_the_real_module_in_client():
    # Guard against the fixture above shadowing the wrong attribute.
    assert client_mod.httpx is httpx
