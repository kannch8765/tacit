"""The OpenAI-compatible client: request shape, session header, bounded retries, and clear failures."""
import io
import json
import urllib.error

import pytest

from tacit.llm import Client, LLMError, client_from_env


class Opener:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return io.BytesIO(json.dumps(response).encode())


def http(code):
    return urllib.error.HTTPError("u", code, "x", {}, None)


OK = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}


def client(opener, **kw):
    return Client("https://gw.example/v1/", "k", "m", sleep=lambda s: None, opener=opener, **kw)


def test_request_shape_and_session_header():
    opener = Opener(OK)
    reply = client(opener).chat([{"role": "user", "content": "x"}], [{"name": "search", "parameters": {}}], session="s1")
    (request,) = opener.requests
    body = json.loads(request.data)
    assert reply["content"] == "hi"
    assert request.full_url == "https://gw.example/v1/chat/completions"
    assert request.get_header("X-opencode-session") == "s1" and request.get_header("Authorization") == "Bearer k"
    assert body["model"] == "m" and body["tools"] == [{"type": "function", "function": {"name": "search", "parameters": {}}}]


def test_retries_transient_errors_then_succeeds():
    opener = Opener(http(429), urllib.error.URLError("reset"), OK)
    assert client(opener).chat([], [])["content"] == "hi" and len(opener.requests) == 3


@pytest.mark.parametrize("responses, detail", [
    ([http(401)], "HTTP 401"),  # not retried
    ([http(503)] * 4, "HTTP 503"),  # retries are bounded
    ([{"choices": []}], "bad response shape"),
])
def test_failures_raise_llm_error(responses, detail):
    opener = Opener(*responses)
    with pytest.raises(LLMError, match=detail):
        client(opener, retries=3).chat([], [])
    assert len(opener.requests) == len(responses)


def test_client_from_env(tmp_path):
    with pytest.raises(LLMError, match="MODEL"):
        client_from_env({})
    (tmp_path / "key").write_text("secret\n")
    c = client_from_env({"TACIT_LLM_MODEL": "deepseek-v4-flash", "TACIT_LLM_API_KEY_FILE": str(tmp_path / "key")})
    assert (c.api_key, c.base_url) == ("secret", "https://opencode.ai/zen/v1")
