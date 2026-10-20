"""A small OpenAI-compatible chat-completions client (stdlib only) with tool calling.

The provider is config, not code. By default it points at the opencode Zen gateway; any endpoint that speaks
`POST /chat/completions` with `tools` works.

    TACIT_LLM_BASE_URL       default https://opencode.ai/zen/v1
    TACIT_LLM_MODEL          required, e.g. deepseek-v4-flash
    TACIT_LLM_API_KEY        or TACIT_LLM_API_KEY_FILE (a file holding the key)
"""
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://opencode.ai/zen/v1"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """The provider call failed (after retries). The observer treats this as a failed turn."""


class Client:
    def __init__(self, base_url, api_key, model, timeout=120, retries=3, backoff=2.0, sleep=time.sleep, opener=None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep
        self.opener = opener or urllib.request.urlopen

    def chat(self, messages, tools, session=None):
        """One completion. Returns the assistant message dict (`content`, optional `tool_calls`)."""
        body = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = [{"type": "function", "function": t} for t in tools]
            body["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        if session:
            # opencode groups a conversation's requests by this header (prompt caching, usage per session)
            headers["x-opencode-session"] = session
        request = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(),
                                         headers=headers, method="POST")
        delay = self.backoff
        for attempt in range(self.retries + 1):
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read())
                break
            except urllib.error.HTTPError as error:
                retryable = error.code in RETRY_STATUS
                detail = f"HTTP {error.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                retryable, detail = True, type(error).__name__
            except ValueError as error:  # unparseable body
                retryable, detail = False, f"bad response: {error}"
            if not retryable or attempt == self.retries:
                raise LLMError(detail)
            self.sleep(delay)
            delay *= 2
        try:
            return payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise LLMError(f"bad response shape: {error}") from None


def client_from_env(env=os.environ):
    model = env.get("TACIT_LLM_MODEL")
    if not model:
        raise LLMError("TACIT_LLM_MODEL is not set")
    key = env.get("TACIT_LLM_API_KEY")
    if not key and env.get("TACIT_LLM_API_KEY_FILE"):
        key = pathlib.Path(env["TACIT_LLM_API_KEY_FILE"]).expanduser().read_text().strip()
    if not key:
        raise LLMError("TACIT_LLM_API_KEY or TACIT_LLM_API_KEY_FILE is not set")
    return Client(env.get("TACIT_LLM_BASE_URL", DEFAULT_BASE_URL), key, model)
