"""A scripted stand-in for the OpenAI-compatible client: no network, no tokens.

Each script step is one model reply: a list of tool calls `(name, args)` (args may be a raw string, to send invalid
JSON), a plain string (a final text reply), or a callable `(messages) -> step` that can read earlier tool results.
When the script runs out, the model ends its turn.
"""
import copy
import json

from tacit.llm import LLMError


class FakeClient:
    def __init__(self, *script, fail=None):
        self.script = list(script)
        self.fail = fail  # raise LLMError on this 1-based call number
        self.requests = []

    def chat(self, messages, tools, session=None):
        self.requests.append({"messages": copy.deepcopy(messages), "tools": tools, "session": session})
        if self.fail == len(self.requests):
            raise LLMError("HTTP 503")
        step = self.script.pop(0) if self.script else "done"
        if callable(step):
            step = step(messages)
        if isinstance(step, str):
            return {"role": "assistant", "content": step}
        calls = [{"id": f"call_{len(self.requests)}_{i}", "type": "function",
                  "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}
                 for i, (name, args) in enumerate(step)]
        return {"role": "assistant", "content": None, "tool_calls": calls}

    def tool_results(self, name=None):
        """Tool results the model saw, parsed, in order (from the last request)."""
        messages = self.requests[-1]["messages"] if self.requests else []
        calls = {c["id"]: c["function"]["name"] for m in messages for c in m.get("tool_calls", [])}
        return [json.loads(m["content"]) for m in messages
                if m["role"] == "tool" and (name is None or calls.get(m["tool_call_id"]) == name)]

    def payload(self):
        return json.dumps([r["messages"] for r in self.requests], ensure_ascii=False)
