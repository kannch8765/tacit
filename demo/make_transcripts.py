"""Four short, made-up Claude Code sessions for the live smoke test (no real session content)."""
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
n = 0


def msg(role, content, ts):
    global n
    n += 1
    return {"type": role, "uuid": f"demo-{n}", "timestamp": f"2026-10-{ts}",
            "message": {"role": role, "content": content}}


def tool_use(tid, name, inp):
    return [{"type": "tool_use", "id": tid, "name": name, "input": inp}]


def tool_result(tid, text):
    return [{"type": "tool_result", "tool_use_id": tid, "content": text}]


SESSIONS = {
    # 1. a decision with its rationale, plus routine noise to ignore
    "s1-decision": [
        msg("user", "pydantic 3 just broke our request validators again. Let's pin pydantic < 3 for now, "
                    "we can revisit when 3.2 ships the fix for custom validators.", "15T01:00:00Z"),
        msg("assistant", [{"type": "text", "text": "Pinning it now."}] + tool_use("t1", "Bash", {"command": "uv add 'pydantic<3'"}), "15T01:00:05Z"),
        msg("user", tool_result("t1", "Resolved 41 packages in 312ms\n - pydantic==3.0.1\n + pydantic==2.11.4"), "15T01:00:08Z"),
        msg("assistant", [{"type": "text", "text": "Done: pydantic is pinned to 2.11.4 and the validator tests pass again."}], "15T01:00:30Z"),
    ],
    # 2. a gotcha (symptom + fix), with a secret that must be redacted
    "s2-gotcha": [
        msg("user", "the staging deploy hangs at 'waiting for health check' again", "16T02:00:00Z"),
        msg("assistant", [{"type": "text", "text": "Let me look at the deploy log."}] + tool_use("t2", "Bash", {"command": "tail -5 deploy.log"}), "16T02:00:04Z"),
        msg("user", tool_result("t2", "connect ETIMEDOUT 10.8.0.12:443\nexport DEPLOY_TOKEN=dpl_live_9f8e7d6c5b4a3f2e1d0c"), "16T02:00:06Z"),
        msg("assistant", [{"type": "text", "text": "The health check can't reach 10.8.0.12, which is inside the staging VPN."}], "16T02:00:20Z"),
        msg("user", "ah right, I forgot the VPN. turned it on and the deploy went through. this bites everyone on day one.", "16T02:03:00Z"),
    ],
    # 3. an explicit "remember that" intent + a world update grounded by a tool result
    "s3-intent": [
        msg("user", "remember that Mika is the PM for the payments project now, Ken moved to the data team", "17T03:00:00Z"),
        msg("assistant", [{"type": "text", "text": "Noted. Let me check the team page."}] + tool_use("t3", "WebFetch", {"url": "https://wiki.example.com/payments/team"}), "17T03:00:04Z"),
        msg("user", tool_result("t3", "Payments team (updated 2026-10-14): PM: Mika Sato. Previous PM Ken Ito moved to Data Platform."), "17T03:00:07Z"),
        msg("assistant", [{"type": "text", "text": "Confirmed on the wiki: Mika Sato has been PM since the 14th."}], "17T03:00:15Z"),
    ],
    # 4. a later session where a whisper should help (asks about the pin), in Japanese
    "s4-whisper": [
        msg("user", "pydantic 3.2 出たらしいけど、そろそろ上げていい？前に何か問題あったっけ", "19T04:00:00Z"),
    ],
}

for name, lines in SESSIONS.items():
    (out / f"{name}.jsonl").write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines))
print(" ".join(sorted(SESSIONS)))
