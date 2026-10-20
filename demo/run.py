"""Live smoke test: run the real observer on each demo session in order, then print what landed (cards, outcomes,
tool errors, whisper). Usage: run.py <transcripts dir> <tacit home>"""
import json
import logging
import pathlib
import sys
import time

from tacit import observer, stats
from tacit.llm import client_from_env
from tacit.store import Store

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
src, home = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
store = Store(home / "tacit.db")
client = client_from_env()

for path in sorted(src.glob("*.jsonl")):
    session = path.stem
    store.enqueue(session, path)
    batch = store.next_batch(session)
    started = time.monotonic()
    result = observer.observe(store, batch, client)
    print(f"\n=== {session}: settled={result.settled} reasons={result.retry_reasons} rounds={result.rounds} "
          f"{time.monotonic() - started:.1f}s redactions={result.redactions}")
    for e in result.tool_errors:
        print(f"  tool error: {e.name} [{e.kind}] {e.message[:300]}")
    for r in store.db.execute("select outcome, code, card, reason from outcomes where batch = ? order by rowid", (batch.batch,)):
        print(f"  outcome: {r['outcome']:<10} {r['code'] or '':<26} {r['card'] or ''} {(r['reason'] or '')[:120]}")
    for r in store.db.execute("select i.text, o.outcome, o.reason from intents i left join intent_outcomes o "
                              "on o.intent = i.intent where i.batch = ?", (batch.batch,)):
        print(f"  intent: {r['text'][:60]!r} -> {r['outcome']} {r['reason'] or ''}")
    if result.whisper:
        print("  WHISPER:\n    " + result.whisper.replace("\n", "\n    "))

print("\n=== cards")
for c in store.current():
    print(f"  {c['card']} {c['bucket']:<14} v{c['version']} r{c['reinforcements']} {c['title']}  {json.dumps(c['fields'], ensure_ascii=False)}")
    for e in store.evidence(c["card"]):
        print(f"      [{e['role']}/{e['kind']}] {e['quote'][:110]!r}")
print("\n=== health:", stats.check(store)["overall"])
