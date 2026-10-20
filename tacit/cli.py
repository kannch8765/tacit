"""tacit command line.

    tacit hook stop                    # Claude Code Stop hook: queue the session, start the background worker
    tacit hook prompt                  # UserPromptSubmit hook: print pending whispers + fast-lane matches as context
    tacit worker                       # drain queued batches (normally started by the Stop hook)
    tacit search <words>               # active cards, best match first
    tacit show <card>                  # one card with its evidence and links
    tacit history <card>               # every version on the card's lineage
    tacit review                       # pending link suggestions (the human inbox)
    tacit review accept|dismiss <id>   # decide one; accepting applies it
    tacit stats [--days N]             # counts only (JSON)
    tacit health                       # ok / warn / alert per check (JSON); exit code 0 / 1 / 2
    tacit report [--out DIR]           # report.html + metrics.json (aggregates only)

The store lives at $TACIT_HOME/tacit.db (default ~/.tacit). Set TACIT_NO_WORKER=1 to queue without observing.
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import pathlib
import sys

from tacit import report, stats, whisper, worker
from tacit.buckets import Rejected
from tacit.store import Store, stable_id

PREAMBLE = ("Notes from the team knowledge base (tacit). Quotes are exact transcript excerpts; dates say when each "
            "card became true or was recorded. Treat them as context, and say so if they look outdated.")


def home():
    return pathlib.Path(os.environ.get("TACIT_HOME", pathlib.Path.home() / ".tacit"))


def store_path():
    return home() / "tacit.db"


def hook_stop(store, stdin):
    """Queue the session and start the worker. A hook must never break Claude Code, so bad input is ignored (and
    reported on stderr)."""
    try:
        payload = json.loads(stdin.read() or "{}")
        session, transcript = payload.get("session_id"), payload.get("transcript_path")
        if session and transcript:
            store.enqueue(session, transcript)
            if os.environ.get("TACIT_NO_WORKER") != "1":
                worker.spawn(str(home()))
    except Exception as error:  # noqa: BLE001
        print(f"tacit: stop hook ignored: {type(error).__name__}", file=sys.stderr)


def hook_prompt(store, stdin, stdout):
    """Hand pending whispers (async lane) and clear full-text matches (sync lane, no model) to the next prompt."""
    try:
        payload = json.loads(stdin.read() or "{}")
        session, prompt = payload.get("session_id"), payload.get("prompt") or ""
        if not session:
            return
        pending = store.pending_whispers(session)
        texts = [w["text"] for w in pending]
        store.mark_delivered([w["whisper"] for w in pending])
        for card, text in whisper.prefetch(store, session, prompt):
            whisper_id = stable_id("whisper", {"session": session, "sync": hashlib.sha256(prompt.encode()).hexdigest(),
                                               "card": card["card"]})
            if store.queue_whisper(whisper_id, session, "sync", card["card"], text, delivered=True):
                texts.append(text)
        if texts:
            context = PREAMBLE + "\n\n" + "\n\n".join(texts)
            json.dump({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": context}}, stdout)
    except Exception as error:  # noqa: BLE001
        print(f"tacit: prompt hook ignored: {type(error).__name__}", file=sys.stderr)


def line(card):
    when = card["recorded_at"][:10]
    extra = f"  [superseded by {card['superseded_by']}]" if card["status"] == "superseded" else ""
    return f"{card['card']}  {card['bucket']:<14} v{card['version']}  {when}  {card['title']}{extra}"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tacit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    hook = sub.add_parser("hook")
    hook.add_argument("event", choices=["stop", "prompt"])
    sub.add_parser("worker")
    search = sub.add_parser("search")
    search.add_argument("words", nargs="+")
    search.add_argument("--limit", type=int, default=5)
    for name in ("show", "history"):
        sub.add_parser(name).add_argument("card")
    review = sub.add_parser("review")
    review.add_argument("action", nargs="?", choices=["accept", "dismiss"])
    review.add_argument("review", nargs="?")
    sub.add_parser("stats").add_argument("--days", type=int, default=7)
    sub.add_parser("health")
    sub.add_parser("report").add_argument("--out", type=pathlib.Path)
    args = parser.parse_args(argv)
    if args.cmd == "review" and args.action and not args.review:
        parser.error("review accept|dismiss needs a review id")
    tz = datetime.datetime.now().astimezone().tzinfo

    store = Store(store_path())
    try:
        if args.cmd == "hook" and args.event == "stop":
            hook_stop(store, sys.stdin)
        elif args.cmd == "hook":
            hook_prompt(store, sys.stdin, sys.stdout)
        elif args.cmd == "worker":
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
            worker.run(store, str(home()))
        elif args.cmd == "search":
            for card in store.search(" ".join(args.words), args.limit):
                print(line(card))
        elif args.cmd == "show":
            card = store.get(args.card)
            if card is None:
                parser.exit(1, f"no card {args.card}\n")
            print(line(card))
            for k, v in card["fields"].items():
                print(f"  {k}: {v}")
            for e in store.evidence(args.card):
                print(f"  evidence [{e['role']} line {e['line']}]: {e['quote']}")
            for link in store.links_of(args.card):
                print(f"  link: {link['src']} {link['kind']} {link['dst']}")
        elif args.cmd == "history":
            for card in store.history(args.card):
                print(line(card))
        elif args.cmd == "review" and args.action:
            try:
                store.decide_review(args.review, args.action == "accept")
            except Rejected as error:
                parser.exit(1, f"tacit: {error}\n")
            print(f"{args.review} {'accepted' if args.action == 'accept' else 'dismissed'}")
        elif args.cmd == "review":
            for r in store.reviews():
                print(f"{r['review']}  {r['relation']:<11} {r['created_at'][:10]}  {r['reason']}")
                for card in r["cards"]:
                    found = store.get(card)
                    print(f"    {line(found) if found else card}")
        elif args.cmd == "stats":
            print(stats.summary_json(store, tz, stats.recent_days(datetime.datetime.now(tz), tz, args.days)))
        elif args.cmd == "health":
            result = stats.check(store)
            print(json.dumps(result, indent=2))
            return stats.LEVELS.index(result["overall"])
        elif args.cmd == "report":
            day = datetime.datetime.now(tz).date().isoformat()
            print(report.write(store, tz, day, args.out or home() / "report"))
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
