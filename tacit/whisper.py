"""Whispers: grounded, source-faithful context for the next prompt.

Selection happens in the observer turn: the agent may pick only a card (and 1-3 of its quote snippets) that its own
search returned in this turn. This module owns the parts that don't need a model: cutting a card's evidence into
quotable snippets, describing what was surfaced (so a wrong pick can self-correct), and rendering the final text.
"""
import re

from tacit.store import stable_id
from tacit.text import CJK, tokens

SNIPPET_CHARS = 600
QUOTABLE_KINDS = (None, "user_text", "assistant_text", "tool_result")  # raw tool-call JSON is never quoted
# CJK sentence marks end a sentence anywhere; Latin . ! ? only before whitespace or the end, so "v3.2" or "e.g."
# inside a word never splits a quote
SENTENCE_END = re.compile(r"[。！？]+[」』”’]?|[.!?]+[\"')\]」』”’]?(?=\s|$)")


def sentences(line):
    out, start = [], 0
    for match in SENTENCE_END.finditer(line):
        out.append(line[start:match.end()])
        start = match.end()
    out.append(line[start:])
    return [s.strip() for s in out if s.strip()]


def split_source_text(text):
    """Exact excerpts of a quote: one per sentence, each at most SNIPPET_CHARS. Never rewords anything."""
    pieces = []
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        for sentence in sentences(line):
            for offset in range(0, len(sentence), SNIPPET_CHARS):
                if sentence[offset:offset + SNIPPET_CHARS]:
                    pieces.append(sentence[offset:offset + SNIPPET_CHARS])
    return pieces


def bound(items, limit=8):
    """At most `limit` snippets, keeping the oldest and the newest halves."""
    unique = list({s["snippet_id"]: s for s in items}.values())
    if len(unique) <= limit:
        return unique
    return unique[:(limit + 1) // 2] + unique[len(unique) - limit // 2:]


def snippets(store, card, limit=8):
    result = []
    for e in store.evidence(card):
        if e["kind"] not in QUOTABLE_KINDS:
            continue
        for index, quote in enumerate(split_source_text(e["quote"])):
            snippet = stable_id("snip", {"card": card, "evidence": e["evidence_id"] or e["line"], "index": index, "quote": quote})
            result.append({"snippet_id": snippet, "role": e["role"], "quote": quote, "batch": e["batch"]})
    return bound(result, limit)


def search_result(store, card):
    """What a search returns to the observer: the card plus quotable snippets."""
    return {"card": card["card"], "bucket": card["bucket"], "title": card["title"], "body": card["body"],
            "fields": card["fields"], "version": card["version"], "recorded_at": card["recorded_at"],
            "valid_from": card["valid_from"], "reinforcements": card["reinforcements"],
            "snippets": [{k: s[k] for k in ("snippet_id", "role", "quote")} for s in snippets(store, card["card"])]}


def describe_surfaced(surfaced):
    """Lists what deliver_whisper may select this turn, so the model can retry correctly in the same turn."""
    if not surfaced:
        return "Nothing was surfaced yet; call search with purpose=whisper first."
    entries = [f"{card} (snippet_ids: {', '.join(s) or 'none'})" for card, s in surfaced.items()]
    return "Surfaced this turn: " + "; ".join(entries) + "."


# ---------------------------------------------------------------------------------------------------- weaving

HISTORY_BUDGET = 420  # chars for earlier versions
HISTORY_MAX = 3
RELATED_TOP_K = 2
RELATED_BUDGET = 420  # delivery budget only: the link graph itself has no cap
MMR_RELEVANCE_WEIGHT = 0.75
def jaccard(a, b):
    common = len(a & b)
    return common / (len(a) + len(b) - common) if common else 0.0


def coverage(query, candidate):
    return len(query & candidate) / len(query) if query else 0.0


def first_sentence(quote):
    found = sentences(quote.strip().split("\n")[0])
    return found[0] if found else quote.strip()


def history_lines(store, card):
    """Earlier versions of the card, newest first, each with at most one short exact quote."""
    lines, remaining = [], HISTORY_BUDGET
    for prior in [store.get(c) for c in store.predecessors(card["card"])[:HISTORY_MAX]]:
        line = f"  was (until {(prior['superseded_at'] or '')[:10]}): {prior['title']}"
        if len(line) > remaining:
            continue
        lines.append(line)
        remaining -= len(line)
        quote = next((e for e in reversed(store.evidence(prior["card"])) if e["kind"] in QUOTABLE_KINDS and e["quote"].strip()), None)
        if quote:
            quoted = f"    [{quote['role']}] “{first_sentence(quote['quote'])}”"
            if len(quoted) <= min(remaining, 220):  # never truncate a quote: include it whole or not at all
                lines.append(quoted)
                remaining -= len(quoted)
    return lines


def related_lines(store, card, topic, known=frozenset()):
    """Linked current cards. Contradictions always come first (they are warnings), then at most two related cards
    chosen by lexical MMR: relevant to the topic, not redundant with each other. Returns (lines, included_card_ids)."""
    neighbours = {}
    for link in store.links_of(card["card"]):
        if link["kind"] == "supersedes":
            continue  # the lineage is already shown as history
        other = link["dst"] if link["src"] == card["card"] else link["src"]
        neighbours.setdefault(other, link["kind"])
    main_tokens, topic_tokens = tokens(card["title"]), tokens(topic or "")
    seen, candidates = {card["title"].strip().lower()}, []
    heading = "  related:"
    remaining = RELATED_BUDGET - len(heading)
    for other, kind in sorted(neighbours.items()):
        found = store.get(other)
        if found is None or found["status"] != "active" or other in known or found["title"].strip().lower() in seen:
            continue
        seen.add(found["title"].strip().lower())
        line = f"  - {'⚠ contradicts: ' if kind == 'contradicts' else ''}{found['title']} ({found['bucket']}, {other})"
        if len(line) > remaining:
            continue
        t = tokens(found["title"])
        closeness = jaccard(t, main_tokens)
        relevance = 0.7 * coverage(topic_tokens, t) + 0.3 * closeness if topic_tokens else closeness
        candidates.append({"card": found, "line": line, "tokens": t, "relevance": relevance, "closeness": closeness})
    # If some neighbours match the topic, don't pad with unrelated ones; if none do (e.g. paraphrases), keep the
    # human-confirmed links as a fallback rather than hiding them all.
    selected = []
    for c in [c for c in candidates if c["line"].startswith("  - ⚠")]:
        if len(c["line"]) <= remaining:
            selected.append(c)
            remaining -= len(c["line"])
    candidates = [c for c in candidates if c not in selected]
    pool = [c for c in candidates if c["relevance"] > 0] or candidates
    warnings = len(selected)
    while len(selected) - warnings < RELATED_TOP_K:
        best, best_score = None, float("-inf")
        for c in pool:
            if c in selected or len(c["line"]) > remaining:
                continue
            redundancy = max([c["closeness"]] + [jaccard(c["tokens"], s["tokens"]) for s in selected])
            score = MMR_RELEVANCE_WEIGHT * c["relevance"] - (1 - MMR_RELEVANCE_WEIGHT) * redundancy
            if score > best_score or (score == best_score and best and c["card"]["recorded_at"] > best["card"]["recorded_at"]):
                best, best_score = c, score
        if best is None:
            break
        selected.append(best)
        remaining -= len(best["line"])
    if not selected:
        return [], []
    return [heading] + [c["line"] for c in selected], [c["card"]["card"] for c in selected]


def render(store, card, picked, topic="", known=frozenset()):
    """The woven whisper: the card with its date and version, its fields, the picked exact quotes, what it replaced,
    and up to two related cards. Returns (text, related_card_ids)."""
    when = (card["valid_from"] or card["recorded_at"])[:10]
    lines = [f"tacit · {card['bucket']} (as of {when}, v{card['version']}, {card['card']}): {card['title']}"]
    lines += [f"  {name}: {value}" for name, value in card["fields"].items()]
    lines += [f"  [{s['role']}] “{s['quote']}”" for s in picked]
    lines += history_lines(store, card)
    related, included = related_lines(store, card, topic, known)
    return "\n".join(lines + related), included


# ---------------------------------------------------------------------------------------------------- sync lane

STOPWORDS = frozenset("""a an and are as at be but by can could did do does for from had has have how i if in into is it its
just let me my no not now of on or our please should so that the their them then there these they this to too up us
was we were what when where which who why will with would you your""".split())
SYNC_LIMIT = 2


def prefetch(store, session, prompt, limit=SYNC_LIMIT):
    """The fast lane: cards that clearly match the prompt, by full-text search only (no model call). A card needs at
    least two meaningful terms in common with the prompt (or one, for a one-word prompt). Returns [(card, text)]."""
    terms = {t for t in tokens(prompt) if t not in STOPWORDS and (len(t) > 2 or CJK.fullmatch(t))}
    if not terms:
        return []
    known = set(store.session_known(session))
    need = 1 if len(terms) == 1 else 2
    result = []
    for card in store.search(" ".join(sorted(terms)), limit=10, exclude=sorted(known)):
        text = " ".join([card["title"], card["body"], *card["fields"].values()])
        if len(terms & tokens(text)) < need:
            continue
        picked = snippets(store, card["card"], limit=1)
        rendered, _ = render(store, card, picked, topic=prompt, known=known)
        result.append((card, rendered))
        if len(result) == limit:
            break
    return result
