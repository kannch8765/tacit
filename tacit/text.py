"""Lexical tokens shared by search and whisper ranking: Latin/number words (2+ chars) and CJK character bigrams.

SQLite's default FTS5 tokenizer treats a whole CJK run (even with Latin glued on, like "デプロイはVPN必須") as one token,
so Japanese and Chinese text would only match itself exactly. tacit indexes and queries these tokens instead.
"""
import re
import unicodedata

LATIN = re.compile(r"[^\W_]+")
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]+")  # kana (incl. ー), CJK ideographs, half-width kana


def tokens(text):
    text = unicodedata.normalize("NFKC", text).lower()
    result = set()
    for run in CJK.findall(text):
        result.update(run[i:i + 2] for i in range(len(run) - 1))
    for word in LATIN.findall(CJK.sub(" ", text)):
        if len(word) > 1:
            result.add(word)
    return result
