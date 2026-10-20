# Live smoke test

Four short, made-up sessions (a decision, a gotcha with a fake secret, a "remember that" intent confirmed by a tool
result, and a later Japanese question that should bring back a card), run through the real observer.

```sh
uv run python demo/make_transcripts.py /tmp/tacit-demo/transcripts
TACIT_LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai \
TACIT_LLM_MODEL=gemini-3.1-flash-lite TACIT_LLM_API_KEY_FILE=~/.config/tacit/key \
  uv run python demo/run.py /tmp/tacit-demo/transcripts /tmp/tacit-demo/home
```

Each run prints every batch's outcomes, tool errors, intents and whisper, then the cards with their evidence.

The README GIF is drawn from that store: `uv run --with pillow python demo/gif.py /tmp/tacit-demo/home docs/demo.gif`.
Every tacit line in it (the whisper, `tacit show`, `tacit health`) is real CLI output, run on a copy of the store.
