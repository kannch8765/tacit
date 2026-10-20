"""Render the README GIF from a store that demo/run.py filled. Every tacit line in it is real CLI output from that
store (run on a copy, so the store is untouched). Usage: gif.py <tacit home> <out.gif>"""
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

from tacit.store import Store

W, H, PAD, SIZE = 960, 560, 28, 17
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # index 5 = Noto Sans Mono CJK JP
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
BG, FG, DIM, ORANGE, GREEN, EYE = "#1f1e1d", "#e8e6e3", "#8c8a86", "#d97757", "#8fbf7f", "#1f1e1d"
font = ImageFont.truetype(FONT, SIZE, index=5)
bold = ImageFont.truetype(BOLD, SIZE, index=5)
big = ImageFont.truetype(BOLD, 44, index=5)
LINE = SIZE + 9

# Clawd, decoded from the Claude Code banner glyphs (one pixel per quarter block, so pixels are twice as tall as
# wide, like a terminal cell's quarters): o = body, e = eye
CLAWD = ["   oooooooooooo  ",
         "   ooeooooooeoo  ",
         " oooooooooooooooo",
         "   oooooooooooo  ",
         "    o o    o o   "]


def tacit(home, *args, stdin=None):
    env = dict(os.environ, TACIT_HOME=str(home), TACIT_NO_WORKER="1")
    return subprocess.run(["tacit", *args], input=stdin, capture_output=True, text=True, env=env, check=True).stdout


def wrap(text, width, f=font):
    out = []
    for raw in text.split("\n"):
        line = ""
        for ch in raw:
            if f.getlength(line + ch) > width:
                cut = line.rfind(" ")
                out.append(line[:cut] if cut > 0 else line)
                line = line[cut + 1:] if cut > 0 else ""
            line += ch
        out.append(line)
    return out


def clawd(draw, x, y, px, blink=False):
    for r, row in enumerate(CLAWD):
        for c, cell in enumerate(row):
            if cell in "oe":
                fill = ORANGE if cell == "o" or blink else EYE
                draw.rectangle([x + c * px, y + 2 * r * px, x + (c + 1) * px - 1, y + 2 * (r + 1) * px - 1], fill=fill)


class Screen:
    """A terminal page: a header (Claude Code's, with Clawd, or a plain shell's), a caption, blocks of colored lines.
    tacit itself only appears as what it is inside Claude Code: the hook lines and the context it adds."""

    def __init__(self, title, app="Claude Code"):
        self.title, self.app, self.blocks, self.frames = title, app, [], []

    def render(self, caption="", blink=False):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        x = PAD
        if self.app == "Claude Code":
            clawd(d, PAD, 14, 2, blink)
            x += 46
        d.text((x, 16), self.app, font=bold, fill=FG)
        d.text((x + bold.getlength(self.app + "  "), 16), self.title, font=font, fill=DIM)
        d.line([PAD, 50, W - PAD, 50], fill="#3a3836")
        y = 64
        for kind, lines, color in self.blocks:
            if kind == "box":
                y += 10
                top = y - 8
                for text in lines:
                    d.text((PAD + 16, y), text, font=font, fill=color)
                    y += LINE
                d.rounded_rectangle([PAD, top, W - PAD, y + 2], radius=8, outline=ORANGE, width=2)
                y += 14
            else:
                for text, c in lines:
                    d.text((PAD, y), text, font=font, fill=c)
                    y += LINE
        if caption:
            d.rectangle([0, H - 44, W, H], fill="#2a2826")
            d.text((PAD, H - 34), caption, font=font, fill=FG)
        return im

    def hold(self, ms, caption="", blink_at=None):
        self.frames.append((self.render(caption), ms))
        if blink_at:
            self.frames.append((self.render(caption, blink=True), 140))
            self.frames.append((self.render(caption), blink_at))

    def say(self, lines, color=FG, ms=500, caption=""):
        """Lines wrap at the screen edge; a redacted secret is highlighted."""
        self.blocks.append(("text", [(w, ORANGE if "[redacted:" in t else color)
                                     for t in lines for w in wrap(t, W - 2 * PAD)], None))
        self.hold(ms, caption)

    def type(self, prefix, text, color=FG, caption="", step=3):
        """Type text after a prompt prefix, a few characters per frame."""
        width = W - 2 * PAD
        self.blocks.append(("text", [], None))
        for n in range(0, len(text) + step, step):
            lines = wrap(prefix + text[:n], width)
            self.blocks[-1] = ("text", [(t, color) for t in lines], None)
            self.frames.append((self.render(caption), 45))
        self.hold(400, caption)

    def box(self, text, color=FG, ms=500, caption=""):
        self.blocks.append(("box", wrap(text, W - 2 * PAD - 32), color))
        self.hold(ms, caption)


def title_card(frames, line1, line2, ms):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text(((W - big.getlength("tacit")) // 2, 190), "tacit", font=big, fill=ORANGE)
    for i, t in enumerate((line1, line2)):
        d.text(((W - font.getlength(t)) // 2, 270 + i * 32), t, font=font, fill=FG if i == 0 else DIM)
    frames.append((im, ms))


def main():
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    home = pathlib.Path(tempfile.mkdtemp(prefix="tacit-gif-"))
    shutil.copy(src / "tacit.db", home / "tacit.db")
    store = Store(home / "tacit.db")
    pending = store.pending_whispers("s4-whisper")
    decision = pending[0]["card"]
    user_quote = next(e["quote"] for e in store.evidence(decision) if e["role"] == "user")
    outcome = store.db.execute("select outcome from outcomes where card = ? order by rowid limit 1", (decision,)).fetchone()[0]
    quotes = len(store.evidence(decision))
    gotcha = store.current("gotcha")[0]["card"]
    hook = json.loads(tacit(home, "hook", "prompt", stdin=json.dumps(
        {"session_id": "s4-whisper", "prompt": "pydantic 3.2 出たらしいけど、そろそろ上げていい？"})))
    context = hook["hookSpecificOutput"]["additionalContext"].split("\n\n", 1)[1]
    shown = tacit(home, "show", gotcha).rstrip("\n").split("\n")
    health = json.loads(tacit(home, "health"))

    frames = []
    title_card(frames, "a team knowledge base that writes itself from your Claude Code sessions",
               "and only says what it can prove", 1800)

    s = Screen("Wed 10-15 · session A")
    s.type("> ", user_quote, caption="A teammate makes a decision in chat with Claude Code.")
    s.say(["● Pinning it now.", "  ⎿ uv add 'pydantic<3'   - pydantic==3.0.1  + pydantic==2.11.4"], DIM, 900,
          "A teammate makes a decision in chat with Claude Code.")
    s.say(["", "Stop hook · tacit queued the session"], ORANGE, 700, "When the session stops, tacit reads it in the background.")
    s.say([f"tacit observer · {outcome} decision {decision}, grounded by {quotes} exact quotes"], GREEN, 2600,
          "When the session stops, tacit reads it in the background.")
    frames += s.frames

    s = Screen("Sun 10-19 · session B, another day")
    s.type("> ", "pydantic 3.2 出たらしいけど、そろそろ上げていい？前に何か問題あったっけ",
           caption="Days later, someone asks. (In Japanese: works across languages.)")
    s.say(["", "UserPromptSubmit hook · tacit adds to Claude's context:"], ORANGE, 500,
          "Days later, someone asks. (In Japanese: works across languages.)")
    s.box(context, FG, 5200, "Claude sees the team's decision, dated, with the exact words it came from.")
    frames += s.frames

    s = Screen("tacit CLI · the ledger", app="$ shell")
    s.say([f"$ tacit show {gotcha}"], FG, 500, "Every card keeps its evidence. Secrets are redacted before storage.")
    s.say(shown, DIM, 4200, "Every card keeps its evidence. Secrets are redacted before storage.")
    s.say(["", "$ tacit health"], FG, 400, "Health and stats read numbers only, never card text.")
    s.say([f"overall: {health['overall']}  ·  " + "  ".join(
        f"{c['name'].split('_')[0]} {c['level']}" for c in health["checks"][:4])], GREEN, 2600,
        "Health and stats read numbers only, never card text.")
    frames += s.frames

    title_card(frames, "local SQLite · any OpenAI-compatible model · MIT", "github.com/kannch8765/tacit", 2600)

    images = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f, _ in frames]
    images[0].save(out, save_all=True, append_images=images[1:], duration=[ms for _, ms in frames], loop=0,
                   optimize=True, disposal=1)
    shutil.rmtree(home)
    print(f"{out}: {len(frames)} frames, {out.stat().st_size // 1024} KiB")


if __name__ == "__main__":
    main()
