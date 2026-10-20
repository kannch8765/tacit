"""Assemble docs/index.html, the tacit tech guide (GitHub Pages). Usage: python site/build.py [out_dir]"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TITLE = "tacit — Tech guide"
DESCRIPTION = ("tacit: a team knowledge base that writes itself from your Claude Code sessions, and only says what it "
               "can prove. How it observes, grounds, versions and whispers knowledge back.")

EXTRA_CSS = """
.emoji-icon{font-size:78px;line-height:1;align-items:flex-end}
figure{margin:0}
.tables{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:16px 0}
.tcard{border:1px solid var(--border);border-radius:6px;padding:10px 14px;font-size:.86rem}
.tcard .tname{font-family:var(--mono);font-weight:650;margin-bottom:4px}
.tcard .tnote{color:var(--muted);font-size:.78rem;margin:0 0 6px}
.tcard ul{list-style:none;padding:0;margin:0}.tcard li{margin:2px 0;font-family:var(--mono);font-size:.8rem;color:var(--muted)}
.tcard b{color:var(--text);font-weight:600}.tcard i{font-style:normal;font-size:.68rem;color:#eb5757;border:1px solid rgba(235,87,87,.35);border-radius:3px;padding:0 3px}
.table-wrap{overflow-x:auto;margin:14px 0}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{border:1px solid var(--border);padding:7px 10px;vertical-align:top;text-align:left}
th{background:var(--code-bg);font-weight:600}
td:first-child{white-space:nowrap}
td code{font-size:.78rem;color:var(--text);background:none;padding:0;overflow-wrap:anywhere}
.code-wrap.out pre{white-space:pre-wrap;overflow-wrap:anywhere}
.demo-gif{display:block;width:100%;height:auto;margin:18px 0 6px;border:1px solid var(--border);border-radius:8px}
.cap{color:var(--muted);font-size:.82rem;margin:4px 0 0}
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:16px 0}
.stat{border:1px solid var(--border);border-radius:6px;padding:10px 12px}
.stat .v{font-size:1.5rem;font-weight:650;line-height:1.2}.stat .l{color:var(--muted);font-size:.78rem}
@media(max-width:600px){.tables{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
"""

SCRIPT = """
document.querySelectorAll('.copy').forEach(btn=>{
  btn.addEventListener('click',async()=>{
    const text=btn.parentElement.querySelector('pre').innerText;
    try{await navigator.clipboard.writeText(text);btn.textContent='Copied';setTimeout(()=>btn.textContent='Copy',1100)}
    catch{btn.textContent='Select'}
  });
});
document.querySelectorAll('.toc-rail a').forEach(a=>{a.setAttribute('aria-label',a.dataset.label);a.setAttribute('title',a.dataset.label)});
const allLinks=[...document.querySelectorAll('.toc-rail a, .toc-panel a')];
const ids=[...new Set(allLinks.map(a=>a.getAttribute('href')).filter(Boolean))];
const sections=ids.map(id=>document.querySelector(id)).filter(Boolean);
const setActive=id=>allLinks.forEach(a=>a.classList.toggle('active',a.getAttribute('href')===id));
const io=new IntersectionObserver(entries=>{
  const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);
  if(visible[0]) setActive('#'+visible[0].target.id);
},{rootMargin:'-15% 0px -72% 0px',threshold:[0,.1,.5]});
sections.forEach(s=>io.observe(s));
setActive(ids[0]);
"""


def build(out: Path) -> None:
    css = (HERE / "base.css").read_text().replace("</style>", EXTRA_CSS + "</style>")
    body = (HERE / "body.html").read_text()
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light" />
<meta name="description" content="{DESCRIPTION}" />
<title>{TITLE}</title>
{css}
</head>
<body>
{body}
<script>{SCRIPT}</script>
</body>
</html>
"""
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html)
    (out / ".nojekyll").write_text("")


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "docs")
