"""Gate 2: build a self-contained HTML review sheet for checking every reference answer.

Usage:
    uv run python scripts/eval_set/build_review_sheet.py
Open data/review/review_sheet.html in a browser. Decisions save in the browser as you go;
"Export decisions" downloads review_decisions.json, which `assemble_eval_set.py apply-review` reads.
"""

from __future__ import annotations

import json
from pathlib import Path

from pitwall.eval_items import read_jsonl

SOURCE = Path("data/drafts/eval_candidates.jsonl")
OUT = Path("data/review/review_sheet.html")

ORDER = {"test": 0, "dev": 1, "safety": 2, "probe": 3, "spare": 4}

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>PITWALL Reference Review</title>
<style>
:root{--bg:#f5f6f8;--panel:#fff;--ink:#151a23;--muted:#5b6472;--line:#dde1e8;--acc:#6a34e0;--accs:#ece5fd;--ok:#11795a;--oks:#e0f2ea;--fix:#a86400;--fixs:#fbefd9;--bad:#b8322a;--bads:#fbe5e3}
@media (prefers-color-scheme:dark){:root{--bg:#0e1117;--panel:#161a22;--ink:#e8ebf1;--muted:#9aa3b2;--line:#2a303b;--acc:#a88bff;--accs:#2a2145;--ok:#4cc38a;--oks:#12291f;--fix:#e3a446;--fixs:#2f2410;--bad:#f07167;--bads:#351a1b;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--panel);border-bottom:1px solid var(--line);padding:10px max(16px,env(safe-area-inset-left));display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header b{font-size:16px}.bar{flex:1 1 160px;height:8px;background:var(--line);border-radius:9px;overflow:hidden}.bar span{display:block;height:100%;background:var(--acc);width:0}
select,button,textarea{font:inherit;color:inherit}select,button{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
main{max-width:900px;margin:0 auto;padding:16px;display:grid;gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px 16px;display:grid;gap:8px;scroll-margin-top:70px}
.card.focus{outline:2px solid var(--acc)}.card.ok{border-left:6px solid var(--ok)}.card.fix{border-left:6px solid var(--fix)}.card.drop{border-left:6px solid var(--bad)}
.meta{font:12px ui-monospace,Menlo,monospace;color:var(--muted);display:flex;gap:10px;flex-wrap:wrap}.q{font-size:17px;font-weight:600}
.ref{background:var(--bg);border-radius:6px;padding:8px 12px}.ref li{margin:2px 0}.crit{color:var(--acc);font-weight:700}
.flag{font-size:13px;color:var(--fix)}.auto{font-size:13px;color:var(--ok)}.btns{display:flex;gap:8px;flex-wrap:wrap}
.btns button.on{color:#fff}.b-ok.on{background:var(--ok)}.b-fix.on{background:var(--fix)}.b-drop.on{background:var(--bad)}
textarea{width:100%;min-height:44px;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px}
.help{font-size:13px;color:var(--muted)}
</style></head><body>
<header><b>Reference review</b><span id="count"></span><div class="bar"><span id="bar"></span></div>
<select id="split"><option value="review">to review (no spares)</option><option value="">all splits</option><option>test</option><option>dev</option><option>safety</option><option>probe</option><option>spare</option></select>
<select id="flagged"><option value="">all items</option><option value="1">flagged only</option><option value="u">undecided only</option></select>
<button id="export">Export decisions</button><label class="help"><input type="file" id="import" accept=".json" hidden><u style="cursor:pointer">Import</u></label></header>
<main><p class="help">Check that each <b>answer key</b> is correct and fair. <b>OK</b> = correct as written. <b>Fix</b> = write the correction in the box. <b>Drop</b> = remove the question.
Keys: <b>j/k</b> next/previous · <b>1</b> OK · <b>2</b> Fix · <b>3</b> Drop. ★ = critical point (missing it makes an answer wrong). Saved automatically in this browser.</p><div id="list"></div></main>
<script>
const ITEMS = __DATA__;
const KEY = "pitwall-review-" + __VERSION__;
let dec = {}; try { dec = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let focusIdx = 0;
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function ref(it){
  let h = "";
  if (it.reference_kind === "numeric") h += `<div><b>Answer: ${esc(it.gold)} ${esc(it.unit)}</b> <span class="help">(${esc(it.expression)})</span></div>` +
     (it.check_python_matches_claude ? `<div class="auto">✓ Python's calculation matches the drafted answer "${esc(it.stated_answer)}"; check the question's wording and the calculation.</div>` : `<div class="flag">⚠ Python and the draft disagree; check carefully.</div>`);
  if (it.reference_kind === "short_answer") h += `<div><b>Answer: ${esc(it.short_answer)}</b>${it.aliases.length ? ` <span class="help">also accept: ${it.aliases.map(esc).join(", ")}</span>` : ""}</div>`;
  if (it.key_points.length) h += "<ul>" + it.key_points.map(k => `<li>${k.critical ? '<span class="crit">★</span> ' : ""}${esc(k.text)}</li>`).join("") + "</ul>";
  if (it.notes) h += `<div class="help">Grader note: ${esc(it.notes)}</div>`;
  return h;
}
function visible(){
  const s = document.getElementById("split").value, f = document.getElementById("flagged").value;
  return ITEMS.filter(it => (!s || (s === "review" ? it.split !== "spare" : it.split === s)) && (!f || (f === "1" ? (it.check_near_duplicate_of.length || it.check_python_matches_claude === false) : !dec[it.id])));
}
function render(){
  const list = document.getElementById("list"), vis = visible();
  list.innerHTML = vis.map((it, i) => {
    const d = dec[it.id] || {};
    return `<section class="card ${d.decision || ""} ${i === focusIdx ? "focus" : ""}" id="c-${it.id}" data-id="${it.id}">
      <div class="meta"><span>${it.id}</span><span>${it.split}</span><span>${it.category}</span><span>${esc(it.subtopic)}</span><span>${esc(it.author)}</span></div>
      <div class="q">${esc(it.question)}</div><div class="ref">${ref(it)}</div>
      ${it.check_near_duplicate_of.length ? `<div class="flag">⚠ Similar wording to ${it.check_near_duplicate_of.join(", ")}; keep both only if they test different things.</div>` : ""}
      <div class="btns"><button class="b-ok ${d.decision === "ok" ? "on" : ""}" data-d="ok">1 · OK</button><button class="b-fix ${d.decision === "fix" ? "on" : ""}" data-d="fix">2 · Fix</button><button class="b-drop ${d.decision === "drop" ? "on" : ""}" data-d="drop">3 · Drop</button></div>
      <textarea placeholder="What's wrong / the corrected answer (needed for Fix)">${esc(d.note || "")}</textarea></section>`;
  }).join("");
  const done = ITEMS.filter(it => dec[it.id]).length;
  const todo = ITEMS.filter(it => it.split !== "spare"), doneReview = todo.filter(it => dec[it.id]).length;
  document.getElementById("count").textContent = `${doneReview} / ${todo.length} reviewed`;
  document.getElementById("bar").style.width = (100 * doneReview / todo.length) + "%";
}
function save(){ try { localStorage.setItem(KEY, JSON.stringify(dec)); } catch (e) {} }
function decide(id, d){ dec[id] = Object.assign(dec[id] || {}, {decision: d}); save(); render(); }
document.getElementById("list").addEventListener("click", e => { const b = e.target.closest("button[data-d]"); if (b) decide(b.closest(".card").dataset.id, b.dataset.d); });
document.getElementById("list").addEventListener("input", e => { if (e.target.tagName === "TEXTAREA") { const id = e.target.closest(".card").dataset.id; dec[id] = Object.assign(dec[id] || {}, {note: e.target.value}); save(); }});
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA" || e.target.tagName === "SELECT") return;
  const vis = visible(); if (!vis.length) return;
  if (e.key === "j" || e.key === "k") { focusIdx = Math.max(0, Math.min(vis.length - 1, focusIdx + (e.key === "j" ? 1 : -1))); render(); document.getElementById("c-" + vis[focusIdx].id).scrollIntoView({block: "center"}); }
  if (["1","2","3"].includes(e.key)) { decide(vis[focusIdx].id, {1: "ok", 2: "fix", 3: "drop"}[e.key]); if (e.key === "2") document.querySelector(`#c-${vis[focusIdx].id} textarea`).focus(); }
});
["split","flagged"].forEach(id => document.getElementById(id).addEventListener("change", () => { focusIdx = 0; render(); }));
document.getElementById("export").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(dec, null, 2)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "review_decisions.json"; a.click();
});
document.getElementById("import").addEventListener("change", async e => { const f = e.target.files[0]; if (f) { dec = Object.assign(dec, JSON.parse(await f.text())); save(); render(); } });
render();
</script></body></html>"""


def main() -> None:
    rows = sorted(read_jsonl(SOURCE), key=lambda r: (ORDER[r["split"]], r["category"], r["id"]))
    version = json.dumps(len(rows))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        PAGE.replace("__DATA__", json.dumps(rows, ensure_ascii=False)).replace("__VERSION__", version), encoding="utf-8"
    )
    print(f"wrote {OUT} with {len(rows)} items")


if __name__ == "__main__":
    main()
