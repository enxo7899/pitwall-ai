"""Adjudication: the labeller takes a second, unblinded look at every pass/fail disagreement with the judge.

The blind labels (data/labels/blind_labels.json) stay untouched and remain the basis of the reported
human-judge agreement. This sheet records a separate, adjudicated decision per disputed field, made with
the judge's evidence on screen, so the report can show both numbers.

Usage:
    uv run python scripts/evaluation/build_recheck_sheet.py
Writes data/labels/recheck_sheet.html; its "Export" button saves recheck_decisions.json.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from pitwall.eval_items import read_jsonl

LABELS = Path("data/labels")
JUDGEMENTS = {
    "A": "experiments/results/judgements/exam-c500_judgements.jsonl",
    "B": "experiments/results/judgements/exam-c500_judgements.jsonl",
    "C": "experiments/results/judgements/exam-c315_judgements.jsonl",
}
ANSWERS = {
    "A": "experiments/results/answers/test_A_B_C3epoch.jsonl",
    "B": "experiments/results/answers/test_A_B_C3epoch.jsonl",
    "C": "experiments/results/answers/test_C.jsonl",
}

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PITWALL Re-check</title><style>
:root{--bg:#f6f6f4;--card:#fff;--fg:#1b1b19;--muted:#66665f;--line:#e0e0d9;--accent:#c8102e;--soft:#f1efe8}
@media (prefers-color-scheme:dark){:root{--bg:#151514;--card:#1e1e1c;--fg:#ececea;--muted:#a3a39c;--line:#34342f;--accent:#ff5a6e;--soft:#262623}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,sans-serif}
header{position:sticky;top:0;z-index:2;background:var(--bg);border-bottom:1px solid var(--line);padding:10px max(16px,4vw);display:flex;gap:12px;align-items:center;flex-wrap:wrap}
main{max-width:900px;margin:0 auto;padding:16px max(16px,4vw)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 18px}
.q{font-weight:600}.meta,.help{color:var(--muted);font-size:13px}
.key{background:var(--soft);border-radius:8px;padding:8px 12px;font-size:14px;margin:8px 0}
.ans{white-space:pre-wrap;border-left:3px solid var(--line);padding-left:12px;margin:10px 0;max-height:420px;overflow:auto}
.field{border-top:1px dashed var(--line);padding-top:10px;margin-top:10px}
.why{font-size:14px;margin:4px 0 8px}
button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:4px 10px;cursor:pointer;margin:2px}
button.on{background:var(--accent);border-color:var(--accent);color:#fff}
</style></head><body>
<header><b>Re-check</b><span id="count"></span><button id="export">Export decisions</button></header>
<main><p class="help">Each card is an answer where your blind label and the judge disagree on pass/fail. Read the answer against the
answer key, read the judge's reason, then choose the value you now think is right. Your original label is pre-selected.
Choosing the judge's value is fine when the judge found something you missed; keeping yours is fine when you still disagree.</p>
<div id="list"></div></main>
<script>
const CARDS = __DATA__;
const KEY = "pitwall-recheck-" + CARDS.length;
let dec = {}; try { dec = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const scale = f => f === "correct" ? ["yes", "no"] : [0, 1, 2, 3];
const value = (c, f) => (dec[c.id] && f in dec[c.id]) ? dec[c.id][f] : c.yours[f];
const done = c => c.fields.every(f => dec[c.id] && f in dec[c.id]);
function render() {
  document.getElementById("count").textContent = `${CARDS.filter(done).length} / ${CARDS.length} decided`;
  document.getElementById("list").innerHTML = CARDS.map(c => `<div class="card"><div class="meta">${c.id} · ${c.category}</div>
    <div class="q">${c.question}</div><div class="key">${c.key}</div><div class="ans">${c.answer}</div>` +
    c.fields.map(f => `<div class="field"><b>${f}</b> · you said <b>${c.yours[f]}</b>, the judge said <b>${c.judge[f]}</b>
      <div class="why">Judge: ${c.why[f]}</div>` + scale(f).map(v => `<button data-id="${c.id}" data-f="${f}" data-v="${v}"
      class="${value(c, f) === v && dec[c.id] && f in dec[c.id] ? "on" : ""}">${v}</button>`).join("") + `</div>`).join("") + `</div>`).join("");
}
document.getElementById("list").addEventListener("click", e => {
  const b = e.target.closest("button[data-id]"); if (!b) return;
  const v = isNaN(+b.dataset.v) ? b.dataset.v : +b.dataset.v;
  dec[b.dataset.id] = Object.assign(dec[b.dataset.id] || {}, {[b.dataset.f]: v});
  try { localStorage.setItem(KEY, JSON.stringify(dec)); } catch (e) {}
  render();
});
document.getElementById("export").addEventListener("click", () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(dec, null, 2)], {type: "application/json"}));
  a.download = "recheck_decisions.json"; a.click();
});
render();
</script></body></html>"""


def answer_key(item: dict) -> str:
    head = ""
    if item["reference_kind"] == "numeric":
        head = f"<b>Answer:</b> {item['gold']} {html.escape(item['unit'])}<br>"
    elif item["reference_kind"] == "short_answer":
        head = f"<b>Answer:</b> {html.escape(item['short_answer'])}<br>"
    points = "".join(f"<li>{'★ ' if k['critical'] else ''}{html.escape(k['text'])}</li>" for k in item["key_points"])
    return f"{head}<b>Answer key</b> (★ = must-have)<ul>{points}</ul>"


def judge_reason(form: dict, field: str, item: dict) -> str:
    if field == "style":
        return html.escape(form["style_evidence"])
    if field == "helpful":
        return html.escape(form["helpfulness_evidence"])
    statuses = ", ".join(f"point {p['point']}: {p['status']}" for p in form["key_points"])
    extra = (
        f' Extra error ({"serious" if form["error_is_material"] else "minor"}): "{html.escape(form["error_quote"])}"'
        if form["other_factual_error"]
        else ""
    )
    found = f" Final answer found: {html.escape(form['extracted_answer'])}." if form["extracted_answer"] else ""
    return f"Key points: {statuses}.{found}{extra}"


def main() -> None:
    labels = json.loads((LABELS / "blind_labels.json").read_text())
    key = json.loads((LABELS / "blind_key.json").read_text())
    items = {r["id"]: r for r in read_jsonl(Path("data/eval/test.jsonl"))}
    judged = {
        (a, r["item_id"]): r
        for a, p in JUDGEMENTS.items()
        for r in read_jsonl(Path(p))
        if r["arm"] == a and r["sample"] == 0
    }
    answers = {
        (a, g["item_id"]): g["answer"]
        for a, p in ANSWERS.items()
        for g in read_jsonl(Path(p))
        if g["arm"] == a and g["sample"] == 0
    }

    cards = []
    for card_id, meta in sorted(key.items()):
        if meta["part"] != "label":
            continue
        mine, judge = labels[card_id], judged[(meta["arm"], meta["item_id"])]
        theirs = {"style": judge["style"], "helpful": judge["helpful"], "correct": "yes" if judge["correct"] else "no"}
        disputed = [
            f
            for f in ("correct", "helpful", "style")
            if (f == "correct" and mine[f] != theirs[f])
            or (f != "correct" and (mine[f] >= 2) != ((theirs[f] or 0) >= 2))
        ]
        if not disputed:
            continue
        item = items[meta["item_id"]]
        cards.append(
            {
                "id": card_id,
                "category": item["category"],
                "question": html.escape(item["question"]),
                "key": answer_key(item),
                "answer": html.escape(answers[(meta["arm"], meta["item_id"])]),
                "fields": disputed,
                "yours": {f: mine[f] for f in disputed},
                "judge": {f: theirs[f] for f in disputed},
                "why": {f: judge_reason(judge["form"], f, item) for f in disputed},
            }
        )
    (LABELS / "recheck_sheet.html").write_text(
        PAGE.replace("__DATA__", json.dumps(cards, ensure_ascii=False)), encoding="utf-8"
    )
    print(
        f"wrote {LABELS / 'recheck_sheet.html'}: {len(cards)} answers, {sum(len(c['fields']) for c in cards)} disputed scores"
    )


if __name__ == "__main__":
    main()
