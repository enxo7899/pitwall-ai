"use client";

import { useEffect, useMemo, useState } from "react";
import { Markdown } from "@/components/ui/Markdown";
import { Verdict } from "@/components/ui/Verdict";
import { ARM_META, ARMS, type Arm, CATEGORY_LABEL, type ExplorerItem } from "@/lib/data";
import styles from "./Explorer.module.css";

type Outcome = "all" | "c_only" | "b_only" | "both" | "none" | "handwritten";
const OUTCOMES: { key: Outcome; label: string; test: (i: ExplorerItem) => boolean }[] = [
  { key: "all", label: "All questions", test: () => true },
  { key: "c_only", label: "Only the fine-tuned model succeeded", test: (i) => i.arms.C.joint && !i.arms.B.joint },
  { key: "b_only", label: "Only the prompted model succeeded", test: (i) => i.arms.B.joint && !i.arms.C.joint },
  { key: "both", label: "Both styled models succeeded", test: (i) => i.arms.B.joint && i.arms.C.joint },
  { key: "none", label: "No arm succeeded", test: (i) => ARMS.every((a) => !i.arms[a].joint) },
  { key: "handwritten", label: "Questions I wrote by hand", test: (i) => i.handwritten },
];
const STATUS_LABEL = { present: "present", missing: "missing", contradicted: "contradicted" } as const;

/** Every test question with all three answers and the judge's form, filterable and deep-linkable. */
export function Explorer() {
  const [items, setItems] = useState<ExplorerItem[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [category, setCategory] = useState("all");
  const [outcome, setOutcome] = useState<Outcome>("all");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [arm, setArm] = useState<Arm>("C");

  useEffect(() => {
    fetch("/data/explorer.json")
      .then((r) => r.json())
      .then((data: ExplorerItem[]) => {
        setItems(data);
        const fromHash = window.location.hash.slice(1);
        setSelected(data.some((d) => d.id === fromHash) ? fromHash : data[0].id);
      })
      .catch(() => setFailed(true));
    const onHash = () => setSelected(window.location.hash.slice(1) || null);
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    // keep the selected question visible in the list, e.g. after following a #link from the home page
    document.querySelector(`[data-id="${selected}"]`)?.scrollIntoView({ block: "nearest" });
  }, [selected, items]);

  const visible = useMemo(() => {
    if (!items) return [];
    const test = OUTCOMES.find((o) => o.key === outcome)!.test;
    const q = query.trim().toLowerCase();
    return items.filter(
      (i) => (category === "all" || i.category === category) && test(i) && (!q || i.question.toLowerCase().includes(q)),
    );
  }, [items, category, outcome, query]);

  if (failed) return <p className={styles.empty}>The results could not be loaded. Please refresh the page.</p>;
  if (!items) return <p className={styles.empty}>Loading the 120 test questions…</p>;

  const item = items.find((i) => i.id === selected) ?? visible[0] ?? items[0];
  const categories = Object.keys(CATEGORY_LABEL);

  function choose(id: string) {
    setSelected(id);
    history.replaceState(null, "", `#${id}`);
    if (window.innerWidth < 1000) document.getElementById("detail")?.scrollIntoView({ behavior: "smooth" });
  }

  return (
    <div className={styles.explorer}>
      <div className={styles.filters}>
        <input
          type="search"
          placeholder="Search questions"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search questions"
        />
        <select value={outcome} onChange={(e) => setOutcome(e.target.value as Outcome)} aria-label="Filter by outcome">
          {OUTCOMES.map((o) => (
            <option key={o.key} value={o.key}>
              {o.label}
            </option>
          ))}
        </select>
        <div className={styles.cats} role="group" aria-label="Filter by category">
          {["all", ...categories].map((c) => (
            <button key={c} type="button" aria-pressed={category === c} onClick={() => setCategory(c)}>
              {c === "all" ? "All" : CATEGORY_LABEL[c]}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.split}>
        <div className={styles.list}>
          <p className={`${styles.count} label`}>
            {visible.length} of {items.length} · joint success A B C
          </p>
          <ul>
            {visible.map((i) => (
              <li key={i.id}>
                <button
                  type="button"
                  data-id={i.id}
                  onClick={() => choose(i.id)}
                  aria-current={i.id === item.id ? "true" : undefined}
                >
                  <span className={styles.rowTop}>
                    <span className="mono">{i.id}</span>
                    <span
                      className={styles.lights}
                      aria-label={ARMS.map((a) => `${a} ${i.arms[a].joint ? "success" : "fail"}`).join(", ")}
                    >
                      {ARMS.map((a) => (
                        <span key={a} data-ok={i.arms[a].joint} />
                      ))}
                    </span>
                  </span>
                  <span className={styles.rowQ}>{i.question}</span>
                </button>
              </li>
            ))}
            {visible.length === 0 && <li className={styles.none}>No questions match these filters.</li>}
          </ul>
        </div>

        <section id="detail" className={styles.detail} aria-label="Selected question">
          <header className={styles.qHead}>
            <span className="label">
              {item.id} · {CATEGORY_LABEL[item.category]}
              {item.handwritten && " · written by hand"}
            </span>
            <h2>{item.question}</h2>
          </header>

          <div className={styles.key}>
            <p className="label">Answer key</p>
            <p className={styles.keyText}>{item.answer_key}</p>
            <ol className={styles.points}>
              {item.key_points.map((k, n) => (
                <li key={k.text}>
                  <span className={styles.statuses}>
                    {ARMS.map((a) => (
                      <span
                        key={a}
                        data-status={item.arms[a].key_points[n]}
                        title={`${a}: ${STATUS_LABEL[item.arms[a].key_points[n]]}`}
                      />
                    ))}
                  </span>
                  <span>
                    {k.critical && <b className={styles.critical}>critical</b>} {k.text}
                  </span>
                </li>
              ))}
            </ol>
            <p className={styles.keyLegend}>
              Squares: A · B · C. <i data-status="present" /> present <i data-status="missing" /> missing{" "}
              <i data-status="contradicted" /> contradicted
            </p>
          </div>

          <div className={styles.tabs} role="tablist" aria-label="Arm">
            {ARMS.map((a) => (
              <button key={a} type="button" role="tab" aria-selected={arm === a} onClick={() => setArm(a)} data-arm={a}>
                <span className="swatch" style={{ background: ARM_META[a].color }} />
                {ARM_META[a].name}
                <span className={styles.tabMark} data-ok={item.arms[a].joint}>
                  {item.arms[a].joint ? "✓" : "✗"}
                </span>
              </button>
            ))}
          </div>

          <ArmPanel v={item.arms[arm]} />
        </section>
      </div>
    </div>
  );
}

function ArmPanel({ v }: { v: ExplorerItem["arms"][Arm] }) {
  return (
    <div className={styles.panel} role="tabpanel">
      <Verdict v={v} />
      <dl className={styles.evidence}>
        <div>
          <dt>Judge on style</dt>
          <dd>{v.style_evidence}</dd>
        </div>
        <div>
          <dt>Judge on helpfulness</dt>
          <dd>{v.helpful_evidence}</dd>
        </div>
        {v.error_quote && (
          <div>
            <dt>Extra error · {v.error_serious ? "serious, fails the answer" : "minor, recorded only"}</dt>
            <dd>&ldquo;{v.error_quote}&rdquo;</dd>
          </div>
        )}
        {v.extracted && (
          <div>
            <dt>Final answer read by the judge</dt>
            <dd className="mono">{v.extracted}</dd>
          </div>
        )}
      </dl>
      <div className={styles.answer}>
        <p className="label">
          Answer · {v.tokens} tokens{v.finished ? "" : " · cut off at the limit"}
        </p>
        <Markdown>{v.answer}</Markdown>
      </div>
    </div>
  );
}
