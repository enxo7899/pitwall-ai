"use client";

import Link from "next/link";
import { useRef, useState } from "react";
import { Markdown } from "@/components/ui/Markdown";
import { StartLights } from "@/components/ui/StartLights";
import { Verdict } from "@/components/ui/Verdict";
import { ARM_META, type Arm, type ArmVerdict, CATEGORY_LABEL, type ExplorerItem } from "@/lib/data";
import { type AnswerStats, streamAnswer } from "@/lib/engine";
import styles from "./LiveConsole.module.css";

type Mode = "single" | "compare";
type Slot =
  | { state: "empty" }
  | { state: "waiting"; waking: boolean }
  | { state: "streaming" | "done"; text: string; stats?: AnswerStats }
  | { state: "recorded"; text: string; verdict: ArmVerdict; itemId: string }
  | { state: "error"; message: string };

const EMPTY: Record<Arm, Slot> = { A: { state: "empty" }, B: { state: "empty" }, C: { state: "empty" } };
const LIVE_ORDER: Arm[] = ["C", "B", "A"]; // the fine-tuned answer first; the GPU serves one request at a time

export function LiveConsole({ showcase }: { showcase: ExplorerItem[] }) {
  const [mode, setMode] = useState<Mode>("single");
  const [question, setQuestion] = useState("");
  const [slots, setSlots] = useState<Record<Arm, Slot>>(EMPTY);
  const [busy, setBusy] = useState(false);
  const run = useRef(0); // a newer run makes older streams stop writing

  const arms: Arm[] = mode === "single" ? ["C"] : ["A", "B", "C"];
  const update = (arm: Arm, slot: Slot) => setSlots((prev) => ({ ...prev, [arm]: slot }));

  function showRecorded(item: ExplorerItem) {
    run.current += 1;
    setBusy(false);
    setQuestion(item.question);
    setSlots({
      A: { state: "recorded", text: item.arms.A.answer, verdict: item.arms.A, itemId: item.id },
      B: { state: "recorded", text: item.arms.B.answer, verdict: item.arms.B, itemId: item.id },
      C: { state: "recorded", text: item.arms.C.answer, verdict: item.arms.C, itemId: item.id },
    });
  }

  async function ask() {
    const q = question.trim();
    if (!q || busy) return;
    const id = ++run.current;
    setBusy(true);
    setSlots({ ...EMPTY, ...Object.fromEntries(arms.map((a) => [a, { state: "waiting", waking: false }])) });
    for (const arm of LIVE_ORDER.filter((a) => arms.includes(a))) {
      try {
        let last: { text: string; stats: AnswerStats } | null = null;
        for await (const chunk of streamAnswer(q, arm, () => update(arm, { state: "waiting", waking: true }))) {
          if (run.current !== id) return;
          last = chunk;
          update(arm, { state: chunk.stats.done ? "done" : "streaming", text: chunk.text, stats: chunk.stats });
        }
        // a stream cut off mid-answer still ends in a settled card, without a blinking cursor
        if (last && !last.stats.done) update(arm, { state: "done", text: last.text });
      } catch (error) {
        if (run.current !== id) return;
        const failed: Slot = { state: "error", message: error instanceof Error ? error.message : String(error) };
        // quota or engine errors affect every arm alike, so the arms still waiting get the same message
        setSlots((prev) => ({
          A: prev.A.state === "waiting" ? failed : prev.A,
          B: prev.B.state === "waiting" ? failed : prev.B,
          C: prev.C.state === "waiting" ? failed : prev.C,
          [arm]: failed,
        }));
        break;
      }
    }
    if (run.current === id) setBusy(false);
  }

  return (
    <div className={`panel ${styles.console}`}>
      <div className="panel-head">
        <span className="label">Team radio · ask the model</span>
        <div className={styles.modes} role="radiogroup" aria-label="Which models answer">
          {(["single", "compare"] as Mode[]).map((m) => (
            <button key={m} type="button" role="radio" aria-checked={mode === m} onClick={() => setMode(m)}>
              {m === "single" ? "Fine-tuned" : "Compare all three"}
            </button>
          ))}
        </div>
      </div>

      <form
        className={styles.form}
        onSubmit={(e) => {
          e.preventDefault();
          ask();
        }}
      >
        <label htmlFor="question" className="visually-hidden">
          Your question
        </label>
        <textarea
          id="question"
          value={question}
          maxLength={1000}
          rows={2}
          placeholder="Ask anything: a recipe, a spreadsheet formula, a tricky email…"
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              ask();
            }
          }}
        />
        <button type="submit" className="btn btn-primary" disabled={busy || !question.trim()}>
          {busy ? "On track…" : "Ask live"}
        </button>
      </form>

      <div className={styles.samples}>
        <span className="label">From the test set, with the judge's verdict</span>
        <div className={styles.chips}>
          {showcase.map((item) => (
            <button key={item.id} type="button" className={styles.chip} onClick={() => showRecorded(item)}>
              <span className={styles.chipCat}>{CATEGORY_LABEL[item.category]}</span>
              {item.question}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.outputs} data-mode={mode}>
        {arms.map((arm) => (
          <AnswerCard key={arm} arm={arm} slot={slots[arm]} />
        ))}
      </div>

      <p className={styles.note}>
        Live answers come from the evaluated model, prompt and sampling settings, capped at 512 tokens, so wording
        varies between runs. The engine runs on Hugging Face ZeroGPU hardware, which lends it a GPU for each question;
        the first question after a quiet spell can take up to a minute.
      </p>
    </div>
  );
}

function AnswerCard({ arm, slot }: { arm: Arm; slot: Slot }) {
  const meta = ARM_META[arm];
  return (
    <article className={styles.card} data-arm={arm} aria-live="polite">
      <header className={styles.cardHead}>
        <span className={styles.cardArm}>
          <span className="swatch" style={{ background: meta.color }} />
          {meta.name}
        </span>
        <span className="label">{badge(slot)}</span>
      </header>
      <div className={styles.cardBody}>
        {slot.state === "empty" && (
          <p className={styles.placeholder}>
            {arm === "C" ? "Ask a question, or pick one from the test set above." : "Waiting for a question."}
          </p>
        )}
        {slot.state === "waiting" && (
          <StartLights label={slot.waking ? "Starting the GPU · can take up to a minute" : "Waiting for the GPU"} />
        )}
        {(slot.state === "streaming" || slot.state === "done" || slot.state === "recorded") && (
          <Markdown>{slot.text + (slot.state === "streaming" ? " ▍" : "")}</Markdown>
        )}
        {slot.state === "error" && <p className={styles.error}>{slot.message}</p>}
      </div>
      {slot.state === "recorded" && (
        <footer className={styles.cardFoot}>
          <Verdict v={slot.verdict} />
          <Link href={`/explorer#${slot.itemId}`} className={styles.why}>
            Judge&apos;s notes →
          </Link>
        </footer>
      )}
      {slot.state === "done" && slot.stats?.done && (
        <footer className={`${styles.cardFoot} mono tabular`}>
          <span>{slot.stats.tokens} tokens</span>
          <span>{slot.stats.seconds?.toFixed(1)} s</span>
          <span>{slot.stats.tokens_per_second} tok/s</span>
          <span>{slot.stats.finished ? "finished" : `cut at ${slot.stats.max_new_tokens} tokens`}</span>
        </footer>
      )}
    </article>
  );
}

function badge(slot: Slot): string {
  switch (slot.state) {
    case "recorded":
      return "Recorded in evaluation";
    case "waiting":
      return "Queued";
    case "streaming":
      return "Live";
    case "done":
      return "Live · done";
    case "error":
      return "No answer";
    default:
      return "";
  }
}
