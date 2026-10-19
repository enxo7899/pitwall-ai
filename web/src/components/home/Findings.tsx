import { site } from "@/lib/data";
import { pct, signed } from "@/lib/format";
import styles from "./Findings.module.css";

/** The four findings, one per "sector", each with its honest qualifier. */
export function Findings() {
  const { arms, comparisons, latency } = site;
  const style = comparisons.C_vs_B.style;
  const joint = comparisons.C_vs_B.joint_success;
  const sectors = [
    {
      tag: "S1 · Style",
      headline: `${pct(arms.C.style.rate)} vs ${pct(arms.B.style.rate)}`,
      body: `of answers reach the F1 style threshold: fine-tuned vs prompted. A gap of ${signed(style.diff)} points, 95% CI ${signed(style.low)} to ${signed(style.high)}.`,
      tone: "good",
    },
    {
      tag: "S2 · Speed",
      headline: `${Math.round(latency.C.median_first_token_ms)} ms vs ${Math.round(latency.B.median_first_token_ms)} ms`,
      body: `to the first token. The prompt resends ${latency.B.median_prompt_tokens - latency.C.median_prompt_tokens} tokens with every request; the adapter needs none.`,
      tone: "good",
    },
    {
      tag: "S3 · Accuracy",
      headline: `${pct(arms.C.correct.rate)} vs ${pct(arms.B.correct.rate)}`,
      body: `correct: no reliable difference. Both trail the plain base model (${pct(arms.A.correct.rate)}): on a 1.7B model, style costs accuracy.`,
      tone: "flat",
    },
    {
      tag: "S4 · Overall",
      headline: `${signed(joint.diff)} pts`,
      body: `joint success over the prompt. The 95% interval (${signed(joint.low)} to ${signed(joint.high)}) crosses zero, and under a stricter error rule the two tie at ${arms.C.joint_strict.s} of ${arms.C.joint_strict.n}: a lead, not a proven win.`,
      tone: "flat",
    },
  ];
  return (
    <div className={styles.grid}>
      {sectors.map((s) => (
        <article key={s.tag} className={styles.sector} data-tone={s.tone}>
          <span className="label">{s.tag}</span>
          <p className={`${styles.headline} mono tabular`}>{s.headline}</p>
          <p className={styles.body}>{s.body}</p>
        </article>
      ))}
    </div>
  );
}
