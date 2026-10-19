import type { ArmVerdict } from "@/lib/data";
import styles from "./Verdict.module.css";

/** A 0–3 score as four pips; pips at or above the pass mark (2) are lit. */
export function Pips({ value, label }: { value: number | null; label: string }) {
  return (
    <span
      className={styles.pips}
      aria-label={`${label} ${value ?? "not scored"} of 3`}
      title={`${label}: ${value ?? "–"}/3`}
    >
      {[1, 2, 3].map((i) => (
        <span
          key={i}
          className={styles.pip}
          data-on={value !== null && i <= value}
          data-pass={value !== null && value >= 2}
        />
      ))}
    </span>
  );
}

/** The judge's verdict on one answer, as compact timing-screen style badges. */
export function Verdict({ v }: { v: ArmVerdict }) {
  return (
    <div className={styles.verdict}>
      <span className={styles.item}>
        <span className="label">Style</span>
        <Pips value={v.style} label="Style" />
      </span>
      <span className={styles.item}>
        <span className="label">Helpful</span>
        <Pips value={v.helpful} label="Helpfulness" />
      </span>
      <span className={styles.item}>
        <span className="label">Correct</span>
        <span className={styles.mark} data-ok={v.correct === true}>
          {v.correct ? "Yes" : "No"}
        </span>
      </span>
      <span className={styles.joint} data-ok={v.joint}>
        {v.joint ? "Joint success" : "Not a joint success"}
      </span>
    </div>
  );
}
