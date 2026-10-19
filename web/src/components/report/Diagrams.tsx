import { ARM_META, ARMS, type Arm } from "@/lib/data";
import c from "../charts/chart.module.css";
import styles from "./Diagrams.module.css";

const ARM_INPUTS: Record<Arm, { prompt: boolean; adapter: boolean }> = {
  A: { prompt: false, adapter: false },
  B: { prompt: true, adapter: false },
  C: { prompt: false, adapter: true },
};

/** What each arm receives: the only differences are the system prompt (B) and the adapter (C). */
export function ArmsDiagram() {
  return (
    <div className={styles.arms}>
      {ARMS.map((arm) => {
        const io = ARM_INPUTS[arm];
        return (
          <div key={arm} className={styles.armRow} data-arm={arm}>
            <span className={styles.armTag}>
              <span className="swatch" style={{ background: ARM_META[arm].color }} />
              <b>{arm}</b> {ARM_META[arm].name}
            </span>
            <span className={styles.flow}>
              <span className={styles.box} data-off={!io.prompt}>
                {io.prompt ? "F1 prompt · 451 tokens" : "no prompt"}
              </span>
              <span className={styles.plus}>+</span>
              <span className={styles.box}>question</span>
              <span className={styles.arrow} aria-hidden="true">
                →
              </span>
              <span className={styles.box} data-model>
                Qwen3-1.7B
                {io.adapter && <span className={styles.adapter}>+ LoRA adapter</span>}
              </span>
              <span className={styles.arrow} aria-hidden="true">
                →
              </span>
              <span className={styles.box}>answer</span>
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** LoRA in one picture: the big weight matrix stays frozen; two thin matrices learn a correction. */
export function LoraDiagram() {
  return (
    <svg
      viewBox="0 0 760 250"
      className={c.svg}
      role="img"
      aria-label="LoRA: a frozen weight matrix W plus a trained low-rank update B times A"
    >
      <defs>
        <pattern id="frozen" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke="var(--line-strong)" strokeWidth="2" />
        </pattern>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--ink-3)" />
        </marker>
      </defs>
      <text x="30" y="128" className={c.label}>
        x
      </text>
      <path d="M48 124 H120 V62 H200" fill="none" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd="url(#arrow)" />
      <path d="M120 124 V190 H200" fill="none" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd="url(#arrow)" />
      <rect x="204" y="18" width="250" height="88" fill="url(#frozen)" stroke="var(--line-strong)" rx="2" />
      <text x="329" y="56" textAnchor="middle" className={c.label}>
        W · pretrained weight
      </text>
      <text x="329" y="76" textAnchor="middle" className={c.tick}>
        frozen · 2,048 × 2,048 in a typical layer
      </text>
      <path d="M204 158 L274 178 L274 202 L204 222 Z" fill="var(--amber)" opacity="0.85" />
      <text x="239" y="243" textAnchor="middle" className={c.tickStrong}>
        A · 16 × 2,048
      </text>
      <path d="M384 178 L454 158 L454 222 L384 202 Z" fill="var(--amber)" opacity="0.85" />
      <text x="419" y="243" textAnchor="middle" className={c.tickStrong}>
        B · 2,048 × 16
      </text>
      <path d="M276 190 H382" fill="none" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd="url(#arrow)" />
      <text x="329" y="182" textAnchor="middle" className={c.tick}>
        rank 16
      </text>
      <path d="M456 62 H540 V112" fill="none" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd="url(#arrow)" />
      <path d="M456 190 H540 V138" fill="none" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd="url(#arrow)" />
      <circle cx="540" cy="125" r="12" fill="var(--bg-panel)" stroke="var(--ink-2)" strokeWidth="1.5" />
      <text x="540" y="130" textAnchor="middle" className={c.label}>
        +
      </text>
      <path d="M552 125 H600" fill="none" stroke="var(--ink-3)" strokeWidth="1.5" markerEnd="url(#arrow)" />
      <text x="610" y="121" className={c.label}>
        h = Wx + 2·BAx
      </text>
      <text x="610" y="141" className={c.tick}>
        α / r = 32 / 16 = 2
      </text>
      <text x="470" y="210" className={c.tick}>
        trained · B starts at zero,
      </text>
      <text x="470" y="224" className={c.tick}>
        so step 0 equals the base model
      </text>
    </svg>
  );
}

/** A numbered chain of steps, used for the data pipeline and the judging flow. */
export function Steps({ steps }: { steps: { title: string; body: string; tag?: string }[] }) {
  return (
    <ol className={styles.steps}>
      {steps.map((s, i) => (
        <li key={s.title}>
          <span className={styles.stepNo}>{String(i + 1).padStart(2, "0")}</span>
          <span className={styles.stepText}>
            <b>{s.title}</b>
            {s.tag && <span className={styles.stepTag}>{s.tag}</span>}
            <span>{s.body}</span>
          </span>
        </li>
      ))}
    </ol>
  );
}
