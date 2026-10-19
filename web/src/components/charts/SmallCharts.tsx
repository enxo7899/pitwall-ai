import { ARM_META, ARMS, type SiteData } from "@/lib/data";
import styles from "./SmallCharts.module.css";

/** Horizontal bars for one measure per arm, drawn with CSS so they reflow on phones. */
function ArmBars({
  title,
  values,
  format,
  max,
}: {
  title: string;
  values: Record<string, number>;
  format: (v: number) => string;
  max: number;
}) {
  return (
    <div className={styles.group}>
      <p className="label">{title}</p>
      {ARMS.map((arm) => (
        <div key={arm} className={styles.row}>
          <span className={styles.name}>{ARM_META[arm].short}</span>
          <span className={styles.track}>
            <span
              className={styles.bar}
              style={{ width: `${(values[arm] / max) * 100}%`, background: ARM_META[arm].color }}
            />
          </span>
          <span className={`${styles.num} mono tabular`}>{format(values[arm])}</span>
        </div>
      ))}
    </div>
  );
}

export function LatencyBars({ latency }: { latency: SiteData["latency"] }) {
  const pick = (key: "median_prompt_tokens" | "median_first_token_ms" | "median_decode_tokens_per_s") =>
    Object.fromEntries(ARMS.map((a) => [a, latency[a][key]]));
  return (
    <div className={styles.three}>
      <ArmBars
        title="Prompt tokens per request"
        values={pick("median_prompt_tokens")}
        format={(v) => `${v}`}
        max={500}
      />
      <ArmBars
        title="Time to first token"
        values={pick("median_first_token_ms")}
        format={(v) => `${Math.round(v)} ms`}
        max={460}
      />
      <ArmBars
        title="Decoding speed"
        values={pick("median_decode_tokens_per_s")}
        format={(v) => `${v.toFixed(1)} tok/s`}
        max={34}
      />
    </div>
  );
}

export function ToneBars({ safety }: { safety: SiteData["safety"] }) {
  const tone = Object.fromEntries(ARMS.map((a) => [a, safety.tone_acceptable[a][0]]));
  const judge = Object.fromEntries(ARMS.map((a) => [a, safety.judge_correct_and_safe[a][0]]));
  return (
    <div className={styles.two}>
      <ArmBars title="Acceptable tone, my strict rule (of 16)" values={tone} format={(v) => `${v}/16`} max={16} />
      <ArmBars
        title="Judge: every critical step, nothing unsafe (of 16)"
        values={judge}
        format={(v) => `${v}/16`}
        max={16}
      />
    </div>
  );
}

const CANARY_LABEL: Record<string, string> = {
  number_changed: "A number changed",
  fact_contradicted: "A fact contradicted",
  critical_step_removed: "A critical step removed",
  style_opener_only: "Style only in the first line",
  no_style: "No F1 style at all",
  padded_verbose: "Padded, verbose answer",
  minor_slip: "Harmless false aside",
};

/** Planted mistakes by type: every cell is one canary answer, green when the judge caught it. */
export function CanaryGrid({ canaries }: { canaries: Record<string, string> }) {
  return (
    <div className={styles.canaries}>
      {Object.entries(canaries).map(([type, score]) => {
        const [caught, total] = score.split("/").map(Number);
        return (
          <div key={type} className={styles.canary}>
            <span className={styles.cells} aria-label={`${caught} of ${total} caught`}>
              {Array.from({ length: total }, (_, i) => (
                <span key={i} data-caught={i < caught} />
              ))}
            </span>
            <span>{CANARY_LABEL[type] ?? type}</span>
          </div>
        );
      })}
    </div>
  );
}

const SPLITS: { key: string; label: string; use: string; color: string }[] = [
  { key: "test", label: "Test", use: "the only results reported", color: "var(--amber)" },
  { key: "dev", label: "Dev", use: "prompt tuning and epoch choice", color: "var(--series-val)" },
  { key: "safety", label: "Safety", use: "serious situations", color: "var(--fail)" },
  { key: "probe", label: "Probes", use: "behaviour checks", color: "var(--ink-3)" },
];

/** The 196 evaluation items as one bar, split by purpose. */
export function EvalComposition({ splits, handwritten }: { splits: Record<string, number>; handwritten: number }) {
  const total = Object.values(splits).reduce((a, b) => a + b, 0);
  return (
    <div className={styles.composition}>
      <div className={styles.stack} role="img" aria-label={`${total} evaluation items by split`}>
        {SPLITS.map((s) => (
          <span key={s.key} style={{ flexGrow: splits[s.key], background: s.color }}>
            {s.key === "test" && (
              <i style={{ width: `${(handwritten / splits.test) * 100}%` }} title={`${handwritten} hand-written`} />
            )}
          </span>
        ))}
      </div>
      <dl className={styles.splits}>
        {SPLITS.map((s) => (
          <div key={s.key}>
            <dt>
              <span className="swatch" style={{ background: s.color }} /> {s.label}
            </dt>
            <dd>
              <b className="mono tabular">{splits[s.key]}</b> {s.use}
              {s.key === "test" && <>, {handwritten} written by hand (hatched)</>}
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
