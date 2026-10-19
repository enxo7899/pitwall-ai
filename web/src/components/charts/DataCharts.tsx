import type { SiteData } from "@/lib/data";
import { linear } from "@/lib/scale";
import c from "./chart.module.css";

const W = 760;

/** From 924 generated questions to 500 training examples, with what each stage removed. */
export function Funnel({ data }: { data: SiteData["data"] }) {
  const stages = [
    { label: "Questions written by the teacher", n: data.requested, note: "" },
    {
      label: "After removing near-duplicates",
      n: data.requested - data.near_duplicates,
      note: `−${data.near_duplicates}`,
    },
    { label: "After removing look-alikes of exam questions", n: data.answered, note: `−${data.eval_lookalikes}` },
    {
      label: "Passed the checker and the code gates",
      n: data.accepted,
      note: `−${data.answered - data.accepted} rejected`,
    },
    {
      label: "Selected for training, balanced by category",
      n: data.selected,
      note: `${data.heldout} kept for validation`,
    },
  ];
  const row = 46;
  const H = stages.length * row + 8;
  const x = linear([0, data.requested], [0, W - 330]);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={c.svg}
      role="img"
      aria-label="Training data funnel from 924 questions to 500 examples"
    >
      {stages.map((s, i) => {
        const last = i === stages.length - 1;
        const y = i * row + 4;
        return (
          <g key={s.label}>
            <text x={0} y={y + 14} className={c.labelMuted}>
              {s.label}
            </text>
            <rect
              x={0}
              y={y + 22}
              width={x(s.n)}
              height={12}
              rx={1.5}
              fill={last ? "var(--amber)" : "var(--line-strong)"}
            />
            {last && (
              <rect
                x={x(s.n)}
                y={y + 22}
                width={x(data.heldout)}
                height={12}
                rx={1.5}
                fill="var(--series-val)"
                opacity={0.55}
              />
            )}
            <text
              x={x(last ? s.n + data.heldout : s.n) + 10}
              y={y + 32}
              className={c.value}
              fill={last ? "var(--amber)" : undefined}
            >
              {s.n}
            </text>
            {s.note && (
              <text x={W} y={y + 32} textAnchor="end" className={c.tick}>
                {s.note}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

const REASONS: Record<string, { label: string; by: "checker" | "code" }> = {
  "checker:information_lost": { label: "A fact, number, step or warning lost", by: "checker" },
  too_long_vs_reference: { label: "Longer than 1.3× the plain answer", by: "code" },
  numbers_missing: { label: "A number from the plain answer missing", by: "code" },
  "checker:claims_added": { label: "New claims added", by: "checker" },
  "checker:meaning_blurred": { label: "Meaning blurred by a metaphor", by: "checker" },
  code_changed: { label: "A code block changed", by: "code" },
  "checker:reference_incorrect": { label: "The teacher's own plain answer was wrong", by: "checker" },
  too_little_style: { label: "Too little F1 language", by: "code" },
  too_long_absolute: { label: "Longer than 450 words", by: "code" },
  "checker:order_changed": { label: "Order of steps changed", by: "checker" },
  question_mentions_racing: { label: "Question itself mentioned racing", by: "code" },
};

/** Why rewrites were rejected; one example can fail several checks. */
export function RejectReasons({ reasons }: { reasons: Record<string, number> }) {
  const rows = Object.entries(reasons).sort((a, b) => b[1] - a[1]);
  const row = 26;
  const H = rows.length * row + 4;
  const max = rows[0][1];
  const x = linear([0, max], [0, W - 380]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={c.svg} role="img" aria-label="Reasons rewrites were rejected">
      {rows.map(([key, n], i) => {
        const meta = REASONS[key] ?? { label: key, by: "code" as const };
        const y = i * row;
        return (
          <g key={key}>
            <text x={290} y={y + 15} textAnchor="end" className={c.labelMuted}>
              {meta.label}
            </text>
            <rect
              x={304}
              y={y + 6}
              width={x(n)}
              height={11}
              rx={1.5}
              fill={meta.by === "checker" ? "var(--series-val)" : "var(--ink-3)"}
            />
            <text x={304 + x(n) + 8} y={y + 15.5} className={c.value}>
              {n}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
