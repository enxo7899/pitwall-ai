import { linear } from "@/lib/scale";
import c from "./chart.module.css";

type Point = { examples: number; style: number; joint: number; correct: number };
const W = 760;
const H = 260;
const M = { l: 46, r: 120, t: 18, b: 38 };
const SERIES = [
  { key: "style", label: "F1 style ≥ 2", color: "var(--amber)" },
  { key: "correct", label: "Correct", color: "var(--series-val)" },
  { key: "joint", label: "Joint success", color: "var(--ink)" },
] as const;

/** Dev-set results for adapters trained on 125, 250 and 500 examples. */
export function LearningCurve({ points }: { points: Point[] }) {
  const x = linear([0, points.length - 1], [M.l + 30, W - M.r]);
  const y = linear([0, 48], [H - M.b, M.t]);
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={c.svg}
      role="img"
      aria-label="Learning curve: style, correctness and joint success by training-set size"
    >
      {[0, 12, 24, 36, 48].map((t) => (
        <g key={t}>
          <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} className={c.grid} />
          <text x={M.l - 8} y={y(t) + 3.5} textAnchor="end" className={c.tick}>
            {t}
          </text>
        </g>
      ))}
      {points.map((p, i) => (
        <text key={p.examples} x={x(i)} y={H - M.b + 20} textAnchor="middle" className={c.tickStrong}>
          {p.examples} examples
        </text>
      ))}
      {SERIES.map((s) => {
        const pts = points.map((p, i) => [x(i), y(p[s.key])] as const);
        const last = pts[pts.length - 1];
        return (
          <g key={s.key}>
            <path
              d={pts.map(([px, py], i) => `${i ? "L" : "M"}${px},${py}`).join("")}
              fill="none"
              stroke={s.color}
              strokeWidth={2}
            />
            {points.map((p, i) => (
              <g key={p.examples}>
                <circle cx={x(i)} cy={y(p[s.key])} r={4} fill={s.color} />
                <text x={x(i)} y={y(p[s.key]) - 9} textAnchor="middle" className={c.tick}>
                  {p[s.key]}
                </text>
              </g>
            ))}
            <text x={last[0] + 14} y={last[1] + 4} className={c.labelMuted} fill={s.color}>
              {s.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
