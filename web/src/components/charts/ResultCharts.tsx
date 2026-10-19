import { ARM_META, ARMS, type ComparisonMetric, METRICS, type PairKey, type SiteData } from "@/lib/data";
import { signed } from "@/lib/format";
import { linear } from "@/lib/scale";
import c from "./chart.module.css";
import styles from "./ResultCharts.module.css";

/** Each arm's rate on each metric, with 95% Wilson intervals: four small panels sharing one scale. */
export function RateBars({ arms }: { arms: SiteData["arms"] }) {
  const W = 300;
  const H = 128;
  const x = linear([0, 1], [64, W - 46]);
  return (
    <div className={styles.multiples}>
      {METRICS.map((m) => (
        <div key={m.key} className={styles.panel}>
          <p className="label">{m.label}</p>
          <svg viewBox={`0 0 ${W} ${H}`} className={c.svg} role="img" aria-label={`${m.label} by arm`}>
            {[0, 0.5, 1].map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={4} y2={H - 18} className={c.grid} />
                <text x={x(t)} y={H - 4} textAnchor="middle" className={c.tick}>
                  {t * 100}%
                </text>
              </g>
            ))}
            {ARMS.map((arm, i) => {
              const r = arms[arm][m.key];
              const y = 10 + i * 34;
              return (
                <g key={arm}>
                  <text x={0} y={y + 12} className={c.tickStrong}>
                    {ARM_META[arm].short}
                  </text>
                  <rect
                    x={x(0)}
                    y={y + 2}
                    width={x(r.rate) - x(0)}
                    height={14}
                    rx={1.5}
                    fill={ARM_META[arm].color}
                    opacity={0.9}
                  />
                  <line x1={x(r.low)} x2={x(r.high)} y1={y + 9} y2={y + 9} stroke="var(--ink)" strokeWidth={1.25} />
                  <line x1={x(r.low)} x2={x(r.low)} y1={y + 5} y2={y + 13} stroke="var(--ink)" strokeWidth={1.25} />
                  <line x1={x(r.high)} x2={x(r.high)} y1={y + 5} y2={y + 13} stroke="var(--ink)" strokeWidth={1.25} />
                  <text x={Math.max(x(r.high), x(r.rate)) + 6} y={y + 13} className={c.value}>
                    {(r.rate * 100).toFixed(1)}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      ))}
    </div>
  );
}

const PAIRS: { key: PairKey; label: string }[] = [
  { key: "C_vs_B", label: "C − B" },
  { key: "C_vs_A", label: "C − A" },
  { key: "B_vs_A", label: "B − A" },
];
const DIFF_METRICS: { key: ComparisonMetric; label: string }[] = [
  { key: "joint_success", label: "Joint success" },
  { key: "style", label: "F1 style" },
  { key: "correct", label: "Correct" },
  { key: "helpful", label: "Helpful" },
];

/** Paired differences in points with 95% bootstrap intervals. Green/red: interval excludes zero. */
export function ForestPlot({ comparisons }: { comparisons: SiteData["comparisons"] }) {
  const W = 300;
  const H = 128;
  return (
    <div className={styles.multiples}>
      {DIFF_METRICS.map((m) => {
        const rows = PAIRS.map((p) => ({ ...p, d: comparisons[p.key][m.key] }));
        const lo = Math.min(0, ...rows.map((r) => r.d.low)) - 6;
        const hi = Math.max(0, ...rows.map((r) => r.d.high)) + 6;
        const x = linear([lo, hi], [52, W - 10]);
        return (
          <div key={m.key} className={styles.panel}>
            <p className="label">{m.label}</p>
            <svg viewBox={`0 0 ${W} ${H}`} className={c.svg} role="img" aria-label={`Paired differences in ${m.label}`}>
              <line x1={x(0)} x2={x(0)} y1={2} y2={H - 18} stroke="var(--ink-2)" strokeWidth={1} />
              <text x={x(0)} y={H - 4} textAnchor="middle" className={c.tick}>
                0
              </text>
              {rows.map((r, i) => {
                const y = 18 + i * 34;
                const color = r.d.low > 0 ? "var(--pass)" : r.d.high < 0 ? "var(--fail)" : "var(--ink-3)";
                return (
                  <g key={r.key}>
                    <text x={0} y={y + 4} className={c.tickStrong}>
                      {r.label}
                    </text>
                    <line
                      x1={x(r.d.low)}
                      x2={x(r.d.high)}
                      y1={y}
                      y2={y}
                      stroke={color}
                      strokeWidth={2.5}
                      strokeLinecap="round"
                    />
                    <circle cx={x(r.d.diff)} cy={y} r={4.5} fill={color} />
                    <text x={x(r.d.diff)} y={y - 9} textAnchor="middle" className={c.value} fill={color}>
                      {signed(r.d.diff)}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        );
      })}
    </div>
  );
}
