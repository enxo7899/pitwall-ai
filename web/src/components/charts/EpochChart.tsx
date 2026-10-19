import type { EpochRow } from "@/lib/data";
import { linear } from "@/lib/scale";
import c from "./chart.module.css";

const W = 760;
const H = 300;
const M = { l: 46, r: 52, t: 30, b: 40 };

/** Dev joint success per epoch (bars) against validation loss (line): the two signals disagree. */
export function EpochChart({ epochs, chosen }: { epochs: EpochRow[]; chosen: number }) {
  const band = (W - M.l - M.r) / epochs.length;
  const bx = (i: number) => M.l + i * band + band * 0.22;
  const bw = band * 0.56;
  const yJ = linear([0, 16], [H - M.b, M.t]);
  const yL = linear([1.6, 2.4], [H - M.b, M.t]);
  const centre = (i: number) => M.l + (i + 0.5) * band;
  const line = epochs.map((e, i) => `${i ? "L" : "M"}${centre(i)},${yL(e.validation_loss)}`).join("");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className={c.svg}
      role="img"
      aria-label="Dev joint success by epoch and validation loss by epoch"
    >
      {[0, 4, 8, 12, 16].map((t) => (
        <g key={t}>
          <line x1={M.l} x2={W - M.r} y1={yJ(t)} y2={yJ(t)} className={c.grid} />
          <text x={M.l - 8} y={yJ(t) + 3.5} textAnchor="end" className={c.tick}>
            {t}
          </text>
        </g>
      ))}
      {[1.6, 1.8, 2.0, 2.2, 2.4].map((t) => (
        <text key={t} x={W - M.r + 8} y={yL(t) + 3.5} className={c.tick} fill="var(--series-val)">
          {t.toFixed(1)}
        </text>
      ))}
      <text x={M.l - 8} y={M.t - 14} textAnchor="end" className={c.tick}>
        /48
      </text>
      <text x={W - M.r + 8} y={M.t - 14} className={c.tick} fill="var(--series-val)">
        val loss
      </text>
      {epochs.map((e, i) => {
        const isChosen = e.epoch === chosen;
        return (
          <g key={e.epoch}>
            <rect
              x={bx(i)}
              y={yJ(e.jsr)}
              width={bw}
              height={yJ(0) - yJ(e.jsr)}
              fill={isChosen ? "var(--amber)" : "var(--line-strong)"}
              rx={1.5}
            />
            <text
              x={centre(i)}
              y={yJ(e.jsr) - 7}
              textAnchor="middle"
              className={c.value}
              fill={isChosen ? "var(--amber)" : undefined}
            >
              {e.jsr}
            </text>
            <text x={centre(i)} y={H - M.b + 18} textAnchor="middle" className={isChosen ? c.tickStrong : c.tick}>
              Epoch {e.epoch}
            </text>
            {isChosen && (
              <text x={centre(i)} y={H - M.b + 32} textAnchor="middle" className={c.tick} fill="var(--amber)">
                chosen
              </text>
            )}
          </g>
        );
      })}
      <path d={line} fill="none" stroke="var(--series-val)" strokeWidth={2} />
      {epochs.map((e, i) => (
        <g key={e.epoch}>
          <circle
            cx={centre(i)}
            cy={yL(e.validation_loss)}
            r={4}
            fill="var(--bg-panel)"
            stroke="var(--series-val)"
            strokeWidth={2}
          />
          <text x={centre(i) + 8} y={yL(e.validation_loss) - 8} className={c.tick} fill="var(--series-val)">
            {e.validation_loss.toFixed(2)}
          </text>
        </g>
      ))}
    </svg>
  );
}
