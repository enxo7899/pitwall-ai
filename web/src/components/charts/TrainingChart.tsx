"use client";

import { useMemo, useRef, useState } from "react";
import type { TrainStep, ValPoint } from "@/lib/data";
import { linear, ticks } from "@/lib/scale";
import c from "./chart.module.css";

type Series = "loss" | "lr" | "accuracy";
const SERIES: { key: Series; label: string }[] = [
  { key: "loss", label: "Loss" },
  { key: "lr", label: "Learning rate" },
  { key: "accuracy", label: "Token accuracy" },
];
const W = 760;
const H = 330;
const M = { l: 52, r: 18, t: 26, b: 34 };
const WINDOW = 15; // the bold line is a 15-step centred mean, as in the README figure

function centredMean(values: number[], window: number): number[] {
  const half = Math.floor(window / 2);
  return values.map((_, i) => {
    const part = values.slice(Math.max(0, i - half), i + half + 1);
    return part.reduce((a, b) => a + b, 0) / part.length;
  });
}

const path = (points: [number, number][]) =>
  points.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");

type Props = { steps: TrainStep[]; validation: ValPoint[]; stepsPerEpoch: number; chosenEpoch: number };

/** Training curves over all steps, with epoch boundaries, the chosen checkpoint and a hover readout. */
export function TrainingChart({ steps, validation, stepsPerEpoch, chosenEpoch }: Props) {
  const [series, setSeries] = useState<Series>("loss");
  const [hover, setHover] = useState<number | null>(null);
  const svg = useRef<SVGSVGElement>(null);

  const total = steps[steps.length - 1].step;
  const chosenStep = chosenEpoch * stepsPerEpoch;
  const { raw, smooth } = useMemo(() => {
    const values = steps.map((s) => (series === "lr" ? s.lr * 1e4 : series === "accuracy" ? s.accuracy : s.loss));
    return { raw: values, smooth: centredMean(values, WINDOW) };
  }, [steps, series]);
  const val = series === "lr" ? [] : validation.map((v) => ({ ...v, y: series === "accuracy" ? v.accuracy : v.loss }));
  const domain: [number, number] = series === "loss" ? [0.4, 3.6] : series === "lr" ? [0, 2.2] : [0.45, 0.9];

  const x = linear([0, total], [M.l, W - M.r]);
  const y = linear(domain, [H - M.b, M.t]);
  const yTicks = ticks(domain[0], domain[1], 6);
  const epochs = Array.from({ length: Math.round(total / stepsPerEpoch) }, (_, i) => (i + 1) * stepsPerEpoch);
  const bestVal = val.length ? val.reduce((a, b) => ((series === "loss" ? b.y < a.y : b.y > a.y) ? b : a)) : null;

  const at = hover === null ? null : steps[hover];
  const valAt = at ? [...val].reverse().find((v) => v.step <= at.step) : undefined;
  const unit = series === "lr" ? " ×10⁻⁴" : "";
  const fmt = (v: number) => (series === "accuracy" ? `${(v * 100).toFixed(1)}%` : v.toFixed(series === "lr" ? 2 : 3));

  function onMove(e: React.PointerEvent<SVGSVGElement>) {
    const box = svg.current?.getBoundingClientRect();
    if (!box) return;
    const step = ((e.clientX - box.left) / box.width) * W;
    const value = ((step - M.l) / (W - M.l - M.r)) * total;
    setHover(Math.max(0, Math.min(steps.length - 1, Math.round(value) - 1)));
  }

  return (
    <div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 12, justifyContent: "space-between", marginBottom: 14 }}>
        <div className={c.toggle} role="group" aria-label="Series">
          {SERIES.map((s) => (
            <button key={s.key} type="button" aria-pressed={series === s.key} onClick={() => setSeries(s.key)}>
              {s.label}
            </button>
          ))}
        </div>
        <div className={c.legend}>
          <span>
            <i className={c.key} style={{ background: "var(--ink)" }} /> training ({WINDOW}-step mean)
          </span>
          {series !== "lr" && (
            <span>
              <i className={c.key} style={{ background: "var(--series-val)" }} /> validation (130 held-out examples)
            </span>
          )}
          <span>
            <i className={c.key} style={{ background: "var(--amber)" }} /> chosen checkpoint
          </span>
        </div>
      </div>
      <p className={c.readout} aria-live="polite">
        {at ? (
          <>
            Step <b>{at.step}</b> · epoch <b>{at.epoch.toFixed(2)}</b> · training <b>{fmt(raw[hover!])}</b>
            {unit} (mean {fmt(smooth[hover!])})
            {valAt && (
              <>
                {" "}
                · validation <b>{fmt(valAt.y)}</b> at step {valAt.step}
              </>
            )}
          </>
        ) : (
          "Hover or drag across the chart to read any step."
        )}
      </p>
      <svg
        ref={svg}
        viewBox={`0 0 ${W} ${H}`}
        className={c.svg}
        role="img"
        aria-label={`Training ${series} over ${total} steps with validation points; checkpoint ${chosenStep} chosen.`}
        onPointerMove={onMove}
        onPointerLeave={() => setHover(null)}
        style={{ touchAction: "pan-y" }}
      >
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={M.l} x2={W - M.r} y1={y(t)} y2={y(t)} className={c.grid} />
            <text x={M.l - 8} y={y(t) + 3.5} textAnchor="end" className={c.tick}>
              {series === "accuracy" ? `${Math.round(t * 100)}%` : t}
            </text>
          </g>
        ))}
        {epochs.map((s, i) => (
          <g key={s}>
            <line x1={x(s)} x2={x(s)} y1={M.t} y2={H - M.b} className={c.axis} strokeDasharray="2 4" />
            <text x={x(s - stepsPerEpoch / 2)} y={M.t - 10} textAnchor="middle" className={c.tick}>
              E{i + 1}
            </text>
          </g>
        ))}
        <line x1={x(chosenStep)} x2={x(chosenStep)} y1={M.t} y2={H - M.b} stroke="var(--amber)" strokeWidth={1.5} />
        <text x={x(chosenStep) - 6} y={H - M.b - 8} textAnchor="end" className={c.tickStrong} fill="var(--amber)">
          checkpoint-{chosenStep}
        </text>
        <path
          d={path(raw.map((v, i) => [x(steps[i].step), y(v)]))}
          fill="none"
          stroke="var(--ink-3)"
          strokeWidth={1}
          opacity={0.45}
        />
        <path
          d={path(smooth.map((v, i) => [x(steps[i].step), y(v)]))}
          fill="none"
          stroke="var(--ink)"
          strokeWidth={2}
        />
        {val.length > 0 && (
          <path d={path(val.map((v) => [x(v.step), y(v.y)]))} fill="none" stroke="var(--series-val)" strokeWidth={2} />
        )}
        {val.map((v) => (
          <circle
            key={v.step}
            cx={x(v.step)}
            cy={y(v.y)}
            r={3.5}
            fill="var(--bg-panel)"
            stroke="var(--series-val)"
            strokeWidth={2}
          />
        ))}
        {bestVal && (
          <text
            x={x(bestVal.step)}
            y={y(bestVal.y) - 12}
            textAnchor="middle"
            className={c.tickStrong}
            fill="var(--series-val)"
          >
            {series === "loss" ? "lowest" : "highest"} {fmt(bestVal.y)}
          </text>
        )}
        {[0, 100, 200, 300, total].map((t) => (
          <text key={t} x={x(t)} y={H - M.b + 18} textAnchor="middle" className={c.tick}>
            {t}
          </text>
        ))}
        <text x={W - M.r} y={H - 2} textAnchor="end" className={c.tick}>
          optimiser step
        </text>
        {at && (
          <g pointerEvents="none">
            <line x1={x(at.step)} x2={x(at.step)} y1={M.t} y2={H - M.b} stroke="var(--ink-2)" strokeWidth={1} />
            <circle cx={x(at.step)} cy={y(smooth[hover!])} r={4} fill="var(--ink)" />
          </g>
        )}
      </svg>
    </div>
  );
}
