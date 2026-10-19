import styles from "./Circuit.module.css";

// An original, made-up circuit: a long main straight, a right-hand sweep, esses across the top and
// a hairpin back onto the straight. The points are smoothed into one closed curve (Catmull-Rom).
const POINTS: [number, number][] = [
  [110, 285],
  [420, 285],
  [500, 270],
  [530, 225],
  [515, 180],
  [460, 165],
  [430, 130],
  [450, 85],
  [520, 62],
  [545, 40],
  [420, 30],
  [300, 52],
  [240, 98],
  [282, 140],
  [232, 176],
  [140, 160],
  [72, 190],
  [60, 246],
];

function closedCurve(points: [number, number][]): string {
  const n = points.length;
  const p = (i: number) => points[(i + n) % n];
  let d = `M${p(0)[0]},${p(0)[1]}`;
  for (let i = 0; i < n; i++) {
    const [p0, p1, p2, p3] = [p(i - 1), p(i), p(i + 1), p(i + 2)];
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += ` C${c1[0].toFixed(1)},${c1[1].toFixed(1)} ${c2[0].toFixed(1)},${c2[1].toFixed(1)} ${p2[0]},${p2[1]}`;
  }
  return `${d} Z`;
}

const TRACK = closedCurve(POINTS);

/** Decorative track map with timing-sector colours and a car lapping it (hidden for reduced motion). */
export function Circuit({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 600 320" className={`${styles.circuit} ${className ?? ""}`} aria-hidden="true">
      <path id="pitwall-track" d={TRACK} className={styles.asphalt} />
      <path d={TRACK} pathLength={300} className={styles.sector} stroke="var(--best)" strokeDasharray="100 200" />
      <path d={TRACK} pathLength={300} className={styles.sector} stroke="var(--pass)" strokeDasharray="0 100 100 100" />
      <path d={TRACK} pathLength={300} className={styles.sector} stroke="var(--warn)" strokeDasharray="0 200 100 0" />
      <g transform="translate(120 285)">
        <rect x={-2} y={-9} width={4} height={18} className={styles.finish} />
      </g>
      <text x={150} y={310} className={styles.tag}>
        S1
      </text>
      <text x={548} y={150} className={styles.tag}>
        S2
      </text>
      <text x={250} y={20} className={styles.tag}>
        S3
      </text>
      <circle r={5} className={styles.car}>
        <animateMotion dur="16s" repeatCount="indefinite" rotate="auto">
          <mpath href="#pitwall-track" />
        </animateMotion>
      </circle>
    </svg>
  );
}
