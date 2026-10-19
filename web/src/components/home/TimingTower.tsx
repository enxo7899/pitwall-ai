import { ARM_META, type Arm, METRICS, type Metric, site } from "@/lib/data";
import { pct, signed } from "@/lib/format";
import styles from "./TimingTower.module.css";

/** The test-set results as a timing tower: ordered by joint success, best value per column in purple. */
export function TimingTower() {
  const order = (["A", "B", "C"] as Arm[]).sort((a, b) => site.arms[b].joint.s - site.arms[a].joint.s);
  const best = Object.fromEntries(
    METRICS.map(({ key }) => [key, Math.max(...order.map((arm) => site.arms[arm][key].s))]),
  ) as Record<Metric, number>;
  const gap = site.comparisons.C_vs_B.joint_success;

  return (
    <figure className={`panel ${styles.tower}`} aria-label="Test set results by arm">
      <div className="panel-head">
        <span className="label">Test set · 120 questions</span>
        <span className={styles.legend}>
          <span className={styles.bestDot} aria-hidden="true" /> best in column
        </span>
      </div>
      <div className={styles.scroll}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col" className={styles.pos}>
                Pos
              </th>
              <th scope="col">Arm</th>
              {METRICS.map((m) => (
                <th scope="col" key={m.key} className={styles.num}>
                  {m.short}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {order.map((arm, i) => (
              <tr key={arm} data-arm={arm}>
                <td className={styles.pos}>
                  <span className="mono">P{i + 1}</span>
                </td>
                <th scope="row">
                  <div className={styles.arm}>
                    <span className={styles.armBar} style={{ background: ARM_META[arm].color }} aria-hidden="true" />
                    <span>
                      <span className={styles.armName}>{ARM_META[arm].name}</span>
                      <span className={styles.armSetup}>{ARM_META[arm].setup}</span>
                    </span>
                  </div>
                </th>
                {METRICS.map(({ key }) => {
                  const r = site.arms[arm][key];
                  return (
                    <td key={key} className={styles.num} data-best={r.s === best[key] && r.s > 0}>
                      <span className={styles.value}>{pct(r.rate)}</span>
                      <span className={styles.count}>
                        {r.s}/{r.n}
                      </span>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <figcaption className={styles.caption}>
        <span>
          <strong>Joint</strong> = in F1 style <em>and</em> correct <em>and</em> helpful, judged per answer.
        </span>
        <span className="mono tabular">
          Gap P1–P2 {signed(gap.diff)} pts · 95% CI {signed(gap.low)} to {signed(gap.high)} · not a reliable difference
        </span>
      </figcaption>
    </figure>
  );
}
