"use client";

import { useState } from "react";
import { ARM_META, ARMS, CATEGORY_LABEL, type SiteData } from "@/lib/data";
import c from "./chart.module.css";
import styles from "./CategoryGrid.module.css";

type View = "correct" | "joint";

/** Results per category and arm (out of 15 questions each); shading follows the count. */
export function CategoryGrid({ categories }: { categories: SiteData["categories"] }) {
  const [view, setView] = useState<View>("correct");
  const names = Object.keys(categories.A);
  return (
    <div className={styles.wrap}>
      <div className={c.toggle} role="group" aria-label="Measure">
        {(["correct", "joint"] as View[]).map((v) => (
          <button key={v} type="button" aria-pressed={view === v} onClick={() => setView(v)}>
            {v === "correct" ? "Correct" : "Joint success"}
          </button>
        ))}
      </div>
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">Category</th>
            {ARMS.map((arm) => (
              <th scope="col" key={arm}>
                <span className="swatch" style={{ background: ARM_META[arm].color }} /> {ARM_META[arm].short}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {names.map((name) => (
            <tr key={name}>
              <th scope="row">{CATEGORY_LABEL[name] ?? name}</th>
              {ARMS.map((arm) => {
                const cell = categories[arm][name];
                const share = cell[view] / cell.n;
                return (
                  <td key={arm}>
                    <span
                      className={styles.cell}
                      style={{
                        background: `color-mix(in srgb, ${ARM_META[arm].color} ${Math.round(share * 85)}%, var(--bg-raised))`,
                      }}
                      data-dark={share > 0.55}
                    >
                      {cell[view]}
                      <small>/{cell.n}</small>
                    </span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
