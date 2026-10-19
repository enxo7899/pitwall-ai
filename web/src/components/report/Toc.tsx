"use client";

import { useEffect, useState } from "react";
import styles from "./Toc.module.css";

export type TocEntry = { id: string; title: string };

/** Sticky contents rail that highlights the section currently being read. */
export function Toc({ entries }: { entries: TocEntry[] }) {
  const [active, setActive] = useState(entries[0]?.id);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (seen) => {
        const visible = seen
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: "-90px 0px -65% 0px" },
    );
    for (const { id } of entries) {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    }
    return () => observer.disconnect();
  }, [entries]);

  return (
    <nav className={styles.toc} aria-label="Report sections">
      <p className="label">Report</p>
      <ol>
        {entries.map((e, i) => (
          <li key={e.id}>
            <a href={`#${e.id}`} aria-current={active === e.id ? "true" : undefined}>
              <span className={styles.no}>{String(i + 1).padStart(2, "0")}</span>
              {e.title}
            </a>
          </li>
        ))}
      </ol>
    </nav>
  );
}
