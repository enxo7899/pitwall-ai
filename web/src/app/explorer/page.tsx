import type { Metadata } from "next";
import { Explorer } from "@/components/explorer/Explorer";
import styles from "./explorer.module.css";

export const metadata: Metadata = {
  title: "Results explorer",
  description: "All 120 test questions with every arm's answer and the judge's verdict and evidence.",
};

export default function ExplorerPage() {
  return (
    <div className={`page ${styles.page}`}>
      <header className={styles.header}>
        <p className="label">Results explorer</p>
        <h1>Every answer, every verdict</h1>
        <p>
          All 120 test questions with the answer each arm gave and what the judge wrote about it: the scores, which key
          points were present, and any extra errors. These are the exact records the results are computed from.
        </p>
      </header>
      <Explorer />
    </div>
  );
}
