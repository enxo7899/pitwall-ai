import Link from "next/link";
import { site } from "@/lib/data";
import styles from "./BuildLaps.module.css";

/** The project as six laps, each linking to its section of the report. */
export function BuildLaps() {
  const { data, eval_set, training, judge } = site;
  const evalTotal = Object.values(eval_set.splits).reduce((a, b) => a + b, 0);
  const laps = [
    {
      href: "/report#exam",
      title: "Lock the exam",
      stat: `${evalTotal} questions`,
      body: `${eval_set.handwritten} written by hand, every answer key verified, fingerprinted with SHA-256 before any training data existed.`,
    },
    {
      href: "/report#data",
      title: "Build the data",
      stat: `${data.requested} → ${data.selected}`,
      body: "An open-weight teacher writes a plain answer, then an F1 rewrite; a second model family checks nothing was lost.",
    },
    {
      href: "/report#training",
      title: "Train the adapter",
      stat: `${training.minutes.toFixed(1)} min`,
      body: `LoRA on every linear layer: ${(training.trainable_params / 1e6).toFixed(1)}M trainable parameters, 1% of the model, one A100.`,
    },
    {
      href: "/report#judging",
      title: "Judge every answer",
      stat: `${judge.canaries_caught}/${judge.canaries_total} canaries`,
      body: "The judge fills a structured form; fixed Python rules decide correctness. Validated on planted mistakes.",
    },
    {
      href: "/report#results",
      title: "Read the result",
      stat: "3 arms",
      body: "Paired bootstrap intervals and exact McNemar tests, with the wording of every conclusion fixed in advance.",
    },
    {
      href: "/report#reproduce",
      title: "Audit and ship",
      stat: "1 command",
      body: "An audit script recomputes every claim; the analysis rebuilds byte for byte. The engine serves this page.",
    },
  ];
  return (
    <ol className={styles.laps}>
      {laps.map((lap, i) => (
        <li key={lap.title}>
          <Link href={lap.href} className={styles.lap}>
            <span className={styles.top}>
              <span className="label">Lap {String(i + 1).padStart(2, "0")}</span>
              <span className={`${styles.stat} mono tabular`}>{lap.stat}</span>
            </span>
            <span className={styles.title}>{lap.title}</span>
            <span className={styles.body}>{lap.body}</span>
            <span className={styles.more} aria-hidden="true">
              Read →
            </span>
          </Link>
        </li>
      ))}
    </ol>
  );
}
