import Link from "next/link";
import { BuildLaps } from "@/components/home/BuildLaps";
import { Circuit } from "@/components/home/Circuit";
import { Findings } from "@/components/home/Findings";
import { LiveConsole } from "@/components/home/LiveConsole";
import { TimingTower } from "@/components/home/TimingTower";
import { site } from "@/lib/data";
import styles from "./page.module.css";

export default function Home() {
  return (
    <>
      <section className={styles.hero}>
        <div className={`page ${styles.heroGrid}`}>
          <div className={styles.intro}>
            <p className="label">LoRA fine-tuning study · Qwen3-1.7B</p>
            <h1 className={styles.title}>
              Any question,
              <br />
              answered from
              <br />
              the <span className={styles.accent}>pit wall.</span>
            </h1>
            <p className={styles.lede}>
              I fine-tuned a 1.7B open model to answer everyday questions in the language of Formula 1, then tested
              whether it beats a carefully tuned system prompt on {site.eval_set.splits.test} questions I locked before
              training. Short answer: a far more consistent voice and a faster first token, but no gain in accuracy.
            </p>
            <div className={styles.actions}>
              <Link href="#live" className="btn btn-primary">
                Ask the model
              </Link>
              <Link href="/report" className="btn">
                Read the report
              </Link>
            </div>
          </div>
          <div className={styles.towerWrap}>
            <Circuit className={styles.circuit} />
            <TimingTower />
          </div>
        </div>
      </section>

      <section id="live" className={`page ${styles.section}`}>
        <div className={styles.sectionHead}>
          <h2 className={styles.h2}>Live from the engine</h2>
          <p>
            The fine-tuned model on its own, or all three arms side by side: the base model, the base model with the F1
            system prompt, and the adapter.
          </p>
        </div>
        <LiveConsole showcase={site.showcase} />
      </section>

      <section className={`page ${styles.section}`}>
        <div className={styles.sectionHead}>
          <h2 className={styles.h2}>What the evaluation found</h2>
          <p>Fine-tuned against the tuned prompt, on the same 120 test questions, judged the same way.</p>
        </div>
        <Findings />
      </section>

      <section className={`page ${styles.section}`}>
        <div className={styles.sectionHead}>
          <h2 className={styles.h2}>How it was built</h2>
          <p>
            Six steps, each closed with tests and evidence before the next began. The report walks through all of them.
          </p>
        </div>
        <BuildLaps />
      </section>
    </>
  );
}
