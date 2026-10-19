import Link from "next/link";
import { LINKS } from "@/lib/data";
import styles from "./Footer.module.css";
import { Wordmark } from "./Header";

export function Footer() {
  return (
    <footer className={styles.footer}>
      <div className="chequer" aria-hidden="true" />
      <div className={`page ${styles.grid}`}>
        <div className={styles.about}>
          <Wordmark />
          <p>
            A fine-tuning study: Qwen3-1.7B with a LoRA adapter that answers any question in Formula 1 language,
            measured against the base model and a tuned system prompt on a test set locked before training.
          </p>
        </div>
        <div>
          <p className="label">Project</p>
          <ul>
            <li>
              <Link href="/#live">Live demo</Link>
            </li>
            <li>
              <Link href="/report">Interactive report</Link>
            </li>
            <li>
              <Link href="/explorer">Results explorer</Link>
            </li>
          </ul>
        </div>
        <div>
          <p className="label">Source</p>
          <ul>
            <li>
              <a href={LINKS.github}>Code, data and evaluation ↗</a>
            </li>
            <li>
              <a href={LINKS.model}>Adapter on Hugging Face ↗</a>
            </li>
            <li>
              <a href={LINKS.space}>Inference engine ↗</a>
            </li>
          </ul>
        </div>
      </div>
      <div className={`page ${styles.legal}`}>
        <span>Apache-2.0 · 2026</span>
        <span>
          An independent project. Not affiliated with Formula 1, the FIA, or any team or driver. Model answers can be
          wrong: do not rely on them for medical, legal, financial or safety decisions.
        </span>
      </div>
    </footer>
  );
}
