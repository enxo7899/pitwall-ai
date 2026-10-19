import Link from "next/link";
import { LINKS } from "@/lib/data";
import { EngineStatus } from "./EngineStatus";
import styles from "./Header.module.css";

export function Wordmark() {
  return (
    <span className={styles.wordmark}>
      <svg viewBox="0 0 26 16" aria-hidden="true" className={styles.mark}>
        <path d="M1 15 L6 1 H25 L20 15 Z" fill="var(--amber)" />
        <path d="M9.6 15 L14.6 1" stroke="var(--bg)" strokeWidth="2.6" />
      </svg>
      PITWALL<span className={styles.ai}>AI</span>
    </span>
  );
}

export function Header() {
  return (
    <header className={styles.header}>
      <div className="kerb" aria-hidden="true" />
      <div className={`page ${styles.inner}`}>
        <Link href="/" className={styles.home} aria-label="PITWALL AI home">
          <Wordmark />
        </Link>
        <nav className={styles.nav} aria-label="Main">
          <Link href="/#live">Live</Link>
          <Link href="/report">Report</Link>
          <Link href="/explorer">Explorer</Link>
          <a href={LINKS.github} className={styles.external}>
            Code ↗
          </a>
          <a href={LINKS.model} className={styles.external}>
            Model ↗
          </a>
        </nav>
        <EngineStatus />
      </div>
    </header>
  );
}
