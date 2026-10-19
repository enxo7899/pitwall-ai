"use client";

import { useState } from "react";
import styles from "./CodeBlock.module.css";

/** A shell snippet with a copy button; falls back to selecting the text if the clipboard is refused. */
export function CodeBlock({ code, label }: { code: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  async function copy(e: React.MouseEvent<HTMLButtonElement>) {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      const pre = e.currentTarget.parentElement?.querySelector("pre");
      if (pre) window.getSelection()?.selectAllChildren(pre);
    }
  }
  return (
    <div className={styles.block}>
      <div className={styles.bar}>
        <span className="label">{label ?? "shell"}</span>
        <button type="button" onClick={copy} className={styles.copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  );
}
