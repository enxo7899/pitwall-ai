"use client";

import { useEffect, useState } from "react";
import { type EngineStage, engineStage } from "@/lib/engine";
import styles from "./EngineStatus.module.css";

const COPY: Record<EngineStage, { label: string; title: string }> = {
  checking: { label: "Engine", title: "Checking the inference engine…" },
  online: { label: "Engine online", title: "The model is loaded and ready to answer." },
  asleep: {
    label: "Engine asleep",
    title: "The engine sleeps when idle. Your first question wakes it (about a minute).",
  },
  starting: { label: "Engine starting", title: "The engine is starting up and will be ready shortly." },
  offline: { label: "Engine offline", title: "The live engine is unavailable. Recorded answers still work." },
  unknown: { label: "Engine", title: "Engine status unavailable. Recorded answers still work." },
};

/** A small status light: green when the model is loaded, amber while asleep or starting. */
export function EngineStatus() {
  const [stage, setStage] = useState<EngineStage>("checking");

  useEffect(() => {
    let alive = true;
    const check = () => engineStage().then((s) => alive && setStage(s));
    check();
    const timer = window.setInterval(check, 60_000);
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
  }, []);

  return (
    <span className={styles.status} data-stage={stage} title={COPY[stage].title}>
      <span className={styles.dot} aria-hidden="true" />
      <span className={styles.text}>{COPY[stage].label}</span>
    </span>
  );
}
