import styles from "./StartLights.module.css";

/** Five start lights coming on in turn: the waiting state before the first word arrives. */
export function StartLights({ label }: { label: string }) {
  return (
    <div className={styles.wrap} role="status">
      <span className={styles.lights} aria-hidden="true">
        <span />
        <span />
        <span />
        <span />
        <span />
      </span>
      <span className="label">{label}</span>
    </div>
  );
}
