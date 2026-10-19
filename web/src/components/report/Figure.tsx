import styles from "./Figure.module.css";

type Props = {
  n: number;
  title: string;
  caption?: React.ReactNode;
  controls?: React.ReactNode;
  children: React.ReactNode;
};

/** A numbered figure frame: title bar, the chart, and a caption saying what it shows. */
export function Figure({ n, title, caption, controls, children }: Props) {
  return (
    <figure className={styles.figure}>
      <div className={styles.head}>
        <span className={styles.title}>
          <span className="label">Fig. {n}</span>
          {title}
        </span>
        {controls}
      </div>
      <div className={styles.body}>{children}</div>
      {caption && <figcaption className={styles.caption}>{caption}</figcaption>}
    </figure>
  );
}
