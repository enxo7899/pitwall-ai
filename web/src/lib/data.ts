// Typed access to the results exported by scripts/tools/export_web_data.py.
// Nothing on the site is typed in by hand: every number comes from this file.
import raw from "@/data/site.json";

export type Arm = "A" | "B" | "C";
export const ARMS: Arm[] = ["A", "B", "C"];

export const ARM_META: Record<Arm, { name: string; short: string; setup: string; color: string }> = {
  A: { name: "Base model", short: "Base", setup: "Qwen3-1.7B, no prompt", color: "var(--arm-a)" },
  B: {
    name: "Base + F1 prompt",
    short: "Prompted",
    setup: "451-token F1 prompt, tuned then frozen",
    color: "var(--arm-b)",
  },
  C: { name: "PITWALL fine-tuned", short: "Fine-tuned", setup: "LoRA adapter, no prompt", color: "var(--arm-c)" },
};

export type Rate = { s: number; n: number; rate: number; low: number; high: number };
export type Metric = "joint" | "style" | "correct" | "helpful";
export const METRICS: { key: Metric; label: string; short: string }[] = [
  { key: "joint", label: "Joint success", short: "Joint" },
  { key: "style", label: "F1 style ≥ 2", short: "Style" },
  { key: "correct", label: "Correct", short: "Correct" },
  { key: "helpful", label: "Helpful ≥ 2", short: "Helpful" },
];

export type Comparison = {
  diff: number;
  low: number;
  high: number;
  p: number;
  only_first: number;
  only_second: number;
  verdict: string;
};
export type PairKey = "C_vs_B" | "C_vs_A" | "B_vs_A";
export type ComparisonMetric = "joint_success" | "style" | "correct" | "helpful" | "joint_success_strict_rule";

export type KeyPointStatus = "present" | "missing" | "contradicted";
export type ArmVerdict = {
  answer: string;
  tokens: number;
  finished: boolean;
  style: number | null;
  helpful: number | null;
  correct: boolean | null;
  joint: boolean;
  key_points: KeyPointStatus[];
  style_evidence: string;
  helpful_evidence: string;
  extracted: string;
  error_quote: string;
  error_serious: boolean;
};
export type ExplorerItem = {
  id: string;
  category: string;
  question: string;
  handwritten: boolean;
  kind: string;
  answer_key: string;
  key_points: { text: string; critical: boolean }[];
  arms: Record<Arm, ArmVerdict>;
};

export type TrainStep = { step: number; epoch: number; loss: number; lr: number; grad_norm: number; accuracy: number };
export type ValPoint = { step: number; epoch: number; loss: number; accuracy: number };
export type EpochRow = {
  epoch: number;
  checkpoint: string;
  n: number;
  jsr: number;
  style: number;
  correct: number;
  helpful: number;
  validation_loss: number;
};

export type SiteData = {
  model: { id: string; revision: string; decoding: Record<string, number | boolean> };
  arms: Record<Arm, Record<Metric, Rate> & { mean_style: number; joint_strict: Rate }>;
  comparisons: Record<PairKey, Record<ComparisonMetric, Comparison>>;
  categories: Record<Arm, Record<string, { correct: number; joint: number; n: number }>>;
  judge: {
    canaries: Record<string, string>;
    canaries_caught: number;
    canaries_total: number;
    retest: Record<string, number>;
    human: {
      n: number;
      kappa_style: number;
      kappa_correct: number;
      kappa_helpful: number;
      disputed: number;
      sided_with_judge: number;
    };
  };
  safety: {
    tone_acceptable: Record<Arm, [number, number]>;
    judge_correct_and_safe: Record<Arm, [number, number]>;
  };
  latency: Record<
    Arm,
    { label: string; median_prompt_tokens: number; median_first_token_ms: number; median_decode_tokens_per_s: number }
  > & { device: string; new_tokens: number };
  data: {
    requested: number;
    near_duplicates: number;
    eval_lookalikes: number;
    answered: number;
    accepted: number;
    selected: number;
    heldout: number;
    by_category: Record<string, number>;
    reject_reasons: Record<string, number>;
    max_opener_share: number;
    distinct_f1_terms: number;
    teacher: string;
    checker: string;
  };
  eval_set: { splits: Record<string, number>; test_by_category: Record<string, number>; handwritten: number };
  training: {
    steps: TrainStep[];
    validation: ValPoint[];
    steps_per_epoch: number;
    total_steps: number;
    minutes: number;
    peak_memory_gb: number;
    gpu: string;
    trainable_params: number;
    total_params: number;
    epochs: EpochRow[];
    chosen_epoch: number;
    learning_curve: { examples: number; style: number; joint: number; correct: number }[];
  };
  showcase: ExplorerItem[];
  timeline: { when: string; commit: string; event: string }[];
};

export const site = raw as unknown as SiteData;

export const CATEGORY_LABEL: Record<string, string> = {
  procedure: "Procedures",
  quantitative: "Calculations",
  factual: "Facts",
  explanation: "Explanations",
  advice: "Advice",
  writing: "Writing",
  technical: "Technical",
  uncertainty: "False premises",
};

export const LINKS = {
  github: "https://github.com/enxo7899/pitwall-ai",
  model: "https://huggingface.co/enxo7899/pitwall-qwen3-1.7b-lora",
  space: "https://huggingface.co/spaces/enxo7899/pitwall-ai",
  report: "https://github.com/enxo7899/pitwall-ai/blob/main/docs/REPORT.pdf",
};
