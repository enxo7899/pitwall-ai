// The live engine: the evaluated model running in a Hugging Face ZeroGPU Space.
// The page talks to it through this site's own /api/answer route, which holds the access token on the
// server, applies a rate limit and streams the answer back. Nothing secret is ever shipped to the page.
import type { Arm } from "./data";

export const ENGINE_SPACE = process.env.NEXT_PUBLIC_ENGINE_SPACE ?? "enxo7899/pitwall-ai";
const NO_ANSWER_MS = 120_000; // give up after two minutes, so the page can never wait forever
const WAKING_HINT_MS = 8_000; // no words yet after 8 seconds: the GPU is starting or queued

export type EngineStage = "checking" | "online" | "asleep" | "starting" | "offline" | "unknown";

export type AnswerStats = {
  arm: Arm;
  label: string;
  done: boolean;
  tokens?: number;
  seconds?: number;
  tokens_per_second?: number | null;
  finished?: boolean;
  max_new_tokens?: number;
};

/** Where the Space is in its life cycle, from the Hub's public API. Never throws. */
export async function engineStage(): Promise<EngineStage> {
  try {
    const res = await fetch(`https://huggingface.co/api/spaces/${ENGINE_SPACE}/runtime`, { cache: "no-store" });
    if (!res.ok) return "unknown";
    const { stage } = (await res.json()) as { stage?: string };
    if (stage === "RUNNING") return "online";
    if (stage === "SLEEPING" || stage === "PAUSED") return "asleep";
    if (stage && /BUILD|START/.test(stage)) return "starting";
    return stage ? "offline" : "unknown";
  } catch {
    return "unknown";
  }
}

/** Streams one arm's answer as it is written: yields the text so far and the latest telemetry. */
export async function* streamAnswer(
  question: string,
  arm: Arm,
  onWaking: () => void = () => {},
): AsyncGenerator<{ text: string; stats: AnswerStats }> {
  const abort = new AbortController();
  const giveUp = window.setTimeout(() => abort.abort(), NO_ANSWER_MS);
  let gotWords = false;
  const hint = window.setTimeout(() => !gotWords && onWaking(), WAKING_HINT_MS);
  try {
    const res = await fetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, arm }),
      signal: abort.signal,
    });
    if (!res.ok || !res.body) {
      const { error } = (await res.json().catch(() => ({}))) as { error?: string };
      throw new Error(error ?? "The engine did not answer. Please try again in a minute.");
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline: number;
      while ((newline = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (!line) continue;
        const message = JSON.parse(line) as { text?: string; stats?: AnswerStats; error?: string };
        if (message.error) throw new Error(message.error);
        if (message.text !== undefined && message.stats) {
          gotWords = gotWords || message.text.length > 0;
          yield { text: message.text, stats: message.stats };
        }
      }
    }
    if (!gotWords) throw new Error("The engine did not answer. Please try again in a minute.");
  } catch (error) {
    if (abort.signal.aborted)
      throw new Error("No answer within two minutes. The engine may be busy; please try again.");
    throw error;
  } finally {
    window.clearTimeout(giveUp);
    window.clearTimeout(hint);
  }
}
