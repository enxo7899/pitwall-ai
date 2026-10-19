// Server-side bridge to the inference engine (a Hugging Face ZeroGPU Space).
//
// Why it exists: the engine must not be callable by anyone directly. This route holds two secrets the
// browser never sees (a read-only Hugging Face token, so answers use the project's GPU allowance, and the
// engine's access key), and it applies the abuse limits below before any GPU time is spent.
// Answers stream back as newline-delimited JSON: {"text", "stats"} while the model writes, {"error"} on failure.
import { Client } from "@gradio/client";

export const runtime = "nodejs";
export const maxDuration = 60; // one answer is capped at 512 tokens, well under a minute

const { HF_TOKEN, PITWALL_KEY, PITWALL_SPACE } = process.env; // set in the Vercel project settings, see web/.env.example
const SPACE = PITWALL_SPACE ?? "enxo7899/pitwall-ai";
const MAX_BODY_BYTES = 4_096;
const LIMITS = [
  { windowMs: 60_000, max: 6, message: "Please slow down: at most 6 answers a minute." },
  {
    windowMs: 3_600_000,
    max: 40,
    message: "You have reached this hour's limit of 40 answers. Please come back later.",
  },
];
const MAX_IN_FLIGHT_PER_VISITOR = 2;
const DAILY_BUDGET = 400; // answers per server instance per day; the GPU allowance itself is the final cap

// In-memory counters, per server instance. They stop bursts and scripts; the Vercel firewall rule in
// front of this route enforces the same idea across instances.
const history = new Map<string, number[]>();
const inFlight = new Map<string, number>();
let day = new Date().toDateString();
let answeredToday = 0;

type Rejection = { status: number; error: string };

function checkLimits(visitor: string): Rejection | null {
  const now = Date.now();
  if (new Date().toDateString() !== day) {
    day = new Date().toDateString();
    answeredToday = 0;
  }
  if (answeredToday >= DAILY_BUDGET) {
    return { status: 503, error: "The live demo has reached today's limit. The recorded answers below still work." };
  }
  if ((inFlight.get(visitor) ?? 0) >= MAX_IN_FLIGHT_PER_VISITOR) {
    return { status: 429, error: "Please wait for the current answer to finish." };
  }
  const longest = Math.max(...LIMITS.map((l) => l.windowMs));
  const hits = (history.get(visitor) ?? []).filter((t) => now - t < longest);
  for (const limit of LIMITS) {
    if (hits.filter((t) => now - t < limit.windowMs).length >= limit.max) {
      history.set(visitor, hits);
      return { status: 429, error: limit.message };
    }
  }
  hits.push(now);
  history.set(visitor, hits);
  if (history.size > 10_000) history.clear(); // bound memory if many different visitors arrive
  return null;
}

/** Only this site's own pages may call the route; other websites (and scripts without an Origin) are refused. */
function fromThisSite(request: Request): boolean {
  const origin = request.headers.get("origin");
  const host = request.headers.get("x-forwarded-host") ?? request.headers.get("host");
  if (!origin || !host) return false;
  try {
    return new URL(origin).host === host;
  } catch {
    return false;
  }
}

/** Turns engine errors into a sentence a visitor can act on; the details go to the server log only. */
function explain(message: string): string {
  if (/quota|runs limit/i.test(message)) {
    return "Today's GPU allowance for the live demo is used up. The recorded answers below still work.";
  }
  return "The engine did not answer. Please try again in a minute.";
}

// One connection per server instance, reused across requests; dropped and rebuilt if it fails.
let connection: Promise<Client> | null = null;
function engine(): Promise<Client> {
  connection ??= Client.connect(SPACE, {
    token: HF_TOKEN as `hf_${string}` | undefined,
    headers: PITWALL_KEY ? { "x-pitwall-key": PITWALL_KEY } : undefined,
  }).catch((error) => {
    connection = null;
    throw error;
  });
  return connection;
}

const reject = ({ status, error }: Rejection) => Response.json({ error }, { status });

export async function POST(request: Request) {
  if (!fromThisSite(request)) {
    console.warn("answer route: refused a request from another origin");
    return reject({ status: 403, error: "This endpoint only serves the PITWALL AI website." });
  }
  if (Number(request.headers.get("content-length") ?? 0) > MAX_BODY_BYTES) {
    return reject({ status: 413, error: "The request is too large." });
  }
  let body: { question?: unknown; arm?: unknown };
  try {
    body = await request.json();
  } catch {
    return reject({ status: 400, error: "The request could not be read." });
  }
  const question = typeof body.question === "string" ? body.question.trim() : "";
  const arm = body.arm;
  if (!question || question.length > 1000 || (arm !== "A" && arm !== "B" && arm !== "C")) {
    return reject({ status: 400, error: "Ask a question of up to 1,000 characters." });
  }
  const visitor = (request.headers.get("x-forwarded-for") ?? "local").split(",")[0].trim();
  const refused = checkLimits(visitor);
  if (refused) {
    console.warn(`answer route: rate limit (${refused.status})`);
    return reject(refused);
  }

  inFlight.set(visitor, (inFlight.get(visitor) ?? 0) + 1);
  answeredToday += 1;
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (message: object) => controller.enqueue(encoder.encode(JSON.stringify(message) + "\n"));
      let sentWords = false;
      try {
        for (let attempt = 1; attempt <= 2 && !sentWords; attempt++) {
          try {
            const job = (await engine()).submit("/answer", { question, arm, seed: 0 });
            let failure = "";
            for await (const message of job) {
              if (message.type === "data") {
                const [text, stats] = message.data as [string, { done?: boolean }];
                send({ text, stats });
                sentWords = sentWords || text.length > 0;
                if (stats?.done) break;
              } else if (message.type === "status" && message.stage === "error") {
                failure = String(message.message ?? "engine error");
                break;
              }
            }
            if (failure) throw new Error(failure);
            if (!sentWords) throw new Error("the engine stream ended without an answer");
          } catch (error) {
            const detail = error instanceof Error ? error.message : String(error);
            console.error(`answer route: attempt ${attempt} failed: ${detail}`);
            connection = null; // reconnect on the next attempt
            const lastTry = attempt === 2 || /quota|runs limit|only serves/i.test(detail);
            if (lastTry && !sentWords) {
              send({ error: explain(detail) });
              break;
            }
            if (!lastTry) await new Promise((r) => setTimeout(r, 1_500));
          }
        }
      } finally {
        inFlight.set(visitor, Math.max(0, (inFlight.get(visitor) ?? 1) - 1));
        controller.close();
      }
    },
  });
  return new Response(stream, {
    headers: { "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-store" },
  });
}
