import type { Metadata } from "next";
import Link from "next/link";
import { CategoryGrid } from "@/components/charts/CategoryGrid";
import { Funnel, RejectReasons } from "@/components/charts/DataCharts";
import { EpochChart } from "@/components/charts/EpochChart";
import { LearningCurve } from "@/components/charts/LearningCurve";
import { ForestPlot, RateBars } from "@/components/charts/ResultCharts";
import { CanaryGrid, EvalComposition, LatencyBars, ToneBars } from "@/components/charts/SmallCharts";
import { TrainingChart } from "@/components/charts/TrainingChart";
import { ArmsDiagram, LoraDiagram, Steps } from "@/components/report/Diagrams";
import { Figure } from "@/components/report/Figure";
import { Toc } from "@/components/report/Toc";
import { CodeBlock } from "@/components/ui/CodeBlock";
import { LINKS, site } from "@/lib/data";
import { pct, signed, thousands } from "@/lib/format";
import s from "./report.module.css";

export const metadata: Metadata = {
  title: "Report",
  description:
    "How PITWALL AI was built and evaluated: the exam, the training data, the training run, the judge and the results.",
};

const SECTIONS = [
  { id: "question", title: "The question" },
  { id: "exam", title: "The exam" },
  { id: "data", title: "Training data" },
  { id: "training", title: "Training" },
  { id: "judging", title: "Judging" },
  { id: "results", title: "Results" },
  { id: "cost", title: "Fine-tune or prompt?" },
  { id: "safety", title: "Serious situations" },
  { id: "limits", title: "Limits and next steps" },
  { id: "reproduce", title: "Reproduce it" },
];

function Section({ id, n, title, children }: { id: string; n: number; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className={s.section}>
      <div className={s.sectionHead}>
        <span className={s.raceNo} aria-hidden="true">
          {String(n).padStart(2, "0")}
        </span>
        <span>
          <span className="label">Sector {n}</span>
          <h2>{title}</h2>
        </span>
      </div>
      {children}
    </section>
  );
}

export default function Report() {
  const { arms, comparisons, training, data, judge, latency, eval_set, safety } = site;
  const cb = comparisons.C_vs_B;
  const ca = comparisons.C_vs_A;
  const ba = comparisons.B_vs_A;
  const val = training.validation;
  const valMin = val.reduce((a, b) => (b.loss < a.loss ? b : a));
  const valLast = val[val.length - 1];
  const lastStep = training.steps[training.steps.length - 1];
  const chosen = training.epochs.find((e) => e.epoch === training.chosen_epoch)!;
  const epoch2 = training.epochs.find((e) => e.epoch === 2)!;
  const evalTotal = Object.values(eval_set.splits).reduce((a, b) => a + b, 0);
  let fig = 0;
  const next = () => ++fig;

  return (
    <div className={`page ${s.layout}`}>
      <aside className={s.rail}>
        <Toc entries={SECTIONS} />
      </aside>

      <article className={s.article}>
        <header className={s.header}>
          <p className="label">Interactive report</p>
          <h1>How PITWALL AI was built and measured</h1>
          <p className={s.lede}>
            The full story behind the live demo: how I built an exam the model could not have seen, generated training
            data without breaking anyone&apos;s terms, trained the adapter, checked the judge, and read the results
            without flattering them. Every chart below is drawn from the files committed in the{" "}
            <a href={LINKS.github}>repository</a>.
          </p>
          <dl className={s.meta}>
            <div>
              <dt>Base model</dt>
              <dd>Qwen3-1.7B · non-thinking</dd>
            </div>
            <div>
              <dt>Method</dt>
              <dd>LoRA r 16 · all linear layers</dd>
            </div>
            <div>
              <dt>Test set</dt>
              <dd>{eval_set.splits.test} questions · frozen</dd>
            </div>
            <div>
              <dt>Judge</dt>
              <dd>Claude Opus 5.5 + code rules</dd>
            </div>
          </dl>
        </header>

        <Section id="question" n={1} title="The question">
          <p className={s.pull}>
            Can a LoRA adapter make a small open model answer everyday questions in Formula 1 language more reliably
            than a well-written system prompt, without making the answers less correct or less helpful?
          </p>
          <p>
            Three arms share one base model, one pinned revision, identical sampling settings and identical seeds. The
            only differences are a system prompt for arm B and the adapter for arm C, so any change in the results comes
            from those two things.
          </p>
          <Figure
            n={next()}
            title="The three arms"
            caption="Arm B's prompt was tuned for two rounds on the dev set (three versions scored 5, 7 and 6 of 48) and then frozen, so the adapter faces a strong baseline, not a straw man."
          >
            <ArmsDiagram />
          </Figure>
          <p>
            <b>The primary metric is joint success:</b> an answer counts only if it is in F1 style (style score ≥ 2 of
            3), correct, and helpful (≥ 2 of 3). An answer that is stylish but wrong fails, and so does one that is
            right but plain. The denominator is every test question: crashes, refusals and cut-off answers count as
            failures.
          </p>
          <p>
            I wrote the metric, the statistics and the exact wording of every possible conclusion into a{" "}
            <a href={`${LINKS.github}/blob/main/docs/PREREGISTRATION.md`}>pre-registration</a> before any training data
            existed, so the result could not be argued into shape afterwards.
          </p>
        </Section>

        <Section id="exam" n={2} title="The exam">
          <p>
            The evaluation set came first. {evalTotal} items in four groups, each with a job, and only the test group is
            used for the reported results. I wrote {eval_set.handwritten} of the test questions myself; the rest were
            generated with Claude Opus 5.5 to my specification. I checked every answer key by hand, and numeric keys are
            computed by Python, never taken from a model.
          </p>
          <Figure n={next()} title="What the 196 evaluation items are for">
            <EvalComposition splits={eval_set.splits} handwritten={eval_set.handwritten} />
          </Figure>
          <p>
            Then I locked it: every file was fingerprinted with SHA-256 and the fingerprints were written into the
            pre-registration. The audit script checks that the files still match and that only one commit ever touched
            them. The order of events matters, so here it is, from the commit history:
          </p>
          <ol className={s.timeline}>
            {site.timeline.map((t) => (
              <li key={t.commit}>
                <span className={`${s.when} mono`}>{t.when.replace("18 Oct 2026, ", "")}</span>
                <span>{t.event}</span>
                <code>{t.commit}</code>
              </li>
            ))}
          </ol>
        </Section>

        <Section id="data" n={3} title="Training data">
          <p>
            Anthropic, OpenAI and Google all forbid using their models&apos; outputs to train competing models. So every
            training example comes from two Apache-2.0 open-weight models I ran myself on a rented A100: a{" "}
            <b>teacher</b> ({data.teacher}) and a <b>checker</b> from a different model family ({data.checker}), so the
            two don&apos;t share blind spots.
          </p>
          <Figure n={next()} title="How one training example is made">
            <Steps
              steps={[
                {
                  title: "Write a question",
                  body: "8 everyday categories × 36 topics, plus 4 kinds of serious situation.",
                  tag: "teacher",
                },
                {
                  title: "Answer it plainly first",
                  body: "A correct, neutral answer: the reference the rewrite is checked against.",
                  tag: "teacher",
                },
                {
                  title: "Rewrite it in F1 language",
                  body: "Under the same style rules arm B's prompt uses, so the comparison is fair.",
                  tag: "teacher",
                },
                {
                  title: "Check nothing was lost",
                  body: "Reject the rewrite if a fact, number, step or warning is missing, a claim is added, steps are reordered or a metaphor blurs the meaning.",
                  tag: "checker",
                },
                {
                  title: "Run the code gates",
                  body: "Every number must survive, code blocks must be unchanged, length within 1.3× the plain answer, no opening phrase on more than 5% of answers.",
                  tag: "code",
                },
                {
                  title: "Retry and select",
                  body: "Rewrites that failed only on length get one retry with a word budget; then 500 are chosen, balanced by category.",
                  tag: "code",
                },
              ]}
            />
          </Figure>
          <Figure
            n={next()}
            title="From 924 questions to 500 examples"
            caption={`The ${data.heldout} accepted examples not selected (blue) were kept aside to measure validation loss. Questions resembling any of the ${evalTotal} evaluation items were removed before answering; the audit finds no near-duplicates and no verbatim copies.`}
          >
            <Funnel data={data} />
          </Figure>
          <Figure
            n={next()}
            title="Why rewrites were rejected"
            caption={
              <>
                Blue: the checker model. Grey: code gates. One example can fail several checks. The checker also caught{" "}
                {data.reject_reasons["checker:reference_incorrect"]} cases where the teacher&apos;s own plain answer was
                wrong.
              </>
            }
          >
            <RejectReasons reasons={data.reject_reasons} />
          </Figure>
        </Section>

        <Section id="training" n={4} title="Training">
          <p>
            LoRA keeps the {(training.total_params / 1e9).toFixed(2)} billion original weights frozen and trains two
            thin matrices next to each linear layer.{" "}
            <b>
              {thousands(training.trainable_params)} parameters train,{" "}
              {pct(training.trainable_params / training.total_params, 2)} of the model.
            </b>{" "}
            The adapter is a 70 MB file, and switching it off gives back the base model exactly, which is what made the
            A-versus-C comparison clean.
          </p>
          <Figure n={next()} title="LoRA in one picture">
            <LoraDiagram />
          </Figure>
          <div className={s.tableWrap}>
            <table className={s.table}>
              <tbody>
                <tr>
                  <th scope="row">Loss</th>
                  <td>
                    Answer tokens only (the question is masked with −100), end-of-turn token included, so it learns to
                    stop
                  </td>
                </tr>
                <tr>
                  <th scope="row">Optimiser</th>
                  <td>
                    AdamW, learning rate 2e-4, cosine schedule, 18 warm-up steps, effective batch 8, bf16, seed 42
                  </td>
                </tr>
                <tr>
                  <th scope="row">Length</th>
                  <td>
                    6 epochs ({training.total_steps} steps), validation every half epoch on {data.heldout} held-out
                    examples, a checkpoint every epoch
                  </td>
                </tr>
                <tr>
                  <th scope="row">Compute</th>
                  <td>
                    One {training.gpu.replace("NVIDIA ", "")}, {training.minutes.toFixed(1)} minutes,{" "}
                    {training.peak_memory_gb} GB peak memory
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <Figure
            n={next()}
            title="Training curves"
            caption={`Each step is one batch of 8 examples, so the raw line is noisy; the bold line is a 15-step mean. Validation loss is measured on examples never trained on.`}
          >
            <TrainingChart
              steps={training.steps}
              validation={val}
              stepsPerEpoch={training.steps_per_epoch}
              chosenEpoch={training.chosen_epoch}
            />
          </Figure>
          <p>
            <b>Reading the curves.</b> Validation loss was lowest at epoch {Math.round(valMin.epoch)} (
            {valMin.loss.toFixed(2)}) and rose to {valLast.loss.toFixed(2)}, while training loss kept falling to{" "}
            {lastStep.loss.toFixed(2)}. That is overfitting in the word-for-word sense: with 500 examples, the adapter
            first learns the style, then starts to memorise the exact wording of its training answers. But validation
            loss is not the goal. It punishes any wording that differs from one reference answer, while the task asks
            whether an answer is in style, correct and helpful.
          </p>
          <Figure
            n={next()}
            title="Choosing the epoch: the task score against validation loss"
            caption={`Bars: dev joint success out of ${chosen.n} questions for each saved epoch. Line: validation loss. The rule was written down before the run: highest dev joint success, ties to the lower validation loss.`}
          >
            <EpochChart epochs={training.epochs} chosen={training.chosen_epoch} />
          </Figure>
          <p>
            So the two signals disagree, and the rule I recorded in advance decided for the task score: epoch{" "}
            {chosen.epoch} ({chosen.jsr}/{chosen.n} on dev, against {epoch2.jsr}/{epoch2.n} at epoch 2). Picking the
            best of six noisy dev scores flatters the winner, which is why only the untouched test set is reported. The
            original 3-epoch adapter, trained before this run, is reported too: on the test set it is not reliably
            different, so this choice does not drive the conclusions.
          </p>
          <Figure
            n={next()}
            title="How much data the style needs"
            caption="Dev-set counts out of 48 for adapters trained on the first 125, 250 and all 500 examples. Exploratory: one seed per size."
          >
            <LearningCurve points={training.learning_curve} />
          </Figure>
          <p>
            125 examples barely moved the style; 250 delivered most of it; 500 added a little more. Correctness stayed
            flat at every size: more data buys a more consistent voice, not more knowledge.
          </p>
        </Section>

        <Section id="judging" n={5} title="Judging">
          <p>
            Every answer was graded by Claude Opus 5.5, a different model family from both the student and the teacher,
            so it is not grading text that resembles its own. It sees one anonymous answer at a time and never learns
            which arm wrote it. It does not hand out a verdict: it fills a structured form, and fixed Python rules turn
            the form into a verdict.
          </p>
          <Figure n={next()} title="From answer to verdict">
            <Steps
              steps={[
                {
                  title: "One anonymous answer",
                  body: "With the question and its answer key: the key points, each marked critical or not.",
                },
                {
                  title: "The judge fills a form",
                  body: "Style 0–3 and helpfulness 0–3, each with quoted evidence; every key point present, missing or contradicted; the final answer extracted; any extra error, marked serious or minor.",
                  tag: "Claude Opus 5.5",
                },
                {
                  title: "Code decides correctness",
                  body: "Every critical point present, nothing contradicted, no serious extra error; numbers within 1% of the key; short answers must match the key or an accepted alias.",
                  tag: "src/pitwall/judge.py",
                },
                { title: "Joint success", body: "Style ≥ 2, correct, and helpfulness ≥ 2, over all 120 questions." },
              ]}
            />
          </Figure>
          <div className={s.tableWrap}>
            <table className={s.table}>
              <thead>
                <tr>
                  <th scope="col">Score</th>
                  <th scope="col">Style</th>
                  <th scope="col">Helpfulness</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th scope="row">0</th>
                  <td>No F1 language</td>
                  <td>Unusable</td>
                </tr>
                <tr>
                  <th scope="row">1</th>
                  <td>Bolted on: one opening line or a few terms</td>
                  <td>Important information missing or wrong</td>
                </tr>
                <tr>
                  <th scope="row">2 · pass</th>
                  <td>F1 framing runs through most of the answer</td>
                  <td>The user could act on it</td>
                </tr>
                <tr>
                  <th scope="row">3</th>
                  <td>Runs through all of it, varied and natural</td>
                  <td>Complete, clear and concise</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p>
            <b>Can the judge be trusted?</b> I didn&apos;t assume so; I tested it. I planted {judge.canaries_total}{" "}
            deliberate mistakes in otherwise good answers, and it caught {judge.canaries_caught} of them, rating the
            harmless slips as minor, as it should.
          </p>
          <Figure
            n={next()}
            title="Planted mistakes caught by the judge"
            caption="Each square is one planted-mistake answer; green means the judge's verdict changed as it should."
          >
            <CanaryGrid canaries={judge.canaries} />
          </Figure>
          <p>
            Re-judging {judge.retest.n} answers gave the same joint verdict {judge.retest.same_joint} times out of{" "}
            {judge.retest.n} and the same correctness verdict {judge.retest.same_correct} times. I also labelled a blind
            sample of {judge.human.n} answers myself (15 per arm) as a rapid first pass. On pass or fail, my style
            labels agreed strongly with the judge (Cohen&apos;s κ = {judge.human.kappa_style.toFixed(2)}), but
            correctness and helpfulness agreed poorly (κ = {judge.human.kappa_correct.toFixed(2)} and{" "}
            {judge.human.kappa_helpful.toFixed(2)}): my quick pass was more lenient. I then reviewed all{" "}
            {judge.human.disputed} disputed scores side by side with the judge&apos;s quoted evidence and sided with the
            judge in {judge.human.sided_with_judge}: it had found real errors a quick read misses. That review was not
            blind, so the correctness and helpfulness scores remain judge-dependent. Every arm faces the same judge, so
            the comparisons between arms are sounder than the absolute rates.
          </p>
        </Section>

        <Section id="results" n={6} title="Results">
          <p>
            All results are on the {eval_set.splits.test} frozen test questions, using the first of three samples per
            question, as pre-registered.
          </p>
          <Figure
            n={next()}
            title="Each arm on each measure"
            caption="Bars: share of the 120 test questions. Whiskers: 95% Wilson intervals."
          >
            <RateBars arms={arms} />
          </Figure>
          <Figure
            n={next()}
            title="Paired differences between arms"
            caption="Points: difference in percentage points on the same questions. Lines: 95% paired-bootstrap intervals (10,000 resamples). Green or red: the interval excludes zero. Grey: no reliable difference."
          >
            <ForestPlot comparisons={comparisons} />
          </Figure>
          <div className={s.findings}>
            <p>
              <b>Primary result.</b> C − B joint success is {signed(cb.joint_success.diff)} points (95% CI{" "}
              {signed(cb.joint_success.low)} to {signed(cb.joint_success.high)}; exact McNemar p ={" "}
              {cb.joint_success.p.toFixed(2)}, {cb.joint_success.only_first} questions where only C succeeded against{" "}
              {cb.joint_success.only_second} where only B did). Under the pre-registered wording that is{" "}
              <b>no reliable difference</b>: the estimate favours the adapter, but the interval touches zero.
            </p>
            <p>
              <b>Style.</b> The adapter makes the voice far more reliable: {signed(cb.style.diff)} points over the
              prompt (CI {signed(cb.style.low)} to {signed(cb.style.high)}), with no prompt at all.
            </p>
            <p>
              <b>Correctness.</b> Adding style costs a 1.7B model accuracy, whichever way it is added: C is{" "}
              {Math.abs(ca.correct.diff).toFixed(1)} points below the base model and B is{" "}
              {Math.abs(ba.correct.diff).toFixed(1)} below. C and B do not differ ({signed(cb.correct.diff)}, CI{" "}
              {signed(cb.correct.low)} to {signed(cb.correct.high)}).
            </p>
            <p>
              <b>Sensitivity.</b> If any extra mistake fails an answer (the strict rule), C and B tie at{" "}
              {arms.C.joint_strict.s}/{arms.C.joint_strict.n} and {arms.B.joint_strict.s}/{arms.B.joint_strict.n}: the
              adapter&apos;s lead disappears, and under either rule there is no reliable difference.
            </p>
          </div>
          <Figure
            n={next()}
            title="Where the answers succeed, by category"
            caption="Counts out of 15 test questions per category. The biggest correctness losses from base to adapter are in writing tasks, calculations and false premises. Procedures fail for every arm: their keys need every critical step."
          >
            <CategoryGrid categories={site.categories} />
          </Figure>
          <p>
            Want to see the answers behind these numbers? The <Link href="/explorer">results explorer</Link> has every
            test question with all three answers and the judge&apos;s notes.
          </p>
        </Section>

        <Section id="cost" n={7} title="Fine-tune or prompt?">
          <p>
            The task allowed either route, so I measured both instead of assuming. Arm B resends its system prompt with
            every request; the adapter carries the style in its weights.
          </p>
          <Figure
            n={next()}
            title="Inference cost per request"
            caption={`Medians over 10 dev questions with ${latency.new_tokens}-token answers on an Apple Silicon laptop (${latency.device}, fp16). The adapter runs unmerged here so it can be switched off per request; merging it into the base weights removes its decoding overhead.`}
          >
            <LatencyBars latency={latency} />
          </Figure>
          <div className={s.compare}>
            <div>
              <p className="label">The adapter gives</p>
              <ul>
                <li>
                  A more reliable voice: {pct(arms.C.style.rate)} of answers reach the style threshold against{" "}
                  {pct(arms.B.style.rate)}
                </li>
                <li>
                  Shorter requests and a faster start: {latency.C.median_prompt_tokens} prompt tokens instead of{" "}
                  {latency.B.median_prompt_tokens}, first token in {Math.round(latency.C.median_first_token_ms)} ms
                  instead of {Math.round(latency.B.median_first_token_ms)} ms
                </li>
                <li>
                  A persona the user cannot talk it out of: asked for &ldquo;no metaphors&rdquo;, it kept the F1 voice
                  in 3 of 3 probes
                </li>
              </ul>
            </div>
            <div>
              <p className="label">The prompt gives</p>
              <ul>
                <li>No training data and no GPU: two rounds of tuning on 48 dev questions</li>
                <li>Easy to change or switch off, which matters for formal writing and crises</li>
                <li>It mostly honoured &ldquo;no metaphors&rdquo;, with at most a light F1 touch</li>
              </ul>
            </div>
          </div>
          <p>
            <b>Conclusion.</b> Fine-tuning delivered substantially stronger Formula 1 style adherence and a faster first
            token than prompting, but did not establish a reliable advantage in combined answer quality. Both approaches
            reduced correctness and helpfulness compared with the base model. For an application prioritising reliable
            information, neither styled version is ready for unrestricted deployment.
          </p>
          <p>
            Neither is a security boundary: prompt injection worked on every arm in at least one of three injection
            probes. The one-off cost of the adapter was about 2 GPU-hours for the data and {training.minutes.toFixed(1)}{" "}
            minutes for the training.
          </p>
        </Section>

        <Section id="safety" n={8} title="Serious situations">
          <p>
            &ldquo;Any question&rdquo; includes questions where style must step aside: someone collapsing, a gas leak, a
            crisis. For the 16 safety questions I applied a strict tone rule of my own: any racing language in an answer
            to a serious situation is unacceptable, even when the safe action is present.
          </p>
          <Figure
            n={next()}
            title="Serious situations: tone and content"
            caption="Left: my tone check (deviation D-2 in the pre-registration). Right: the judge's strict rule, which needs every critical step and nothing unsafe; it fails nearly every answer from every arm, the base model included."
          >
            <ToneBars safety={safety} />
          </Figure>
          <p>
            The adapter was acceptable far more often than the prompt, but still only in {safety.tone_acceptable.C[0]}{" "}
            of 16. This is the clearest case for making the style register-aware.
          </p>
        </Section>

        <Section id="limits" n={9} title="Limits and next steps">
          <div className={s.compare}>
            <div>
              <p className="label">Limits</p>
              <ul>
                <li>One judge model and one human rater</li>
                <li>Synthetic training data from a single teacher model</li>
                <li>One training seed, and only the first of three samples per question judged</li>
                <li>
                  At 120 questions the design detects differences of roughly 10 points, so the primary result is a lead,
                  not a verdict
                </li>
              </ul>
            </div>
            <div>
              <p className="label">With more time</p>
              <ul>
                <li>Make the style register-aware: switch it off for crises and formal writing</li>
                <li>Test whether a 4B model loses less correctness when it adds style</li>
                <li>Add human-written training examples for serious situations</li>
                <li>More judges and raters, several seeds, and all three samples per question</li>
              </ul>
            </div>
          </div>
        </Section>

        <Section id="reproduce" n={10} title="Reproduce it">
          <p>
            Everything above can be checked from the repository. The analysis rebuilds the results byte for byte from
            the committed judgements, and the audit recomputes the headline numbers independently.
          </p>
          <CodeBlock
            label="Check the published numbers"
            code={[
              "git clone https://github.com/enxo7899/pitwall-ai && cd pitwall-ai",
              "uv sync --all-groups",
              'uv run pytest -m "not gpu and not slow"',
              "uv run python scripts/tools/audit.py",
              "uv run python scripts/evaluation/analyze_results.py",
            ].join("\n")}
          />
          <p>
            Re-running the full pipeline costs about $4 of GPU time (one A100 for about two hours) and about $10 of
            Claude API calls at batch prices.
          </p>
          <div className={s.links}>
            <a href={LINKS.github} className="btn">
              Code and data ↗
            </a>
            <a href={LINKS.model} className="btn">
              Adapter on Hugging Face ↗
            </a>
            <a href={`${LINKS.github}/blob/main/docs/REPORT.pdf`} className="btn">
              2-page write-up ↗
            </a>
          </div>
          <p className={s.colophon}>
            I designed and led this project. I used Claude Code (Anthropic) as an AI engineering assistant for research
            and code generation; generated code went through the same tests, audit and gate reviews as everything else.
            Claude Opus 5.5 also generated the evaluation questions I did not write, to my specification, and serves as
            the judge. No Claude output is used as training data.
          </p>
        </Section>
      </article>
    </div>
  );
}
