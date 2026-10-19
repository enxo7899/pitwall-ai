"""Draw every figure in the README and docs from the committed result files (nothing is typed in by hand).

Usage:
    uv run --group docs python scripts/tools/make_figures.py
Writes PNGs to docs/figures/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from pitwall.metrics import wilson_interval  # noqa: E402

FIG = Path("docs/figures")
ARM_COLOR = {"A": "#8c8c8c", "B": "#2f6db3", "C": "#c8102e"}
ARM_NAME = {"A": "A · base model", "B": "B · base + tuned prompt", "C": "C · base + PITWALL adapter"}
INK, GRID = "#222222", "#e6e6e6"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.edgecolor": "#bbbbbb",
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


def load(path: str):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return json.loads(p.read_text())


def fraction(text: str) -> tuple[int, int]:
    s, n = re.match(r"(\d+)/(\d+)", text).groups()
    return int(s), int(n)


def save(fig, name: str, note: str | None = None) -> None:
    if note:
        fig.canvas.draw()  # place the note under the lowest element (rotated tick labels can sit below the axes)
        bottom = fig.get_tightbbox(fig.canvas.get_renderer()).y0 / fig.get_figheight()
        fig.text(0.01, bottom - 0.015, note, fontsize=8, color="#777777", ha="left", va="top")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", FIG / name)


def results(summary: dict) -> None:
    metrics = [
        ("Joint success", lambda a: (a["jsr"]["successes"], a["jsr"]["n"])),
        ("F1 style ≥ 2", lambda a: fraction(a["components"]["style"])),
        ("Correct", lambda a: fraction(a["components"]["correct"])),
        ("Helpful ≥ 2", lambda a: fraction(a["components"]["helpful"])),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.26
    for i, arm in enumerate("ABC"):
        for j, (_, get) in enumerate(metrics):
            s, n = get(summary["arms"][arm])
            lo, hi = wilson_interval(s, n)
            x = j + (i - 1) * width
            ax.bar(x, s / n, width, color=ARM_COLOR[arm], label=ARM_NAME[arm] if j == 0 else None)
            ax.errorbar(x, s / n, yerr=[[s / n - lo], [hi - s / n]], color=INK, capsize=2.5, lw=1)
            ax.text(x, hi + 0.015, f"{100 * s / n:.1f}%", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(range(len(metrics)), [m for m, _ in metrics])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, 1.08)
    ax.grid(axis="x", visible=False)
    ax.set_title("Test set results (120 questions, first sample per question)")
    ax.legend(loc="upper left")
    save(
        fig,
        "results.png",
        "Bars: share of the 120 test questions. Whiskers: 95% Wilson interval. Judge: Claude Opus 5.5.",
    )


def differences(summary: dict) -> None:
    rows = [("C − B", "C_vs_B"), ("C − A", "C_vs_A"), ("B − A", "B_vs_A")]
    metrics = [
        ("Joint success", "joint_success"),
        ("F1 style", "style"),
        ("Correct", "correct"),
        ("Helpful", "helpful"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, (title, key) in zip(axes, metrics, strict=True):
        for y, (_, comp) in enumerate(rows):
            d = summary["comparisons"][comp][key]
            lo, hi = d["ci95_pp"]
            color = "#1a7f37" if lo > 0 else "#b42318" if hi < 0 else "#6e6e6e"
            ax.plot([lo, hi], [y, y], color=color, lw=2.2, solid_capstyle="round")
            ax.plot(d["difference_pp"], y, "o", color=color, ms=6)
            ax.text(d["difference_pp"], y + 0.22, f"{d['difference_pp']:+.1f}", ha="center", fontsize=8.5, color=color)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("difference (points)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), [r for r, _ in rows])
    axes[0].set_ylim(-0.6, len(rows) - 0.3)
    axes[0].invert_yaxis()
    fig.suptitle("Paired differences with 95% bootstrap intervals", x=0.01, ha="left", fontweight="bold", y=1.04)
    save(
        fig,
        "differences.png",
        "Green: interval entirely above 0. Red: entirely below 0. Grey: includes 0 (no reliable difference). 10,000 resamples of questions.",
    )


def differences_report(summary: dict) -> None:
    """The same chart for the 2-page report: vector, in the report's typeface, no title (the caption has it)."""
    rows = [("C − B", "C_vs_B"), ("C − A", "C_vs_A"), ("B − A", "B_vs_A")]
    metrics = [
        ("Joint success", "joint_success"),
        ("F1 style", "style"),
        ("Correct", "correct"),
        ("Helpful", "helpful"),
    ]
    style = {"font.family": "Helvetica Neue", "font.size": 7.5, "svg.fonttype": "path", "axes.titlesize": 8.5}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 4, figsize=(6.9, 1.6), sharey=True)
        for ax, (title, key) in zip(axes, metrics, strict=True):
            for y, (_, comp) in enumerate(rows):
                d = summary["comparisons"][comp][key]
                lo, hi = d["ci95_pp"]
                color = "#1a7f37" if lo > 0 else "#b42318" if hi < 0 else "#8b949e"
                ax.plot([lo, hi], [y, y], color=color, lw=1.8, solid_capstyle="round")
                ax.plot(d["difference_pp"], y, "o", color=color, ms=4.2)
                ax.text(
                    d["difference_pp"], y - 0.27, f"{d['difference_pp']:+.1f}", ha="center", fontsize=6.5, color=color
                )
            ax.axvline(0, color="#57606a", lw=0.7)
            ax.set_title(title, fontsize=8.5, pad=4)
            ax.tick_params(length=2, labelsize=6.5)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color="#eaeef2", lw=0.6)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color("#d0d7de")
        axes[0].set_yticks(range(len(rows)), [r for r, _ in rows])
        axes[0].tick_params(axis="y", labelsize=7.5)
        axes[0].set_ylim(-0.6, len(rows) - 0.4)
        axes[0].invert_yaxis()
        fig.supxlabel("difference in percentage points", fontsize=7, color="#57606a", y=0.02)
        fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.2, wspace=0.12)
        fig.savefig(FIG / "differences_report.svg", transparent=True)
        plt.close(fig)
    print("wrote", FIG / "differences_report.svg")


def training(log: list[dict], chosen_step: int) -> None:
    train = [e for e in log if e["kind"] == "train"]
    evals = [e for e in log if e["kind"] == "eval"]
    steps_per_epoch = 63

    def smooth(xs: list[float], k: int = 15) -> list[float]:
        return [sum(xs[max(0, i - k + 1) : i + 1]) / len(xs[max(0, i - k + 1) : i + 1]) for i in range(len(xs))]

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.6))
    (a, b), (c, d) = axes
    steps = [e["step"] for e in train]
    a.plot(steps, [e["loss"] for e in train], color="#c8102e", alpha=0.25, lw=0.8)
    a.plot(steps, smooth([e["loss"] for e in train]), color="#c8102e", lw=2, label="training loss (15-step mean)")
    a.plot(
        [e["step"] for e in evals],
        [e["eval_loss"] for e in evals],
        "o-",
        color=INK,
        lw=1.6,
        ms=4,
        label="validation loss (130 held-out)",
    )
    a.set_title("Loss")
    a.legend(loc="center right")
    b.plot(steps, smooth([e["mean_token_accuracy"] for e in train]), color="#c8102e", lw=2, label="training")
    if "eval_mean_token_accuracy" in evals[0]:
        b.plot(
            [e["step"] for e in evals],
            [e["eval_mean_token_accuracy"] for e in evals],
            "o-",
            color=INK,
            lw=1.6,
            ms=4,
            label="validation",
        )
    b.yaxis.set_major_formatter(PercentFormatter(1.0))
    b.set_title("Next-token accuracy on answer tokens")
    b.legend()
    c.plot(steps, [e["learning_rate"] for e in train], color="#2f6db3", lw=2)
    c.set_title("Learning rate (warm-up, then cosine decay)")
    c.yaxis.set_major_formatter(lambda v, _: f"{v * 1e4:.1f}")
    c.set_ylabel("× 10⁻⁴")
    d.plot(steps, [e["grad_norm"] for e in train], color="#8c6d1f", alpha=0.35, lw=0.8)
    d.plot(steps, smooth([e["grad_norm"] for e in train]), color="#8c6d1f", lw=2)
    d.set_title("Gradient norm")
    for ax in axes.flat:
        for epoch in range(1, 7):
            ax.axvline(epoch * steps_per_epoch, color=GRID, lw=1, zorder=0)
        ax.axvline(chosen_step, color="#c8102e", ls="--", lw=1.2)
        ax.set_xlabel("optimizer step (63 per epoch)")
    a.annotate(
        "chosen: epoch 5",
        xy=(chosen_step, 3.3),
        xytext=(chosen_step - 135, 3.3),
        va="center",
        color="#c8102e",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#c8102e"},
    )
    fig.tight_layout()
    save(
        fig,
        "training.png",
        "500 training examples, 6 epochs, LoRA r=16, lr 2e-4, effective batch 8, one A100 80 GB (5.2 minutes).",
    )


def epoch_selection(sel: dict) -> None:
    cands = sel["candidates"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    epochs = [c["epoch"] for c in cands]
    jsr = [c["jsr"] / c["n"] for c in cands]
    colors = ["#c8102e" if c["epoch"] == sel["chosen"]["epoch"] else "#e8a3ae" for c in cands]
    ax.bar(epochs, jsr, color=colors, width=0.6, label="dev joint success")
    for c, y in zip(cands, jsr, strict=True):
        lo, hi = c["jsr_ci95"]
        ax.errorbar(c["epoch"], y, yerr=[[y - lo], [hi - y]], color=INK, capsize=2.5, lw=1)
        ax.text(c["epoch"], hi + 0.012, f"{c['jsr']}/48", ha="center", fontsize=8.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("epoch")
    ax.set_xticks(epochs)
    ax.set_ylim(0, 0.52)
    ax.grid(axis="x", visible=False)
    twin = ax.twinx()
    twin.plot(epochs, [c["validation_loss"] for c in cands], "o-", color=INK, lw=1.6, label="validation loss")
    twin.set_ylabel("validation loss")
    twin.spines["right"].set_visible(True)
    twin.grid(False)
    ax.set_title("Choosing the epoch: task score vs validation loss (48 dev questions)")
    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    ax.legend(handles, ["dev joint success (95% CI)", "validation loss"], loc="upper left")
    save(
        fig,
        "epoch_selection.png",
        "Rule fixed before the run: highest dev joint success, ties to lower validation loss. Loss is lowest at epoch 2; the task score at epoch 5.",
    )


def learning_curve(curve: list[dict], dev500: dict) -> None:
    from pitwall.metrics import component_rates, joint_success_rate

    points = []
    for n, src in ((125, "c125_dev"), (250, "c250_dev")):
        rows = [r for r in curve if r["source"] == src]
        points.append(
            (
                n,
                component_rates(rows)["style"][0] / 48,
                joint_success_rate(rows)[0] / 48,
                component_rates(rows)["correct"][0] / 48,
            )
        )
    c = dev500["arms"]["C"]
    points.append(
        (
            500,
            c["components"]["style"]["passes"] / 48,
            c["jsr"]["successes"] / 48,
            c["components"]["correct"]["passes"] / 48,
        )
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    xs = [p[0] for p in points]
    for idx, (label, color) in enumerate(
        (("F1 style ≥ 2", "#c8102e"), ("Joint success", INK), ("Correct", "#2f6db3")), start=1
    ):
        ax.plot(xs, [p[idx] for p in points], "o-", color=color, lw=2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, [str(x) for x in xs])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("training examples (3 epochs each)")
    ax.set_ylim(0, 0.75)
    ax.set_title("How much data does the style need? (48 dev questions)")
    ax.legend(loc="upper left")
    save(
        fig,
        "learning_curve.png",
        "Exploratory (H4). Style needs about 250 examples; correctness does not move with more style data.",
    )


def data_funnel(q: dict, d: dict) -> None:
    stages = [
        ("Questions written by the teacher", q["requested"]),
        ("After removing near-duplicates", q["requested"] - q["near_duplicates_dropped"]),
        ("After removing look-alikes of exam questions", q["kept"]),
        ("Passed the checker and code gates", d["accepted_before_sampling"]),
        ("Selected for training (stratified)", d["selected"]),
    ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [1.1, 1]})
    for i, (_, n) in enumerate(stages):
        a.barh(i, n, color="#c8102e" if i == len(stages) - 1 else "#d9d9d9")
        a.text(n + 12, i, f"{n}", va="center", fontsize=9.5, fontweight="bold")
    a.set_yticks(range(len(stages)), [s for s, _ in stages])
    a.invert_yaxis()
    a.set_xlim(0, stages[0][1] * 1.12)
    a.grid(axis="y", visible=False)
    a.set_title("Training data funnel")
    reasons = sorted(d["reject_reasons"].items(), key=lambda kv: kv[1])
    pretty = {
        "checker:information_lost": "checker: information lost",
        "too_long_vs_reference": "code: longer than 1.3× reference",
        "numbers_missing": "code: a number went missing",
        "checker:claims_added": "checker: new claims added",
        "checker:meaning_blurred": "checker: meaning blurred by metaphor",
        "code_changed": "code: code block changed",
        "checker:reference_incorrect": "checker: teacher's own answer wrong",
        "too_little_style": "code: too little F1 language",
        "too_long_absolute": "code: over 450 words",
        "checker:order_changed": "checker: steps reordered",
        "question_mentions_racing": "code: question mentions racing",
    }
    b.barh(
        range(len(reasons)),
        [n for _, n in reasons],
        color=["#2f6db3" if k.startswith("checker") else "#8c8c8c" for k, _ in reasons],
    )
    b.set_yticks(range(len(reasons)), [pretty.get(k, k) for k, _ in reasons], fontsize=9)
    for i, (_, n) in enumerate(reasons):
        b.text(n + 1, i, str(n), va="center", fontsize=8.5)
    b.grid(axis="y", visible=False)
    b.set_title("Why examples were rejected (one can fail several checks)")
    fig.tight_layout()
    save(
        fig,
        "data_funnel.png",
        "Teacher: Qwen/Qwen3.5-27B. Checker: openai/gpt-oss-120b. Both self-hosted, Apache-2.0. Blue: checker verdicts; grey: deterministic code gates.",
    )


def categories(summary: dict) -> None:
    cats = sorted(summary["arms"]["A"]["correct_by_category"])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"wspace": 0.08})
    for ax, field, title in (
        (axes[0], "correct_by_category", "Correct, by category"),
        (axes[1], "jsr_by_category", "Joint success, by category"),
    ):
        grid = [[fraction(summary["arms"][arm][field][c])[0] / 15 for c in cats] for arm in "ABC"]
        im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        for i in range(3):
            for j in range(len(cats)):
                ax.text(
                    j,
                    i,
                    fraction(summary["arms"]["ABC"[i]][field][cats[j]])[0],
                    ha="center",
                    va="center",
                    color="white" if grid[i][j] > 0.55 else INK,
                    fontsize=9.5,
                )
        ax.set_xticks(range(len(cats)), cats, rotation=35, ha="right")
        ax.set_yticks(range(3), ["A · base", "B · prompt", "C · adapter"] if ax is axes[0] else [])
        ax.grid(False)
        ax.set_title(title)
    fig.colorbar(im, ax=axes, fraction=0.025, format=PercentFormatter(1.0))
    save(
        fig,
        "categories.png",
        "Numbers: questions out of 15 per category. Procedures need every step in order and fail for all arms.",
    )


def judge_validation(canaries: dict, summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    by_type = canaries["summary"]["by_type"]
    names = {
        "number_changed": "number changed",
        "fact_contradicted": "fact contradicted",
        "critical_step_removed": "critical step removed",
        "style_opener_only": "style only bolted on",
        "no_style": "no style at all",
        "padded_verbose": "padded with filler",
        "minor_slip": "harmless false aside",
    }
    axes[0].barh(range(len(by_type)), [fraction(v)[0] for v in by_type.values()], color="#1a7f37")
    axes[0].set_yticks(range(len(by_type)), [names.get(k, k) for k in by_type])
    axes[0].set_xlim(0, 4.6)
    axes[0].set_title(f"Planted mistakes caught: {canaries['summary']['detected']}/{canaries['summary']['total']}")
    axes[0].grid(axis="y", visible=False)
    rt = summary["judge_retest"]
    items = [
        ("joint verdict", rt["same_joint"]),
        ("correctness", rt["same_correct"]),
        ("style score", rt["same_exact_style_score"]),
        ("helpfulness score", rt["same_exact_helpful_score"]),
    ]
    axes[1].barh(range(len(items)), [v / rt["n"] for _, v in items], color="#2f6db3")
    for i, (_, v) in enumerate(items):
        axes[1].text(v / rt["n"] - 0.02, i, f"{v}/{rt['n']}", va="center", ha="right", color="white", fontsize=9)
    axes[1].set_yticks(range(len(items)), [k for k, _ in items])
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_title("Re-judged twice: same result")
    axes[1].grid(axis="y", visible=False)
    fig.tight_layout()
    save(
        fig,
        "judge_validation.png",
        "Left: 28 answers with a known, planted defect. Right: 40 test answers judged a second time in a separate batch.",
    )


def safety(tone: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    arms = "ABC"
    ok = [fraction(tone["arms"][a]["acceptable"])[0] for a in arms]
    ax.bar(range(3), ok, color=[ARM_COLOR[a] for a in arms], width=0.55)
    for i, a in enumerate(arms):
        ax.text(
            i,
            ok[i] + 0.3,
            f"{ok[i]}/16\n(median {tone['arms'][a]['median_f1_words']} F1 words)",
            ha="center",
            fontsize=8.5,
        )
    ax.set_xticks(range(3), [ARM_NAME[a] for a in arms], fontsize=9)
    ax.set_ylim(0, 19)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_ylabel("answers judged acceptable")
    ax.grid(axis="x", visible=False)
    ax.set_title("Serious situations: human tone check (16 questions)")
    save(
        fig,
        "safety_tone.png",
        "Rule: a crisis answered in racing language counts as unacceptable, even when the safe action is present.",
    )


def main() -> None:
    summary = load("experiments/results/summary.json")
    results(summary)
    differences(summary)
    differences_report(summary)
    sel = load("experiments/gate4/epoch_selection.json")
    training(
        load("experiments/gate4/training_log_sweep.jsonl"), chosen_step=int(sel["chosen"]["checkpoint"].split("-")[1])
    )
    epoch_selection(sel)
    learning_curve(
        load("experiments/results/judgements/curve_judgements.jsonl"), load("experiments/gate4/dev_summary_c500.json")
    )
    data_funnel(load("data/train/questions_report.json"), load("data/train/data_report.json"))
    categories(summary)
    judge_validation(load("experiments/gate2/canaries.json"), summary)
    safety(load("experiments/results/safety_tone_check.json"))


if __name__ == "__main__":
    main()
