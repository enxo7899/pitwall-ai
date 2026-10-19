"""Gate 3 training-data pipeline: prompts for the open-weight teacher and checker, plus code-based quality gates.

Pipeline per example:  taxonomy cell -> question (teacher) -> neutral answer (teacher)
                       -> Formula 1 rewrite (teacher, STYLE_RULES) -> faithfulness check (checker) -> code gates.

The neutral answer exists so the rewrite can be checked against it: style is a layer on top of a
correct answer, never a substitute. Standard library only, so the same file runs on the GPU pod.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from pitwall.style import STYLE_RULES

TEACHER_ID = "Qwen/Qwen3.5-27B"
TEACHER_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
CHECKER_ID = "openai/gpt-oss-120b"
CHECKER_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"

# Qwen's recommended non-thinking sampling for general tasks (model card), thinking switched off.
TEACHER_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}

# Same eight categories as the evaluation set, plus serious questions, so training covers what is tested.
# Domains are broad on purpose; the teacher never sees any evaluation question.
TRAIN_CATEGORIES: dict[str, str] = {
    "procedure": "step-by-step how-to for an everyday or practical task",
    "quantitative": "a calculation with numbers: percentages, conversions, rates, money, time",
    "factual": "a question with one clear, well-established factual answer",
    "explanation": "how or why something works",
    "advice": "a non-personal decision, comparison, habit or plan",
    "writing": "write or rewrite a short text (email, note, summary, bio) with stated constraints",
    "technical": "a basic programming, spreadsheet, SQL, command-line or web question",
    "uncertainty": "a question with a false premise, an unknowable answer, or real ambiguity",
    "serious": "an urgent medical, mental-health, dangerous-chemical or high-stakes legal/financial situation",
}

DOMAINS = (
    "cooking and food",
    "cleaning and laundry",
    "home repair and DIY",
    "gardening and plants",
    "travel",
    "personal finance",
    "fitness and exercise",
    "sleep and health habits",
    "pets",
    "parenting",
    "work and careers",
    "studying and exams",
    "language and grammar",
    "history",
    "geography",
    "physics",
    "chemistry",
    "biology and the human body",
    "astronomy and space",
    "weather and climate",
    "computers and phones",
    "Python programming",
    "spreadsheets",
    "databases and SQL",
    "web development",
    "music",
    "art and design",
    "sports (not motorsport)",
    "cars and bikes",
    "shopping and consumer choices",
    "paperwork and admin",
    "relationships and communication",
    "environment and recycling",
    "economics",
    "mathematics",
    "psychology and productivity",
)

NEUTRAL_SYSTEM = (
    "You are a helpful, accurate assistant. Answer the user's question correctly, clearly and completely, "
    "in plain neutral language. Prefer concise answers (usually 60-250 words); use numbered steps for "
    "procedures and code blocks for code. If the question has a false premise or cannot be known, say so plainly. "
    "For urgent medical, mental-health or safety situations, give clear direct guidance first, including "
    "contacting emergency services where appropriate."
)

REWRITE_SYSTEM = (
    "You rewrite assistant answers into the voice of PITWALL AI, an assistant that answers every question in the "
    "language of Formula 1. Rules:\n" + "\n".join(f"- {rule}" for rule in STYLE_RULES) + "\n"
    "- Keep the same information and the same order as the original answer; do not add new facts or advice.\n"
    "- Keep roughly the same length (at most about 1.2 times the original): swap plain wording for racing language "
    "instead of adding extra sentences.\n"
    "- Output only the rewritten answer, with no preamble or comment about the rewrite."
)

CHECKER_SYSTEM = (
    "You are a strict quality checker for a training dataset. You receive a QUESTION, a REFERENCE answer and a "
    "STYLED answer (the reference rewritten in Formula 1 language). Judge the STYLED answer against the "
    "REFERENCE, ignoring racing metaphors themselves. Also judge whether the REFERENCE itself is correct. "
    "Reply with JSON only."
)

CHECKER_SCHEMA = {
    "type": "object",
    "properties": {
        "reference_correct": {"type": "boolean"},
        "reference_problem": {"type": "string"},
        "all_information_kept": {"type": "boolean"},
        "changed_or_missing": {"type": "array", "items": {"type": "string"}},
        "order_kept": {"type": "boolean"},
        "added_claims": {"type": "array", "items": {"type": "string"}},
        "meaning_blurred_by_metaphor": {"type": "boolean"},
        "serious_question_trivialised": {"type": "boolean"},
    },
    "required": [
        "reference_correct",
        "reference_problem",
        "all_information_kept",
        "changed_or_missing",
        "order_kept",
        "added_claims",
        "meaning_blurred_by_metaphor",
        "serious_question_trivialised",
    ],
    "additionalProperties": False,
}

QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}


def taxonomy_cells(per_cell: int, seed_domains: tuple[str, ...] = DOMAINS) -> list[dict]:
    """Every (category, domain) pair, except 'serious', which is domain-free (its four types are the cells)."""
    cells = []
    for category, description in TRAIN_CATEGORIES.items():
        if category == "serious":
            for kind in (
                "medical emergency",
                "mental-health crisis",
                "dangerous activity or chemicals",
                "high-stakes legal or financial",
            ):
                # 5x the usual share: in the dry run only 14 of 24 serious examples survived the checker (decision D5)
                cells.append({"category": category, "domain": kind, "description": description, "n": per_cell * 5})
        else:
            for domain in seed_domains:
                cells.append({"category": category, "domain": domain, "description": description, "n": per_cell})
    return cells


def question_prompt(cell: dict) -> str:
    return (
        f"Write {cell['n']} different questions that real people might type into a chat assistant.\n"
        f"Type: {cell['category']}: {cell['description']}.\nTopic area: {cell['domain']}.\n"
        "Vary length (from a few words to three sentences), tone (casual, formal, rushed, with an occasional typo) "
        "and difficulty. Each question must make sense on its own and have a stable answer (no current events or "
        "live prices). Do not mention Formula 1, motorsport or racing. No personal data.\n"
        'Return JSON: {"questions": [...]}'
    )


def rewrite_prompt(question: str, neutral: str) -> str:
    return f"QUESTION:\n{question}\n\nORIGINAL ANSWER:\n{neutral}\n\nRewrite the original answer as PITWALL AI."


def retry_prompt(question: str, neutral: str, previous_words: int) -> str:
    """Second attempt for a faithful rewrite that ran too long: the same task plus a hard word budget."""
    limit = int(len(neutral.split()) * 1.2)
    return (
        rewrite_prompt(question, neutral) + f"\n\nLength limit: your previous rewrite had {previous_words} words; "
        f"the original has {len(neutral.split())}. This rewrite must be at most {limit} words. Swap plain wording "
        "for racing language instead of adding sentences, and keep every fact, number and step."
    )


def checker_prompt(question: str, neutral: str, styled: str) -> str:
    return (
        f"QUESTION:\n{question}\n\nREFERENCE:\n{neutral}\n\nSTYLED:\n{styled}\n\n"
        "Fields: reference_correct (is the REFERENCE factually correct and does it answer the question?), "
        "reference_problem ('' if none), all_information_kept (every fact, number, unit, step, warning and piece of "
        "code in REFERENCE is also in STYLED, exactly), changed_or_missing (list them), order_kept (steps/points in the "
        "same order), added_claims (new factual claims or advice in STYLED that are not in REFERENCE; metaphors are not "
        "claims), meaning_blurred_by_metaphor (a reader could misunderstand an instruction or fact because of the "
        "racing language), serious_question_trivialised (an urgent or serious question answered flippantly)."
    )


def parse_json_object(text: str) -> dict | None:
    """First JSON object in a model reply (tolerates code fences or stray text); None if there isn't one."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?|\n?```$", "", text).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def checker_accepts(verdict: dict | None) -> tuple[bool, str]:
    """Code decides from the checker's extracted facts; any missing field fails closed."""
    if not verdict:
        return False, "checker_unparseable"
    try:
        if not verdict["reference_correct"]:
            return False, "reference_incorrect"
        if not verdict["all_information_kept"] or verdict["changed_or_missing"]:
            return False, "information_lost"
        if not verdict["order_kept"]:
            return False, "order_changed"
        if verdict["added_claims"]:
            return False, "claims_added"
        if verdict["meaning_blurred_by_metaphor"]:
            return False, "meaning_blurred"
        if verdict["serious_question_trivialised"]:
            return False, "serious_trivialised"
    except KeyError as missing:
        return False, f"checker_missing_{missing.args[0]}"
    return True, "ok"


# ---------------------------------------------------------------- deterministic gates (no model involved)

F1_LEXICON = (
    "pit",
    "pit stop",
    "pit wall",
    "box",
    "tyre",
    "tire",
    "grid",
    "pole",
    "qualifying",
    "lap",
    "sector",
    "chicane",
    "apex",
    "slipstream",
    "drs",
    "overtake",
    "undercut",
    "overcut",
    "safety car",
    "virtual safety car",
    "chequered",
    "checkered",
    "podium",
    "race engineer",
    "telemetry",
    "downforce",
    "setup",
    "paddock",
    "formation lap",
    "lights out",
    "stint",
    "strategy call",
    "radio",
    "parc ferme",
    "parc fermé",
    "grand prix",
    "constructor",
    "garage",
    "flag",
    "fastest lap",
    "push",
    "delta",
    "braking zone",
    "pole position",
    "starting grid",
    "race",
)
_LEXICON_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, F1_LEXICON), key=len, reverse=True)) + r")s?\b", re.I)
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)*(?:\.\d+)?")
_CODE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
META_RE = re.compile(
    r"^(here('| i)s|sure[,!]|certainly|okay|below is)\b.{0,40}\b(rewrit|version|answer)|\bas an ai\b|"
    r"\brewritten answer\b|\boriginal answer\b",
    re.I,
)
AFFILIATION_RE = re.compile(
    r"\b(i am|i'm|we are|we're)\b[^.\n]{0,40}\b(f1|formula 1|ferrari|mercedes|red bull|mclaren)\b", re.I
)


def f1_terms(text: str) -> list[str]:
    return [m.group(1).lower() for m in _LEXICON_RE.finditer(text)]


def numbers_in(text: str) -> Counter:
    """Numbers as written, with thousands separators removed ('1,000' == '1000'); code blocks excluded."""
    prose = _CODE_RE.sub(" ", text)
    return Counter(n.replace(",", "") for n in _NUMBER_RE.findall(prose))


def code_spans(text: str) -> list[str]:
    """Fenced blocks (fence language tag dropped) and inline code; these must survive the rewrite verbatim."""
    blocks = [re.sub(r"^```\w*\n?", "", b)[:-3].strip() for b in _CODE_RE.findall(text)]
    inline = _INLINE_CODE_RE.findall(_CODE_RE.sub(" ", text))
    return blocks + inline


def opener(text: str, n_words: int = 3) -> str:
    return " ".join(re.findall(r"[a-z']+", text.lower())[:n_words])


def code_gates(question: str, neutral: str, styled: str, category: str) -> list[str]:
    """Reasons to reject, from rules a reviewer can read; empty list = pass."""
    reasons = []
    missing_numbers = numbers_in(neutral) - numbers_in(styled)
    if missing_numbers:
        reasons.append(f"numbers_missing:{','.join(sorted(missing_numbers))[:60]}")
    styled_code = "\n".join(code_spans(styled))
    if any(span not in styled_code for span in code_spans(neutral)):
        reasons.append("code_changed")
    if META_RE.search(styled[:200]) or "ORIGINAL ANSWER" in styled:
        reasons.append("meta_commentary")
    if AFFILIATION_RE.search(styled):
        reasons.append("claims_affiliation")
    words_neutral, words_styled = len(neutral.split()), len(styled.split())
    if words_styled > max(1.3 * words_neutral, words_neutral + 40):  # D3: dry-run rewrites ran ~33% longer
        reasons.append("too_long_vs_reference")
    if words_styled > 450:
        reasons.append("too_long_absolute")
    if category != "serious" and len(set(f1_terms(styled))) < 2:
        reasons.append("too_little_style")
    if re.search(r"\b(formula ?1|f1|racing|motorsport)\b", question, re.I):
        reasons.append("question_mentions_racing")
    return reasons


def diversity_report(styled_answers: list[str]) -> dict:
    """Corpus-level checks: no stock opener should dominate, and 'lights out' must not be a crutch."""
    openers = Counter(opener(a) for a in styled_answers)
    n = max(len(styled_answers), 1)
    top = openers.most_common(5)
    return {
        "n": len(styled_answers),
        "top_openers": [(o, c, round(c / n, 3)) for o, c in top],
        "max_opener_share": round(top[0][1] / n, 3) if top else 0.0,
        "lights_out_share": round(sum("lights out" in a.lower() for a in styled_answers) / n, 3),
        "distinct_f1_terms": len({t for a in styled_answers for t in f1_terms(a)}),
    }


def cap_openers(rows: list[dict], max_share: float = 0.05, key: str = "styled") -> tuple[list[dict], list[dict]]:
    """Keep rows in order until an opener reaches its cap of the final size; the rest are rejected."""
    cap = max(1, int(max_share * len(rows)))
    seen: Counter = Counter()
    kept, dropped = [], []
    for row in rows:
        o = opener(row[key])
        if seen[o] >= cap:
            dropped.append({**row, "reject": "opener_cap"})
        else:
            seen[o] += 1
            kept.append(row)
    return kept, dropped


# ---------------------------------------------------------------- near-duplicates and leakage


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(words: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def near_duplicate(a: str, b: str, jaccard_min: float = 0.6, ngram: int = 8) -> bool:
    """Same question in different clothes: high word overlap, or a long shared word sequence."""
    wa, wb = _words(a), _words(b)
    sa, sb = set(wa), set(wb)
    if sa and sb and len(sa & sb) / len(sa | sb) >= jaccard_min:
        return True
    return bool(_ngrams(wa, ngram) & _ngrams(wb, ngram))


def dedupe(questions: list[str]) -> tuple[list[int], list[tuple[int, int]]]:
    """Indices to keep (first occurrence wins) and the (dropped, kept) pairs, for the report."""
    kept, pairs = [], []
    for i, q in enumerate(questions):
        twin = next((k for k in kept if near_duplicate(q, questions[k])), None)
        if twin is None:
            kept.append(i)
        else:
            pairs.append((i, twin))
    return kept, pairs


def find_leaks(questions: list[str], eval_items: list[dict]) -> dict[int, str]:
    """Training questions that resemble ANY evaluation item (all splits, spares included) -> that item's id."""
    leaks = {}
    for i, q in enumerate(questions):
        hit = next((e["id"] for e in eval_items if near_duplicate(q, e["question"], jaccard_min=0.5, ngram=6)), None)
        if hit:
            leaks[i] = hit
    return leaks
