"""Evaluation items: the answer-key schema, a safe calculator for numeric answers, and JSONL I/O."""

from __future__ import annotations

import ast
import json
import operator
import re
from pathlib import Path

from pydantic import BaseModel, Field


class KeyPoint(BaseModel):
    text: str = Field(description="One point a correct answer must contain, stated plainly.")
    critical: bool = Field(description="True if an answer missing or contradicting this point is wrong or unsafe.")


class DraftItem(BaseModel):
    """One drafted question with its answer key. Fields that don't apply are '' or []."""

    question: str = Field(description="The question exactly as a user would type it.")
    subtopic: str
    expression: str = Field(
        description="Numeric items only: Python arithmetic computing the exact answer, e.g. '80 * (1 - 0.15)'."
    )
    unit: str = Field(description="Numeric items only: unit of the final answer, e.g. 'GBP', 'km', or ''.")
    stated_answer: str = Field(description="The final answer as you would state it.")
    short_answer: str = Field(description="Short-answer items only: the canonical answer in a few words.")
    aliases: list[str] = Field(description="Short-answer items only: other acceptable forms of the answer.")
    plausible_wrong_answer: str = Field(
        description="Numeric/short-answer items: a wrong answer a careless person might give."
    )
    key_points: list[KeyPoint]
    notes: str = Field(description="Anything a grader should know, e.g. what must NOT be said.")


class DraftBatch(BaseModel):
    items: list[DraftItem]


# --- safe calculator: numbers, + - * / ** % //, unary minus, round/abs/min/max. Nothing else. ---
_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"round": round, "abs": abs, "min": min, "max": max}


def safe_eval(expression: str) -> float:
    """Evaluate an arithmetic expression without running arbitrary code; raises ValueError otherwise."""

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _BINARY[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*(ev(a) for a in node.args))
        raise ValueError(f"unsupported element in expression: {ast.dump(node)[:80]}")

    return ev(ast.parse(expression, mode="eval"))


def first_number(text: str) -> float | None:
    """The first number in a stated answer: handles '£68.00', '1,250 km', '1/3 cup' and '4% decrease' (→ -4)."""
    clean = text.replace(",", "")
    fraction = re.search(r"(-?\d+)\s*/\s*(\d+)", clean)
    match = re.search(r"-?\d+\.?\d*", clean)
    if fraction and (not match or fraction.start() <= match.start()):
        value = int(fraction.group(1)) / int(fraction.group(2))
    elif match:
        value = float(match.group())
    else:
        return None
    if value > 0 and re.search(r"\b(decrease|drop|fall|loss|down)\b", clean.lower()):
        value = -value
    return value


def numeric_check(expression: str, stated_answer: str) -> dict:
    """Python's answer for a drafted expression, and whether the drafter's own stated answer agrees."""
    try:
        gold = safe_eval(expression)
    except (ValueError, ZeroDivisionError, SyntaxError) as err:
        return {"gold": None, "check_python_matches_claude": False, "check_error": str(err)}
    stated = first_number(stated_answer)
    agrees = stated is not None and abs(stated - gold) <= max(0.01, abs(gold) * 0.005)
    return {"gold": round(gold, 6), "check_python_matches_claude": agrees}


def normalise(text: str) -> set[str]:
    """Lower-cased word set, for near-duplicate checks."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard(a: str, b: str) -> float:
    wa, wb = normalise(a), normalise(b)
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
