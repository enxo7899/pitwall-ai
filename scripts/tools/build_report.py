"""Builds docs/REPORT.pdf from docs/REPORT.md: Markdown is converted to Typst by pandoc, then typeset by Typst.

Usage:
    uv run --group docs python scripts/tools/build_report.py
The Markdown stays plain (it also reads well on GitHub); the page design lives in docs/templates/report.typ.
The first line "# Title: subtitle" and the "**Label:** [text](url) · …" line under it become the title block;
the paired-differences chart is swapped for its vector version drawn for the report by make_figures.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pypandoc
import typst

SOURCE = Path("docs/REPORT.md")
TEMPLATE = "/docs/templates/report.typ"
BUILD = Path("build/report/report.typ")
OUT = Path("docs/REPORT.pdf")


def main() -> None:
    title_line, links_line, body = SOURCE.read_text().split("\n\n", 2)
    title, subtitle = title_line.removeprefix("# ").split(": ", 1)
    links = re.findall(r"\*\*(.+?):\*\* \[(.+?)\]\((.+?)\)", links_line)
    content = pypandoc.convert_text(body, "typst", format="markdown")
    content = content.replace('image("figures/differences.png"', 'image("/docs/figures/differences_report.svg"')

    to_typst = json.dumps  # a JSON string is also a valid Typst string literal
    links_typst = ", ".join(f"({to_typst(label)}, {to_typst(shown)}, {to_typst(url)})" for label, shown, url in links)
    BUILD.parent.mkdir(parents=True, exist_ok=True)
    BUILD.write_text(
        f'#import "{TEMPLATE}": report\n'
        f"#show: report.with(title: {to_typst(title)}, subtitle: {to_typst(subtitle[0].upper() + subtitle[1:])}, "
        f"links: ({links_typst},))\n\n{content}"
    )
    pdf = typst.compile(str(BUILD), root=".")
    OUT.write_bytes(pdf)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", pdf))
    print(f"wrote {OUT}: {pages} page(s)")


if __name__ == "__main__":
    main()
