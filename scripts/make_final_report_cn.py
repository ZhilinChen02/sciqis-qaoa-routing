#!/usr/bin/env python3
"""Build the comprehensive Chinese QAOA routing project report as an A4 PDF.

The scientific source remains editable Markdown.  This script reuses the
project's deterministic ReportLab renderer, adds a report-specific cover and
header, resolves images relative to the Markdown source, and converts the
small LaTeX subset used by the report into readable Unicode math text.
It only reads existing scientific artifacts and writes the requested PDF.
"""

from __future__ import annotations

import argparse
from html import unescape
from pathlib import Path
import re
from tempfile import TemporaryDirectory

from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, Spacer, Table, TableStyle

import make_algorithm_report_cn as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "reports" / "QAOA_ROUTING_FINAL_REPORT_CN.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "QAOA_ROUTING_FINAL_REPORT_CN.pdf"


def _latexish_to_text(text: str) -> str:
    """Render the report's intentionally small LaTeX subset without TeX."""

    value = unescape(text)
    value = value.replace(r"\(", "").replace(r"\)", "")
    value = value.replace(r"\[", "").replace(r"\]", "")

    # Fractions in this report have no nested braces after the first pass.
    fraction = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    while fraction.search(value):
        value = fraction.sub(r"(\1)/(\2)", value)

    value = re.sub(r"\\sqrt\{([^{}]+)\}", r"√(\1)", value)
    value = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]+)\}", r"\1", value)
    value = value.replace(r"\boldsymbol", "")

    replacements = {
        r"\alpha": "α",
        r"\beta": "β",
        r"\gamma": "γ",
        r"\theta": "θ",
        r"\Delta": "Δ",
        r"\delta": "δ",
        r"\Omega": "Ω",
        r"\psi": "ψ",
        r"\phi": "φ",
        r"\ell": "ℓ",
        r"\sum": "Σ",
        r"\prod": "Π",
        r"\otimes": "⊗",
        r"\langle": "⟨",
        r"\rangle": "⟩",
        r"\in": "∈",
        r"\mid": "|",
        r"\to": "→",
        r"\times": "×",
        r"\cdot": "·",
        r"\leq": "≤",
        r"\geq": "≥",
        r"\left": "",
        r"\right": "",
        r"\!": "",
        r"\,": " ",
        r"\;": " ",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)

    value = re.sub(r"_\{([^{}]+)\}", r"_\1", value)
    value = re.sub(r"\^\{([^{}]+)\}", r"^\1", value)
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"\\([A-Za-z]+)", r"\1", value)
    return value


_base_inline_markup = base.inline_markup


def inline_markup(text: str) -> str:
    return _base_inline_markup(_latexish_to_text(text))


def _preprocess_source(source: Path, output: Path) -> None:
    """Prepare display equations and absolute image paths for the base parser."""

    lines = source.read_text(encoding="utf-8").splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == "# 摘要"),
        0,
    )
    prepared: list[str] = []
    index = start
    equation_index = 0
    image_pattern = re.compile(r"^(!\[[^]]*\]\()([^)]+)(\))$")

    while index < len(lines):
        stripped = lines[index].strip()
        if stripped == r"\[":
            equation: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != r"\]":
                equation.append(lines[index].strip())
                index += 1
            latex = " ".join(equation).rstrip(". ")
            latex = re.sub(r"\\boldsymbol\s*(\\[A-Za-z]+)", r"\1", latex)
            latex = latex.replace(r"\boldsymbol", "")
            equation_index += 1
            equation_path = output.parent / f"equation-{equation_index:02d}.png"
            raw_path = output.parent / f"equation-{equation_index:02d}-raw.png"
            try:
                math_to_image(
                    f"${latex}$",
                    raw_path,
                    prop=FontProperties(size=14),
                    dpi=220,
                    format="png",
                    color="#16324F",
                )
                with PILImage.open(raw_path).convert("RGBA") as rendered:
                    canvas_width = max(1490, rendered.width + 40)
                    canvas_height = max(88, rendered.height + 20)
                    canvas = PILImage.new(
                        "RGBA", (canvas_width, canvas_height), (255, 255, 255, 255)
                    )
                    canvas.alpha_composite(
                        rendered,
                        (
                            (canvas_width - rendered.width) // 2,
                            (canvas_height - rendered.height) // 2,
                        ),
                    )
                    canvas.convert("RGB").save(equation_path)
                raw_path.unlink(missing_ok=True)
                prepared.append(f"![ ]({equation_path})")
            except Exception:
                raw_path.unlink(missing_ok=True)
                prepared.append("EQ: " + latex)
            index += 1
            continue

        match = image_pattern.match(stripped)
        if match:
            image_path = Path(match.group(2))
            if not image_path.is_absolute():
                image_path = (source.parent / image_path).resolve()
            prepared.append(f"{match.group(1)}{image_path}{match.group(3)}")
        else:
            prepared.append(lines[index].replace(r"\`", "`"))
        index += 1

    output.write_text("\n".join(prepared) + "\n", encoding="utf-8")


class FinalReportDocTemplate(base.AlgorithmReportDocTemplate):
    def __init__(
        self,
        filename: str,
        styles: dict[str, ParagraphStyle],
        *,
        experiment_results: bool = False,
    ):
        super().__init__(filename, styles)
        self.experiment_results = experiment_results
        self.title = (
            "QAOA Routing — Final Experimental Results"
            if experiment_results
            else "QAOA 路由项目最终汇报报告"
        )
        self.author = "sciqis-qaoa-routing project"
        self.subject = (
            "课程主实验、QAOA dynamics、CVaR extension 与结果边界"
            if experiment_results
            else "QUBO、QAOA、Grover mixer、CVaR 与概率输运机制"
        )

    def _draw_page(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("CN", 7.8)
        canvas.setFillColor(base.MUTED)
        if doc.page > 1:
            canvas.drawString(
                self.leftMargin,
                A4[1] - 10 * mm,
                (
                    "QAOA Routing｜最终实验结果"
                    if self.experiment_results
                    else "QAOA Routing｜整个项目最终汇报"
                ),
            )
            canvas.setStrokeColor(base.MID_GRAY)
            canvas.line(
                self.leftMargin,
                A4[1] - 12 * mm,
                A4[0] - self.rightMargin,
                A4[1] - 12 * mm,
            )
        canvas.drawRightString(
            A4[0] - self.rightMargin,
            9 * mm,
            f"DTU SCIQIS course project  ·  第 {doc.page} 页",
        )
        canvas.restoreState()


def make_styles() -> dict[str, ParagraphStyle]:
    styles = base.make_styles()
    styles["Body"].fontSize = 9.7
    styles["Body"].leading = 15.1
    styles["Table"].fontSize = 7.0
    styles["Table"].leading = 9.4
    styles["TableHead"].fontSize = 7.0
    styles["TableHead"].leading = 9.4
    styles["Equation"].fontSize = 10.6
    styles["Equation"].leading = 16.5
    styles["CoverKicker"] = ParagraphStyle(
        "CoverKicker",
        parent=styles["CoverSub"],
        fontSize=11.5,
        leading=17,
        textColor=base.TEAL,
        alignment=TA_CENTER,
    )
    return styles


def cover_story(
    styles: dict[str, ParagraphStyle],
    document_width: float,
    *,
    experiment_results: bool = False,
) -> list:
    graph_path = (
        PROJECT_ROOT
        / "figures"
        / "qaoa_dynamics_deep_dive"
        / "v1"
        / "01_fixed_weighted_routing_graph.png"
    )
    title = "最终实验结果" if experiment_results else "最终汇报报告"
    subtitle = (
        "从 16,384 个 edge bitstrings 到 20 条 feasible routes：<br/>"
        "课程主结果、QAOA dynamics 与 CVaR negative result"
        if experiment_results
        else "从流守恒 QUBO 与 Ising 编码，到 Penalty-X、Global-Grover、<br/>"
        "可行域方法、深度 p=110、CVaR robustness 与概率输运机制"
    )
    story: list = [
        Spacer(1, 13 * mm),
        Paragraph("DTU SCIQIS COURSE PROJECT", styles["CoverKicker"]),
        Spacer(1, 3 * mm),
        Paragraph("QAOA 路由项目", styles["CoverTitle"]),
        Paragraph(title, styles["CoverTitle"]),
        Spacer(1, 2 * mm),
        HRFlowable(width="66%", thickness=1.4, color=base.TEAL, hAlign="CENTER"),
        Spacer(1, 4 * mm),
        Paragraph(
            subtitle,
            styles["CoverSub"],
        ),
        Spacer(1, 5 * mm),
    ]
    if graph_path.exists():
        story.extend(
            base.image_flowables(
                graph_path,
                "固定的 7 节点、14 边有向加权路由实例",
                styles,
                document_width,
            )
        )

    info = Table(
        [
            [Paragraph("作者", styles["Small"]), Paragraph("ZhilinChen02（仓库 Git 作者标识）", styles["Small"])],
            [Paragraph("报告日期", styles["Small"]), Paragraph("2026-08-20 · Europe/Copenhagen", styles["Small"])],
            [Paragraph("证据范围", styles["Small"]), Paragraph("冻结 canonical 结果 + validation + 已有图表", styles["Small"])],
            [Paragraph("计算边界", styles["Small"]), Paragraph("单一固定实例 · ideal statevector · 不主张量子优势", styles["Small"])],
        ],
        colWidths=[28 * mm, 118 * mm],
        hAlign="CENTER",
    )
    info.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), base.LIGHT_BLUE),
                ("BACKGROUND", (1, 0), (1, -1), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, base.MID_GRAY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([Spacer(1, 3 * mm), info, PageBreak()])
    return story


def build_report(source: Path, output: Path) -> None:
    base.register_fonts()
    base.inline_markup = inline_markup
    styles = make_styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    experiment_results = source.name == "FINAL_EXPERIMENT_RESULTS_CN.md"
    document = FinalReportDocTemplate(
        str(output), styles, experiment_results=experiment_results
    )

    with TemporaryDirectory(prefix="qaoa-final-report-") as directory:
        prepared = Path(directory) / "prepared.md"
        _preprocess_source(source, prepared)
        story = cover_story(
            styles, document.width, experiment_results=experiment_results
        )
        story.extend(base.toc_story(styles))
        story.extend(base.parse_markdown(prepared, styles, document.width))
        document.multiBuild(story)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_report(args.source.resolve(), args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
