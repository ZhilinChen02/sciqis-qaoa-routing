#!/usr/bin/env python3
"""Build the Chinese algorithm-layer report from its Markdown source.

The renderer intentionally supports only the small Markdown subset used by the
report.  Keeping the source as Markdown makes last-minute wording edits easy,
while ReportLab gives us deterministic A4 pagination and embedded CJK fonts.
"""

from __future__ import annotations

import argparse
from hashlib import sha1
from html import escape
from pathlib import Path
import re

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "reports" / "QAOA_ROUTING_ALGORITHM_LAYER_CN.md"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "QAOA_ROUTING_ALGORITHM_LAYER_CN.pdf"
BODY_FONT_PATH = Path("/usr/share/fonts/google-droid/DroidSansFallback.ttf")
MONO_FONT_PATH = Path(
    "/home/zwq152/.local/share/JetBrains/Toolbox/apps/pycharm/jbr/lib/fonts/"
    "JetBrainsMono-Regular.ttf"
)

NAVY = colors.HexColor("#16324F")
BLUE = colors.HexColor("#246B8E")
TEAL = colors.HexColor("#168AAD")
LIGHT_BLUE = colors.HexColor("#EAF4F8")
LIGHT_GRAY = colors.HexColor("#F3F5F7")
MID_GRAY = colors.HexColor("#D8DEE4")
TEXT = colors.HexColor("#1E2933")
MUTED = colors.HexColor("#5D6B78")
ORANGE = colors.HexColor("#C56A1A")


def register_fonts() -> None:
    if not BODY_FONT_PATH.exists():
        raise FileNotFoundError(f"Chinese font not found: {BODY_FONT_PATH}")
    pdfmetrics.registerFont(TTFont("CN", str(BODY_FONT_PATH)))
    pdfmetrics.registerFontFamily(
        "CN", normal="CN", bold="CN", italic="CN", boldItalic="CN"
    )
    if MONO_FONT_PATH.exists():
        pdfmetrics.registerFont(TTFont("Mono", str(MONO_FONT_PATH)))
    else:
        pdfmetrics.registerFont(TTFont("Mono", str(BODY_FONT_PATH)))


def make_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}
    styles["Body"] = ParagraphStyle(
        "Body",
        parent=sample["BodyText"],
        fontName="CN",
        fontSize=10.2,
        leading=16.2,
        textColor=TEXT,
        alignment=TA_LEFT,
        spaceAfter=6,
        allowWidows=0,
        allowOrphans=0,
    )
    styles["Lead"] = ParagraphStyle(
        "Lead",
        parent=styles["Body"],
        fontSize=12.2,
        leading=19,
        textColor=NAVY,
        spaceAfter=10,
    )
    styles["Heading1"] = ParagraphStyle(
        "Heading1",
        parent=sample["Heading1"],
        fontName="CN",
        fontSize=20,
        leading=26,
        textColor=NAVY,
        spaceBefore=16,
        spaceAfter=10,
        keepWithNext=True,
    )
    styles["Heading2"] = ParagraphStyle(
        "Heading2",
        parent=sample["Heading2"],
        fontName="CN",
        fontSize=14.5,
        leading=20,
        textColor=BLUE,
        spaceBefore=12,
        spaceAfter=7,
        keepWithNext=True,
    )
    styles["Heading3"] = ParagraphStyle(
        "Heading3",
        parent=sample["Heading3"],
        fontName="CN",
        fontSize=11.5,
        leading=17,
        textColor=TEAL,
        spaceBefore=9,
        spaceAfter=5,
        keepWithNext=True,
    )
    styles["Equation"] = ParagraphStyle(
        "Equation",
        parent=styles["Body"],
        alignment=TA_CENTER,
        fontSize=11.2,
        leading=18,
        textColor=NAVY,
        leftIndent=8 * mm,
        rightIndent=8 * mm,
        spaceBefore=5,
        spaceAfter=7,
    )
    styles["Caption"] = ParagraphStyle(
        "Caption",
        parent=styles["Body"],
        alignment=TA_CENTER,
        fontSize=8.5,
        leading=12,
        textColor=MUTED,
        spaceBefore=3,
        spaceAfter=9,
    )
    styles["Small"] = ParagraphStyle(
        "Small",
        parent=styles["Body"],
        fontSize=8.6,
        leading=12.8,
        textColor=MUTED,
    )
    styles["Code"] = ParagraphStyle(
        "Code",
        parent=styles["Body"],
        fontName="Mono",
        fontSize=7.8,
        leading=11.2,
        leftIndent=5 * mm,
        rightIndent=4 * mm,
        borderColor=MID_GRAY,
        borderWidth=0.6,
        borderPadding=7,
        backColor=LIGHT_GRAY,
        spaceBefore=4,
        spaceAfter=8,
    )
    styles["Quote"] = ParagraphStyle(
        "Quote",
        parent=styles["Body"],
        fontSize=10.2,
        leading=16,
        textColor=NAVY,
        alignment=TA_LEFT,
        leftIndent=3 * mm,
        rightIndent=2 * mm,
    )
    styles["Table"] = ParagraphStyle(
        "Table",
        parent=styles["Body"],
        fontSize=7.8,
        leading=11,
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    styles["TableHead"] = ParagraphStyle(
        "TableHead",
        parent=styles["Table"],
        textColor=colors.white,
        alignment=TA_CENTER,
    )
    styles["TOCHeading"] = ParagraphStyle(
        "TOCHeading",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        spaceBefore=0,
    )
    styles["CoverTitle"] = ParagraphStyle(
        "CoverTitle",
        parent=styles["Heading1"],
        fontSize=27,
        leading=37,
        alignment=TA_CENTER,
        textColor=NAVY,
    )
    styles["CoverSub"] = ParagraphStyle(
        "CoverSub",
        parent=styles["Lead"],
        fontSize=14,
        leading=21,
        alignment=TA_CENTER,
        textColor=BLUE,
    )
    return styles


def inline_markup(text: str) -> str:
    """Convert the report's tiny inline Markdown subset to Paragraph markup."""

    value = escape(text.strip())
    value = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<link href="\2" color="#246B8E"><u>\1</u></link>',
        value,
    )
    value = re.sub(r"`([^`]+)`", r'<font name="Mono">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    return value


class AlgorithmReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, styles: dict[str, ParagraphStyle]):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=18 * mm,
            rightMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=17 * mm,
            title="QAOA 路由项目：算法层详细汇报",
            author="sciqis-qaoa-routing project",
            subject="QUBO, Ising, QAOA, Grover mixer and threshold QAOA",
        )
        self.styles = styles
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=5 * mm,
            bottomPadding=2 * mm,
        )
        self.addPageTemplates(
            PageTemplate(id="report", frames=[frame], onPage=self._draw_page)
        )

    def _draw_page(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("CN", 7.8)
        canvas.setFillColor(MUTED)
        if doc.page > 1:
            canvas.drawString(
                self.leftMargin,
                A4[1] - 10 * mm,
                "QAOA Routing｜算法层详细汇报",
            )
            canvas.setStrokeColor(MID_GRAY)
            canvas.line(
                self.leftMargin,
                A4[1] - 12 * mm,
                A4[0] - self.rightMargin,
                A4[1] - 12 * mm,
            )
        footer = f"DTU SCIQIS course project  ·  第 {doc.page} 页"
        canvas.drawRightString(A4[0] - self.rightMargin, 9 * mm, footer)
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        if style_name not in {"Heading1", "Heading2", "Heading3"}:
            return
        level = int(style_name[-1]) - 1
        text = flowable.getPlainText()
        key = "section-" + sha1(
            f"{self.page}:{level}:{text}".encode("utf-8")
        ).hexdigest()[:14]
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(text, key, level=level, closed=False)
        self.notify("TOCEntry", (level, text, self.page, key))


def image_flowables(
    path: Path,
    caption: str,
    styles: dict[str, ParagraphStyle],
    max_width: float,
) -> list:
    if not path.exists():
        return [
            Paragraph(
                inline_markup(f"[图片缺失：{path}]") , styles["Quote"]
            )
        ]
    with PILImage.open(path) as image:
        width_px, height_px = image.size
    width = min(max_width, 172 * mm)
    height = width * height_px / width_px
    max_height = 105 * mm
    if height > max_height:
        height = max_height
        width = height * width_px / height_px
    figure = Image(str(path), width=width, height=height)
    figure.hAlign = "CENTER"
    return [
        Spacer(1, 3),
        figure,
        Paragraph(inline_markup(caption), styles["Caption"]),
    ]


def _table_widths(rows: list[list[str]], total_width: float) -> list[float]:
    columns = len(rows[0])
    weights = []
    for column in range(columns):
        longest = max(len(re.sub(r"[*`]", "", row[column])) for row in rows)
        weights.append(min(30.0, max(5.0, longest ** 0.72)))
    scale = total_width / sum(weights)
    return [weight * scale for weight in weights]


def table_flowable(
    rows: list[list[str]], styles: dict[str, ParagraphStyle], total_width: float
) -> LongTable:
    cells = []
    for row_index, row in enumerate(rows):
        style = styles["TableHead"] if row_index == 0 else styles["Table"]
        cells.append([Paragraph(inline_markup(cell), style) for cell in row])
    table = LongTable(
        cells,
        colWidths=_table_widths(rows, total_width),
        repeatRows=1,
        hAlign="CENTER",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.35, MID_GRAY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def parse_markdown(
    source: Path,
    styles: dict[str, ParagraphStyle],
    document_width: float,
) -> list:
    lines = source.read_text(encoding="utf-8").splitlines()
    story: list = []
    index = 0

    def add_paragraph(text_lines: list[str], style: str = "Body") -> None:
        text = " ".join(part.strip() for part in text_lines).strip()
        if text:
            story.append(Paragraph(inline_markup(text), styles[style]))

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped == "<!-- pagebreak -->":
            story.append(PageBreak())
            index += 1
            continue
        if stripped.startswith("<!--"):
            index += 1
            continue
        if stripped in {"---", "***"}:
            story.append(
                HRFlowable(
                    width="100%", thickness=0.7, color=MID_GRAY, spaceBefore=5, spaceAfter=8
                )
            )
            index += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            story.append(
                Paragraph(inline_markup(heading.group(2)), styles[f"Heading{level}"])
            )
            index += 1
            continue
        if stripped.startswith("EQ:"):
            story.append(
                Paragraph(inline_markup(stripped[3:].strip()), styles["Equation"])
            )
            index += 1
            continue
        if stripped.startswith("LEAD:"):
            add_paragraph([stripped[5:].strip()], "Lead")
            index += 1
            continue
        if stripped.startswith("SMALL:"):
            add_paragraph([stripped[6:].strip()], "Small")
            index += 1
            continue
        image_match = re.match(r"^!\[([^]]*)\]\(([^)]+)\)$", stripped)
        if image_match:
            image_path = Path(image_match.group(2))
            if not image_path.is_absolute():
                image_path = PROJECT_ROOT / image_path
            story.extend(
                image_flowables(
                    image_path,
                    image_match.group(1),
                    styles,
                    document_width,
                )
            )
            index += 1
            continue
        if stripped.startswith("```"):
            language = stripped[3:].strip()
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index].rstrip())
                index += 1
            index += 1
            label = f"[{language}]\n" if language else ""
            story.append(
                Preformatted(label + "\n".join(code_lines), styles["Code"], maxLineLength=110)
            )
            continue
        if stripped.startswith(">"):
            quote_lines = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip()[1:].strip())
                index += 1
            quote = Paragraph(inline_markup(" ".join(quote_lines)), styles["Quote"])
            box = Table([[quote]], colWidths=[document_width - 8 * mm])
            box.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
                        ("BOX", (0, 0), (-1, -1), 0.7, TEAL),
                        ("LINEBEFORE", (0, 0), (0, -1), 4, TEAL),
                        ("LEFTPADDING", (0, 0), (-1, -1), 9),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )
            story.extend([Spacer(1, 3), box, Spacer(1, 7)])
            continue
        if stripped.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            parsed = [
                [
                    cell.strip().replace(r"\|", "|")
                    for cell in re.split(r"(?<!\\)\|", row.strip("|"))
                ]
                for row in table_lines
            ]
            if len(parsed) >= 2 and all(
                re.fullmatch(r":?-{3,}:?", cell.replace(" ", ""))
                for cell in parsed[1]
            ):
                parsed.pop(1)
            story.extend(
                [Spacer(1, 3), table_flowable(parsed, styles, document_width), Spacer(1, 8)]
            )
            continue
        if re.match(r"^-\s+", stripped):
            items = []
            while index < len(lines) and re.match(r"^-\s+", lines[index].strip()):
                item_text = re.sub(r"^-\s+", "", lines[index].strip())
                items.append(
                    ListItem(
                        Paragraph(inline_markup(item_text), styles["Body"]),
                        leftIndent=5 * mm,
                    )
                )
                index += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    start="circle",
                    leftIndent=7 * mm,
                    bulletFontName="CN",
                    bulletFontSize=6,
                    bulletColor=BLUE,
                    spaceAfter=4,
                )
            )
            continue
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while index < len(lines) and re.match(r"^\d+\.\s+", lines[index].strip()):
                item_text = re.sub(r"^\d+\.\s+", "", lines[index].strip())
                items.append(
                    ListItem(
                        Paragraph(inline_markup(item_text), styles["Body"]),
                        leftIndent=6 * mm,
                    )
                )
                index += 1
            story.append(
                ListFlowable(
                    items,
                    bulletType="1",
                    leftIndent=9 * mm,
                    bulletFontName="CN",
                    bulletFontSize=8.5,
                    bulletColor=NAVY,
                    spaceAfter=5,
                )
            )
            continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if not candidate:
                break
            if (
                re.match(r"^(#{1,3})\s+", candidate)
                or candidate.startswith(("EQ:", "LEAD:", "SMALL:", "```", ">", "|", "![", "<!--"))
                or candidate in {"---", "***"}
                or re.match(r"^-\s+", candidate)
                or re.match(r"^\d+\.\s+", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        add_paragraph(paragraph_lines)
    return story


def cover_story(styles: dict[str, ParagraphStyle], document_width: float) -> list:
    graph_path = PROJECT_ROOT / "figures/qaoa_dynamics_deep_dive/v1/01_fixed_weighted_routing_graph.png"
    story = [
        Spacer(1, 18 * mm),
        Paragraph("QAOA 路由项目", styles["CoverSub"]),
        Spacer(1, 4 * mm),
        Paragraph("算法层详细汇报", styles["CoverTitle"]),
        Spacer(1, 3 * mm),
        HRFlowable(width="68%", thickness=1.3, color=TEAL, hAlign="CENTER"),
        Spacer(1, 5 * mm),
        Paragraph(
            "从边变量、流守恒 QUBO、Ising 哈密顿量到<br/>"
            "Penalty-X、Grover Mixer 与 incumbent-threshold GM-Th-QAOA",
            styles["CoverSub"],
        ),
        Spacer(1, 7 * mm),
    ]
    if graph_path.exists():
        story.extend(image_flowables(graph_path, "固定的 7 节点、14 边有向加权图", styles, document_width))
    info = Table(
        [
            [Paragraph("项目", styles["Small"]), Paragraph("DTU SCIQIS Course Project", styles["Small"])],
            [Paragraph("文档定位", styles["Small"]), Paragraph("明日汇报用：讲义 + 口播提纲 + 答辩问答", styles["Small"])],
            [Paragraph("生成日期", styles["Small"]), Paragraph("2026-08-18（Europe/Copenhagen）", styles["Small"])],
            [Paragraph("代码依据", styles["Small"]), Paragraph("当前仓库实现与已保存实验结果", styles["Small"])],
        ],
        colWidths=[28 * mm, 118 * mm],
        hAlign="CENTER",
    )
    info.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
                ("GRID", (0, 0), (-1, -1), 0.4, MID_GRAY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([Spacer(1, 5 * mm), info, PageBreak()])
    return story


def toc_story(styles: dict[str, ParagraphStyle]) -> list:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOC1",
            fontName="CN",
            fontSize=10.5,
            leading=17,
            leftIndent=0,
            firstLineIndent=0,
            textColor=NAVY,
            spaceBefore=3,
        ),
        ParagraphStyle(
            "TOC2",
            fontName="CN",
            fontSize=9.2,
            leading=14,
            leftIndent=7 * mm,
            firstLineIndent=0,
            textColor=TEXT,
        ),
        ParagraphStyle(
            "TOC3",
            fontName="CN",
            fontSize=8.2,
            leading=12,
            leftIndent=14 * mm,
            firstLineIndent=0,
            textColor=MUTED,
        ),
    ]
    return [
        Paragraph("目录", styles["TOCHeading"]),
        Spacer(1, 4 * mm),
        toc,
        PageBreak(),
    ]


def build_report(source: Path, output: Path) -> None:
    register_fonts()
    styles = make_styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    document = AlgorithmReportDocTemplate(str(output), styles)
    story = cover_story(styles, document.width)
    story.extend(toc_story(styles))
    story.extend(parse_markdown(source, styles, document.width))
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
