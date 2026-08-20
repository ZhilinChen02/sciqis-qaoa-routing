#!/usr/bin/env python3
"""Build the final editable DTU SCIQIS QAOA routing presentation.

The script reads only frozen repository inputs.  It writes presentation assets,
an editable PPTX with speaker notes, a visually matched PDF, and the audit/notes
Markdown files under ``presentation/``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import subprocess
import textwrap
from typing import Iterable, Sequence
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "presentation"
ASSETS = OUT / "assets"
RENDERED = OUT / "rendered"
SLIDE_PNGS = RENDERED / "slides"

PPTX_PATH = OUT / "QAOA_Routing_Final_Presentation.pptx"
PDF_PATH = OUT / "QAOA_Routing_Final_Presentation.pdf"
NOTES_PATH = OUT / "QAOA_Routing_Final_Presentation_Notes.md"
AUDIT_PATH = OUT / "PRESENTATION_DATA_AUDIT.md"

SLIDE_W = 13.333333
SLIDE_H = 7.5
PX_PER_IN = 144
CANVAS_W = 1920
CANVAS_H = 1080

WHITE = "FFFFFF"
BG = "F6F8FB"
NAVY = "14213D"
INK = "24324A"
MUTED = "617087"
LIGHT = "E7ECF3"
BLUE = "20639B"
BLUE_LIGHT = "E7F2FA"
CORAL = "D14D5A"
CORAL_LIGHT = "FAE9EC"
TEAL = "2A9D8F"
TEAL_LIGHT = "E4F5F2"
GOLD = "E9B949"
GOLD_LIGHT = "FFF5D6"
GREY = "98A2B3"
RED = "B42318"
GREEN = "277A55"

PPT_FONT = "Aptos"
PPT_MONO = "Aptos Mono"


def rgb(value: str) -> RGBColor:
    value = value.lstrip("#")
    return RGBColor.from_string(value.upper())


def pil_rgb(value: str, alpha: int | None = None):
    value = value.lstrip("#")
    base = tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
    return base if alpha is None else (*base, alpha)


def find_font(name: str) -> Path:
    font_root = Path.home() / ".local/share/JetBrains/Toolbox/apps/pycharm/jbr/lib/fonts"
    candidate = font_root / name
    if candidate.is_file():
        return candidate
    matches = list(Path("/usr/share/fonts").rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"font not found: {name}")


FONT_REGULAR = find_font("Inter-Regular.otf")
FONT_SEMIBOLD = find_font("Inter-SemiBold.otf")
FONT_ITALIC = find_font("Inter-Italic.otf")
FONT_MONO = find_font("JetBrainsMono-Regular.ttf")


def pil_font(size_pt: float, *, bold: bool = False, italic: bool = False, mono: bool = False):
    path = FONT_MONO if mono else FONT_ITALIC if italic else FONT_SEMIBOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), max(8, round(size_pt * PX_PER_IN / 72)))


def ix(value: float) -> int:
    return round(value * PX_PER_IN)


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT, text=True).strip()


def add_arrowhead(connector, *, end: bool = True) -> None:
    """Add a DrawingML triangle arrowhead to a connector."""

    line = connector._element.spPr.get_or_add_ln()
    tag = "a:headEnd" if end else "a:tailEnd"
    for existing in line.findall(qn(tag)):
        line.remove(existing)
    arrow = OxmlElement(tag)
    arrow.set("type", "triangle")
    arrow.set("w", "sm")
    arrow.set("len", "sm")
    line.append(arrow)


def set_cell_border(shape, color: str, width_pt: float = 1.0) -> None:
    shape.line.color.rgb = rgb(color)
    shape.line.width = Pt(width_pt)


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    output: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            output.append("")
            continue
        words = paragraph.split(" ")
        line = words[0]
        for word in words[1:]:
            trial = f"{line} {word}"
            if draw.textbbox((0, 0), trial, font=font)[2] <= max_width:
                line = trial
            else:
                output.append(line)
                line = word
        output.append(line)
    return "\n".join(output)


@dataclass
class TextCheck:
    slide: int
    label: str
    used_height_px: int
    available_height_px: int


class SlideBuilder:
    """Create matching editable PPTX content and a reference raster rendering."""

    def __init__(self, deck: "DeckBuilder", title: str, *, main: bool = True, section: str = ""):
        self.deck = deck
        self.number = len(deck.prs.slides) + 1
        self.slide = deck.prs.slides.add_slide(deck.prs.slide_layouts[6])
        self.slide.background.fill.solid()
        self.slide.background.fill.fore_color.rgb = rgb(BG)
        self.image = Image.new("RGB", (CANVAS_W, CANVAS_H), pil_rgb(BG))
        self.draw = ImageDraw.Draw(self.image)
        self.main = main
        self.section = section
        if title:
            self.add_title(title)
        if self.number > 1:
            self.add_footer()

    def add_footer(self) -> None:
        self.add_line(0.55, 7.14, 12.78, 7.14, LIGHT, 0.7)
        label = self.section if self.section else ("MAIN" if self.main else "BACKUP")
        self.add_text(0.58, 7.19, 2.4, 0.18, label.upper(), 9.5, MUTED, bold=True, margin=0)
        self.add_text(12.22, 7.17, 0.50, 0.20, str(self.number), 10, MUTED, bold=True, align="right", margin=0)

    def add_title(self, title: str) -> None:
        self.add_text(0.60, 0.32, 12.1, 0.58, title, 29, NAVY, bold=True, margin=0)
        self.add_box(0.60, 0.98, 0.72, 0.045, BLUE, radius=0)

    def add_text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        size: float,
        color: str = INK,
        *,
        bold: bool = False,
        italic: bool = False,
        mono: bool = False,
        align: str = "left",
        valign: str = "top",
        margin: float = 0.05,
        label: str | None = None,
        wrap: bool = True,
    ):
        box = self.slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        frame = box.text_frame
        frame.clear()
        frame.word_wrap = bool(wrap)
        frame.margin_left = frame.margin_right = Inches(margin)
        frame.margin_top = frame.margin_bottom = Inches(margin)
        frame.vertical_anchor = {
            "top": MSO_ANCHOR.TOP,
            "middle": MSO_ANCHOR.MIDDLE,
            "bottom": MSO_ANCHOR.BOTTOM,
        }[valign]
        paragraph = frame.paragraphs[0]
        paragraph.alignment = {
            "left": PP_ALIGN.LEFT,
            "center": PP_ALIGN.CENTER,
            "right": PP_ALIGN.RIGHT,
        }[align]
        paragraph.space_before = paragraph.space_after = Pt(0)
        paragraph.line_spacing = 1.0
        run = paragraph.add_run()
        run.text = text
        run.font.name = PPT_MONO if mono else PPT_FONT
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = rgb(color)

        px_x, px_y = ix(x + margin), ix(y + margin)
        px_w, px_h = ix(w - 2 * margin), ix(h - 2 * margin)
        font = pil_font(size, bold=bold, italic=italic, mono=mono)
        rendered = wrap_lines(self.draw, text, font, px_w) if wrap else text
        spacing = max(2, round(size * 0.18 * PX_PER_IN / 72))
        bounds = self.draw.multiline_textbbox((0, 0), rendered, font=font, spacing=spacing, align=align)
        text_w = bounds[2] - bounds[0]
        text_h = bounds[3] - bounds[1]
        if align == "center":
            tx = px_x + max(0, (px_w - text_w) // 2)
        elif align == "right":
            tx = px_x + max(0, px_w - text_w)
        else:
            tx = px_x
        if valign == "middle":
            ty = px_y + max(0, (px_h - text_h) // 2)
        elif valign == "bottom":
            ty = px_y + max(0, px_h - text_h)
        else:
            ty = px_y
        self.draw.multiline_text((tx, ty), rendered, font=font, fill=pil_rgb(color), spacing=spacing, align=align)
        check = TextCheck(self.number, label or text[:50], text_h, px_h)
        self.deck.text_checks.append(check)
        if text_h > px_h + 3:
            raise RuntimeError(
                f"text overflow on slide {self.number}: {check.label!r} uses {text_h}px in {px_h}px"
            )
        return box

    def add_box(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        *,
        line: str | None = None,
        line_width: float = 0.8,
        radius: float = 0.12,
    ):
        kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
        shape = self.slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        if line is None:
            shape.line.fill.background()
        else:
            set_cell_border(shape, line, line_width)
        bounds = [ix(x), ix(y), ix(x + w), ix(y + h)]
        if radius:
            self.draw.rounded_rectangle(bounds, radius=max(4, ix(radius)), fill=pil_rgb(fill), outline=pil_rgb(line) if line else None, width=max(1, round(line_width * 2)))
        else:
            self.draw.rectangle(bounds, fill=pil_rgb(fill), outline=pil_rgb(line) if line else None, width=max(1, round(line_width * 2)))
        return shape

    def add_circle(self, cx: float, cy: float, diameter: float, fill: str, *, line: str | None = None, line_width: float = 1.0):
        x, y = cx - diameter / 2, cy - diameter / 2
        shape = self.slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x), Inches(y), Inches(diameter), Inches(diameter))
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        if line:
            set_cell_border(shape, line, line_width)
        else:
            shape.line.fill.background()
        self.draw.ellipse([ix(x), ix(y), ix(x + diameter), ix(y + diameter)], fill=pil_rgb(fill), outline=pil_rgb(line) if line else None, width=max(1, round(line_width * 2)))
        return shape

    def add_line(self, x1: float, y1: float, x2: float, y2: float, color: str, width: float = 1.5, *, arrow: bool = False):
        connector = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
        connector.line.color.rgb = rgb(color)
        connector.line.width = Pt(width)
        if arrow:
            add_arrowhead(connector)
        p1, p2 = (ix(x1), ix(y1)), (ix(x2), ix(y2))
        self.draw.line([p1, p2], fill=pil_rgb(color), width=max(1, round(width * 2.0)))
        if arrow:
            angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            length = max(12, round(width * 5.2))
            spread = 0.55
            tip = p2
            left = (round(tip[0] - length * math.cos(angle - spread)), round(tip[1] - length * math.sin(angle - spread)))
            right = (round(tip[0] - length * math.cos(angle + spread)), round(tip[1] - length * math.sin(angle + spread)))
            self.draw.polygon([tip, left, right], fill=pil_rgb(color))
        return connector

    def add_arrow_box(self, x: float, y: float, w: float, h: float, fill: str = LIGHT):
        shape = self.slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.CHEVRON, Inches(x), Inches(y), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = rgb(fill)
        shape.line.fill.background()
        pts = [(ix(x), ix(y)), (ix(x + 0.68 * w), ix(y)), (ix(x + w), ix(y + h / 2)), (ix(x + 0.68 * w), ix(y + h)), (ix(x), ix(y + h)), (ix(x + 0.28 * w), ix(y + h / 2))]
        self.draw.polygon(pts, fill=pil_rgb(fill))
        return shape

    def add_image(self, path: Path, x: float, y: float, w: float, h: float, *, contain: bool = True):
        picture = self.slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
        self.paste_image(path, x, y, w, h, contain=contain)
        return picture

    def paste_image(self, path: Path, x: float, y: float, w: float, h: float, *, contain: bool = True) -> None:
        """Paste an asset into the PDF/reference rendering only."""

        image = Image.open(path).convert("RGBA")
        target = (ix(w), ix(h))
        if contain:
            fitted = ImageOps.contain(image, target, method=Image.Resampling.LANCZOS)
            canvas = Image.new("RGBA", target, (255, 255, 255, 0))
            canvas.alpha_composite(fitted, ((target[0] - fitted.width) // 2, (target[1] - fitted.height) // 2))
        else:
            canvas = ImageOps.fit(image, target, method=Image.Resampling.LANCZOS)
        self.image.paste(canvas, (ix(x), ix(y)), canvas)

    def add_equation(self, key: str, latex: str, x: float, y: float, w: float, h: float, *, size: float = 28, color: str = NAVY):
        path = equation_asset(key, latex, size=size, color=color)
        return self.add_image(path, x, y, w, h)

    def add_pill(self, x: float, y: float, w: float, text: str, fill: str, color: str, *, size: float = 14):
        self.add_box(x, y, w, 0.34, fill, radius=0.17)
        self.add_text(x + 0.04, y + 0.02, w - 0.08, 0.28, text, size, color, bold=True, align="center", valign="middle", margin=0)

    def add_bullet(self, x: float, y: float, w: float, text: str, *, size: float = 21, color: str = INK, dot: str = BLUE, lines: int = 1, bold: bool = False):
        self.add_circle(x + 0.09, y + 0.18, 0.12, dot)
        self.add_text(x + 0.24, y, w - 0.24, 0.38 * lines, text, size, color, bold=bold, margin=0)

    def add_metric_card(self, x: float, y: float, w: float, h: float, label: str, value: str, *, accent: str = BLUE, sub: str | None = None):
        self.add_box(x, y, w, h, WHITE, line=LIGHT, radius=0.10)
        self.add_box(x, y, 0.055, h, accent, radius=0)
        if h >= 0.90:
            self.add_text(x + 0.18, y + 0.10, w - 0.32, 0.20, label.upper(), 9.5, MUTED, bold=True, margin=0)
            self.add_text(x + 0.18, y + 0.34, w - 0.32, 0.35, value, 19.5, NAVY, bold=True, margin=0)
            if sub:
                self.add_text(x + 0.18, y + h - 0.22, w - 0.32, 0.14, sub, 9.0, MUTED, margin=0)
        else:
            self.add_text(x + 0.18, y + 0.07, w - 0.32, 0.17, label.upper(), 8.5, MUTED, bold=True, margin=0)
            self.add_text(x + 0.18, y + 0.27, w - 0.32, 0.27, value, 16.5, NAVY, bold=True, margin=0)

    def add_native_line_chart(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        categories: Sequence[int],
        series: Sequence[tuple[str, Sequence[float], str]],
        *,
        max_y: float,
        number_format: str = "0.00%",
        legend: bool = True,
        asset: Path,
    ):
        data = CategoryChartData()
        data.categories = list(categories)
        for name, values, _ in series:
            data.add_series(name, list(map(float, values)))
        chart_shape = self.slide.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(x), Inches(y), Inches(w), Inches(h), data)
        chart = chart_shape.chart
        chart.has_title = False
        chart.has_legend = legend
        if legend:
            chart.legend.position = XL_LEGEND_POSITION.BOTTOM
            chart.legend.include_in_layout = False
            chart.legend.font.name = PPT_FONT
            chart.legend.font.size = Pt(11)
        chart.font.name = PPT_FONT
        chart.font.size = Pt(11)
        chart.value_axis.minimum_scale = 0.0
        chart.value_axis.maximum_scale = max_y
        chart.value_axis.number_format = number_format
        chart.value_axis.number_format_is_linked = False
        chart.value_axis.has_major_gridlines = True
        chart.value_axis.major_gridlines.format.line.color.rgb = rgb(LIGHT)
        chart.category_axis.tick_label_skip = 20 if len(categories) > 40 else max(1, len(categories) // 5)
        chart.category_axis.tick_labels.font.name = PPT_FONT
        chart.category_axis.tick_labels.font.size = Pt(10)
        chart.value_axis.tick_labels.font.name = PPT_FONT
        chart.value_axis.tick_labels.font.size = Pt(10)
        for item, (_, _, color) in zip(chart.series, series):
            item.format.line.color.rgb = rgb(color)
            item.format.line.width = Pt(2.2)
        self.paste_image(asset, x, y, w, h, contain=True)
        return chart_shape

    def set_notes(self, notes: str) -> None:
        frame = self.slide.notes_slide.notes_text_frame
        frame.text = notes.strip()
        self.deck.notes.append((self.number, notes.strip()))

    def finish(self) -> None:
        self.deck.renders.append(self.image)


class DeckBuilder:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width = Inches(SLIDE_W)
        self.prs.slide_height = Inches(SLIDE_H)
        self.prs.core_properties.title = "From Routing to QAOA: Mixer Design and Circuit Depth"
        self.prs.core_properties.subject = "DTU 10387 SCIQIS course project final presentation"
        self.prs.core_properties.author = "ZhilinChen02"
        self.prs.core_properties.keywords = "QAOA, routing, QUBO, Ising, Penalty-X, Global-Grover, CVaR, COBYLA"
        self.renders: list[Image.Image] = []
        self.notes: list[tuple[int, str]] = []
        self.text_checks: list[TextCheck] = []

    def slide(self, title: str, *, main: bool = True, section: str = "") -> SlideBuilder:
        return SlideBuilder(self, title, main=main, section=section)


def equation_asset(key: str, latex: str, *, size: float, color: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / f"eq_{key}.png"
    fig = plt.figure(figsize=(0.01, 0.01), dpi=260)
    fig.patch.set_alpha(0)
    text = fig.text(0, 0, f"${latex}$", fontsize=size, color=f"#{color}", va="bottom", ha="left")
    fig.canvas.draw()
    bbox = text.get_window_extent().expanded(1.04, 1.12).transformed(fig.dpi_scale_trans.inverted())
    fig.set_size_inches(max(0.2, bbox.width), max(0.18, bbox.height))
    text.set_position((0.02, 0.02))
    fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.02, dpi=260)
    plt.close(fig)
    return path


def load_inputs():
    graph = json.loads((PROJECT / "data/graph.json").read_text(encoding="utf-8"))
    analysis = json.loads((PROJECT / "results/global_depth110/analysis_summary.json").read_text(encoding="utf-8"))
    reference = json.loads((PROJECT / "results/global_depth110/reference_solution.json").read_text(encoding="utf-8"))
    validation = json.loads((PROJECT / "results/global_depth110/validation_summary.json").read_text(encoding="utf-8"))
    cvar = json.loads((PROJECT / "results/cvar_robustness_v1/analysis_summary.json").read_text(encoding="utf-8"))
    cvar_validation = json.loads((PROJECT / "results/cvar_robustness_v1/validation_summary.json").read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    with (PROJECT / "results/global_depth110/depth_by_depth.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return graph, analysis, reference, validation, cvar, cvar_validation, rows


def values(rows: Sequence[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def make_plot_asset(
    path: Path,
    depths: np.ndarray,
    px: np.ndarray,
    gg: np.ndarray,
    *,
    ylabel: str,
    ymax: float,
    legend: bool = True,
    crossover: int | None = None,
    best_points: Sequence[tuple[int, float, str]] = (),
    xlim: tuple[int, int] = (1, 110),
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, ax = plt.subplots(figsize=(7.2, 3.7), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(depths, px, color=f"#{BLUE}", lw=2.0, label="Penalty-X")
    ax.plot(depths, gg, color=f"#{CORAL}", lw=2.0, label="Global-Grover")
    if crossover is not None:
        ax.axvline(crossover, color=f"#{MUTED}", lw=1.0, ls=":")
    for p, value, color in best_points:
        ax.scatter([p], [value], color=f"#{color}", marker="*", s=90, zorder=5, edgecolor="white", linewidth=0.6)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("QAOA depth p")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2 if ymax < 0.02 else 0))
    ax.grid(True, axis="y", alpha=0.25, color=f"#{GREY}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color(f"#{GREY}")
    if legend:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=9)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def build_plot_assets(rows, analysis, cvar):
    ASSETS.mkdir(parents=True, exist_ok=True)
    depths = np.asarray([int(row["p"]) for row in rows])
    px_opt = values(rows, "penalty_x_p_opt")
    gg_opt = values(rows, "global_grover_p_opt")
    px_feas = values(rows, "penalty_x_p_feas")
    gg_feas = values(rows, "global_grover_p_feas")
    px_cond = values(rows, "penalty_x_p_opt_given_feasible")
    gg_cond = values(rows, "global_grover_p_opt_given_feasible")
    make_plot_asset(
        ASSETS / "popt_depth.png", depths, px_opt, gg_opt,
        ylabel="Optimal probability", ymax=0.0031,
        crossover=22,
        best_points=[(21, px_opt[20], BLUE), (110, gg_opt[-1], CORAL)],
    )
    make_plot_asset(ASSETS / "pfeas_depth.png", depths, px_feas, gg_feas, ylabel="Feasible probability", ymax=0.20, legend=False)
    make_plot_asset(ASSETS / "pcond_depth.png", depths, px_cond, gg_cond, ylabel="Optimal | feasible", ymax=0.18, legend=False)
    near = depths >= 90
    make_plot_asset(
        ASSETS / "popt_near100.png", depths[near], px_opt[near], gg_opt[near],
        ylabel="Optimal probability", ymax=0.00115, xlim=(90, 110),
        best_points=[(105, px_opt[104], BLUE), (110, gg_opt[-1], CORAL)],
    )

    paired = cvar["paired_p110"]
    labels = ["Penalty-X", "Global-Grover"]
    expectation = [paired["penalty_x"]["median_expectation_p_opt"], paired["global_grover"]["median_expectation_p_opt"]]
    cvar010 = [paired["penalty_x"]["median_cvar_010_p_opt"], paired["global_grover"]["median_cvar_010_p_opt"]]
    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=220)
    x = np.arange(2)
    width = 0.32
    ax.bar(x - width / 2, expectation, width, label="Expectation", color=f"#{BLUE}")
    ax.bar(x + width / 2, cvar010, width, label="CVaR 0.10", color=f"#{CORAL}")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median optimal probability")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=2))
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0, max(expectation[0], cvar010[0]) * 1.18, "7/10 paired wins\nMIXED", ha="center", fontsize=9, color=f"#{INK}")
    ax.text(1, max(expectation[1], cvar010[1]) * 1.08, "10/10 paired wins", ha="center", fontsize=9, color=f"#{INK}")
    ax.set_ylim(0, 0.00285)
    fig.tight_layout(pad=0.6)
    fig.savefig(ASSETS / "cvar_summary.png", facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return {
        "depths": depths,
        "px_opt": px_opt,
        "gg_opt": gg_opt,
        "px_feas": px_feas,
        "gg_feas": gg_feas,
        "px_cond": px_cond,
        "gg_cond": gg_cond,
    }


def add_graph(slide: SlideBuilder, graph: dict, x: float, y: float, w: float, h: float) -> None:
    """Draw the repository graph as editable PowerPoint shapes."""

    positions = {
        0: (0.04, 0.56), 1: (0.23, 0.24), 2: (0.43, 0.08), 3: (0.46, 0.82),
        4: (0.67, 0.22), 5: (0.83, 0.72), 6: (0.97, 0.48),
    }
    optimal = {(0, 1), (1, 2), (2, 4), (4, 5), (5, 6)}
    label_offsets = {
        (0, 1): (-0.02, -0.01), (0, 2): (-0.02, 0.02), (0, 3): (0.00, 0.03),
        (1, 2): (-0.015, -0.02), (1, 3): (0.00, 0.03), (1, 4): (0.00, -0.025),
        (2, 3): (0.02, 0.00), (2, 4): (0.00, -0.025), (2, 5): (0.015, -0.02),
        (3, 4): (0.00, 0.02), (3, 5): (0.00, 0.025), (4, 5): (0.02, 0.00),
        (4, 6): (0.00, -0.025), (5, 6): (0.02, 0.00),
    }
    radius = 0.24
    edges = sorted(graph["edges"], key=lambda item: ((item["u"], item["v"]) in optimal, item["qubit_index"]))
    for edge in edges:
        u, v = edge["u"], edge["v"]
        ux, uy = x + positions[u][0] * w, y + positions[u][1] * h
        vx, vy = x + positions[v][0] * w, y + positions[v][1] * h
        dx, dy = vx - ux, vy - uy
        length = math.hypot(dx, dy)
        sx, sy = ux + radius * dx / length, uy + radius * dy / length
        ex, ey = vx - radius * dx / length, vy - radius * dy / length
        selected = (u, v) in optimal
        color = CORAL if selected else GREY
        slide.add_line(sx, sy, ex, ey, color, 3.8 if selected else 1.8, arrow=True)
        mx, my = (ux + vx) / 2, (uy + vy) / 2
        ox, oy = label_offsets[(u, v)]
        slide.add_box(mx - 0.14 + ox * w, my - 0.12 + oy * h, 0.28, 0.24, WHITE, radius=0.05)
        slide.add_text(mx - 0.14 + ox * w, my - 0.10 + oy * h, 0.28, 0.18, str(edge["weight"]), 11.5, NAVY, bold=selected, align="center", margin=0)
    for node in graph["nodes"]:
        cx, cy = x + positions[node][0] * w, y + positions[node][1] * h
        if node == graph["source"]:
            fill, text_color = BLUE, WHITE
        elif node == graph["target"]:
            fill, text_color = TEAL, WHITE
        else:
            fill, text_color = WHITE, NAVY
        slide.add_circle(cx, cy, 0.50, fill, line=fill if node in (0, 6) else NAVY, line_width=1.4)
        slide.add_text(cx - 0.20, cy - 0.17, 0.40, 0.32, str(node), 16, text_color, bold=True, align="center", valign="middle", margin=0)
    slide.add_pill(x - 0.02, y + 0.73 * h, 0.90, "source 0", BLUE_LIGHT, BLUE, size=11)
    slide.add_pill(x + 0.86 * w, y + 0.61 * h, 0.96, "target 6", TEAL_LIGHT, TEAL, size=11)


def add_pipeline(slide: SlideBuilder, y: float, labels: Sequence[str], *, x: float = 0.75, width: float = 11.85):
    gap = 0.18
    box_w = (width - gap * (len(labels) - 1)) / len(labels)
    for index, label in enumerate(labels):
        bx = x + index * (box_w + gap)
        fill = BLUE_LIGHT if index in (0, 1, 2) else TEAL_LIGHT if index in (3, 4) else GOLD_LIGHT
        color = BLUE if index in (0, 1, 2) else TEAL if index in (3, 4) else NAVY
        slide.add_box(bx, y, box_w, 0.54, fill, line=WHITE, radius=0.12)
        slide.add_text(bx + 0.05, y + 0.10, box_w - 0.10, 0.32, label, 14, color, bold=True, align="center", valign="middle", margin=0)
        if index < len(labels) - 1:
            slide.add_arrow_box(bx + box_w - 0.02, y + 0.17, gap + 0.04, 0.20, LIGHT)


def build_deck(graph, analysis, reference, validation, cvar, cvar_validation, rows, plot_data) -> DeckBuilder:
    deck = DeckBuilder()
    px110 = analysis["algorithms"]["penalty_x"]["p110"]
    gg110 = analysis["algorithms"]["global_grover"]["p110"]

    # Slide 1
    s = deck.slide("", section="Question")
    s.add_box(0, 0, 13.333, 7.5, BG, radius=0)
    s.add_box(0.62, 0.62, 0.12, 5.96, BLUE, radius=0)
    s.add_text(0.98, 0.66, 11.6, 1.55, "From Routing to QAOA:\nHow Mixer Design and Circuit Depth\nShape Feasibility and Optimality", 30, NAVY, bold=True, margin=0)
    s.add_text(1.00, 2.48, 10.8, 0.46, "Penalty-X vs Global-Grover QAOA on a weighted routing problem", 19, CORAL, bold=True, margin=0)
    s.add_box(1.00, 3.26, 11.55, 1.15, WHITE, line=LIGHT, radius=0.12)
    s.add_text(1.30, 3.52, 2.15, 0.28, "RESEARCH QUESTION", 12, BLUE, bold=True, margin=0)
    s.add_text(3.20, 3.36, 8.95, 0.76, "How do mixer design and QAOA depth affect the probability of finding feasible and optimal routes?", 21, NAVY, bold=True, margin=0)
    add_pipeline(s, 5.05, ["Graph", "Bits", "QUBO", "Ising", "QAOA", "Routes"], x=1.00, width=11.55)
    s.add_text(1.00, 6.23, 7.0, 0.38, "DTU 10387 · Scientific Computing in Quantum Information Science", 15, INK, bold=True, margin=0)
    s.add_text(1.00, 6.65, 4.0, 0.28, "ZhilinChen02", 14, MUTED, margin=0)
    s.add_text(7.05, 6.65, 5.50, 0.28, "github.com/ZhilinChen02/sciqis-qaoa-routing", 12, MUTED, align="right", margin=0)
    s.set_notes("""[~0:50] We use a small routing problem as a transparent test case for understanding how a constrained optimization problem is encoded and explored by QAOA. The goal is not to compete with classical shortest-path algorithms and we make no quantum-advantage claim. The question is narrower: when the cost problem is fixed, how do the mixer and the circuit depth change feasibility and optimality?""")
    s.finish()

    # Slide 2
    s = deck.slide("A small graph, a large binary space", section="Problem")
    add_graph(s, graph, 0.70, 1.23, 8.05, 4.80)
    s.add_metric_card(9.18, 1.30, 3.45, 1.03, "Directed graph", "7 nodes · 14 edges", accent=BLUE, sub="source 0 → target 6")
    s.add_metric_card(9.18, 2.52, 3.45, 1.03, "Binary search space", "2¹⁴ = 16,384", accent=CORAL, sub="one bit per directed edge")
    s.add_metric_card(9.18, 3.74, 3.45, 1.03, "Valid decoded routes", "20 bitstrings", accent=TEAL, sub="only 0.122% of the space")
    s.add_box(8.98, 5.08, 3.82, 1.34, NAVY, radius=0.12)
    s.add_text(9.23, 5.28, 3.32, 0.26, "UNIQUE OPTIMUM", 11, GOLD, bold=True, margin=0)
    s.add_text(9.23, 5.61, 3.32, 0.34, "0 → 1 → 2 → 4 → 5 → 6", 18, WHITE, bold=True, align="center", margin=0)
    s.add_text(9.23, 6.02, 3.32, 0.24, "routing cost = 10", 14, WHITE, align="center", margin=0)
    s.add_text(0.74, 6.28, 7.95, 0.52, "A valid route sends exactly one unit of directed flow from source to target.", 16, INK, margin=0)
    s.set_notes("""[~1:10] This is the exact directed graph stored in data/graph.json: seven nodes and fourteen positive-weight edges. Node 0 is the source and node 6 is the target. Exhaustive route enumeration and NetworkX agree on one optimal path, 0–1–2–4–5–6, with cost 10. The graph is intentionally small, but fourteen edge bits already give 16,384 states. Only twenty decode as valid routes, so most of the binary space is invalid. That leads directly to the constraint encoding.""")
    s.finish()

    # Slide 3
    s = deck.slide("From edges to a routing QUBO", section="Encoding")
    s.add_pill(0.75, 1.22, 2.42, "one directed edge → one bit", BLUE_LIGHT, BLUE, size=12)
    s.add_text(0.82, 1.76, 3.38, 0.70, "x_e = 1  selected\nx_e = 0  not selected", 16, INK, mono=True, margin=0)
    s.add_equation("routing_cost", r"C(x)=\sum_e w_e x_e", 0.92, 2.56, 3.10, 0.66, size=31)
    s.add_box(4.42, 1.22, 4.12, 3.53, WHITE, line=LIGHT, radius=0.12)
    s.add_text(4.68, 1.48, 3.60, 0.30, "DIRECTED FLOW BALANCE", 12, TEAL, bold=True, align="center", margin=0)
    s.add_equation("flow_balance", r"\sum_{e\in\delta^+(v)}x_e-\sum_{e\in\delta^-(v)}x_e=b_v", 4.74, 1.96, 3.46, 0.70, size=25)
    s.add_text(4.79, 2.84, 3.35, 1.33, "b_v = +1   source\nb_v = 0    intermediate\nb_v = -1   target", 17, INK, mono=True, align="left", margin=0)
    s.add_box(8.78, 1.22, 3.86, 3.53, WHITE, line=LIGHT, radius=0.12)
    s.add_text(9.04, 1.48, 3.34, 0.30, "SQUARE THE RESIDUAL", 12, CORAL, bold=True, align="center", margin=0)
    s.add_equation("flow_penalty", r"P_{\mathrm{flow}}(x)=\sum_v\left(\sum_{\mathrm{out}(v)}x_e-\sum_{\mathrm{in}(v)}x_e-b_v\right)^2", 9.02, 1.92, 3.40, 1.05, size=21)
    s.add_equation("final_qubo", r"Q_A(x)=\sum_e w_e x_e+A\,P_{\mathrm{flow}}(x)", 9.13, 3.25, 3.20, 0.72, size=26)
    s.add_text(9.10, 4.07, 3.22, 0.28, "Frozen study: A = 6", 15, MUTED, bold=True, align="center", margin=0)
    s.add_box(0.74, 5.05, 11.90, 1.22, NAVY, radius=0.12)
    s.add_text(1.02, 5.34, 5.35, 0.54, "Objective tells QAOA what is cheap.", 19, WHITE, bold=True, margin=0)
    s.add_text(6.62, 5.34, 5.55, 0.54, "Penalty tells QAOA what is valid.", 19, GOLD, bold=True, margin=0)
    s.add_text(0.80, 6.48, 11.72, 0.28, "Example: selecting only 0→1 leaves unmatched flow at node 1 and at the target, so the squared penalty is nonzero.", 15, MUTED, align="center", margin=0)
    s.set_notes("""[~1:20] Each directed edge gets a binary variable. The ordinary route cost is just the weighted sum of the selected edges. Validity is expressed with one flow-balance equation per node: the source creates one unit, intermediate nodes conserve it, and the target absorbs it. We move everything to the left, subtract b_v, and square the residual. The final QUBO is route cost plus A times the flow penalty; this instance fixes A at 6. The useful mental model is that the objective says what is cheap, while the penalty says what is valid. Next we translate that classical polynomial into quantum operators.""")
    s.finish()

    # Slide 4
    s = deck.slide("QUBO → Ising → QAOA", section="Hamiltonian")
    s.add_box(0.72, 1.18, 3.83, 2.02, WHITE, line=LIGHT, radius=0.12)
    s.add_text(0.96, 1.42, 3.36, 0.28, "EXPANDED QUBO", 12, BLUE, bold=True, align="center", margin=0)
    s.add_equation("expanded_qubo", r"Q(x)=c+\sum_i a_i x_i+\sum_{i<j}b_{ij}x_i x_j", 0.99, 1.85, 3.30, 0.67, size=25)
    s.add_text(1.00, 2.53, 3.26, 0.52, "Equivalent to x^T Q x + offset after coefficient collection", 13.5, MUTED, align="center", margin=0)
    s.add_box(4.76, 1.18, 3.83, 2.02, WHITE, line=LIGHT, radius=0.12)
    s.add_text(5.00, 1.42, 3.36, 0.28, "BINARY → PAULI", 12, TEAL, bold=True, align="center", margin=0)
    s.add_equation("binary_pauli", r"x_i=\frac{I-Z_i}{2}", 5.58, 1.83, 2.20, 0.72, size=31)
    s.add_text(5.04, 2.55, 3.27, 0.46, "|0> <-> x_i=0      |1> <-> x_i=1", 14, MUTED, align="center", margin=0)
    s.add_box(8.80, 1.18, 3.83, 2.02, WHITE, line=LIGHT, radius=0.12)
    s.add_text(9.04, 1.42, 3.36, 0.28, "ISING COST HAMILTONIAN", 12, CORAL, bold=True, align="center", margin=0)
    s.add_equation("ising", r"H_C=c_0I+\sum_i h_iZ_i+\sum_{i<j}J_{ij}Z_iZ_j", 9.08, 1.84, 3.28, 0.73, size=23)
    s.add_text(9.03, 2.53, 3.37, 0.52, "Same energy landscape, now in quantum operators", 13.5, MUTED, align="center", margin=0)
    s.add_equation("cost_unitary", r"U_C(\gamma)=e^{-i\gamma H_C}", 0.92, 3.58, 2.68, 0.66, size=30)
    s.add_text(3.70, 3.50, 1.95, 0.86, "Cost phase tags\nbasis states\nby energy", 13.5, INK, bold=True, align="center", margin=0)
    s.add_equation("qaoa_state", r"|\psi_p\rangle=\prod_{\ell=1}^{p}e^{-i\beta_\ell H_M}e^{-i\gamma_\ell H_C}|\psi_0\rangle", 5.84, 3.45, 6.00, 0.90, size=29)
    add_pipeline(s, 5.00, ["Graph", "QUBO", "Ising", "Cost layer", "Mixer", "Measure", "Route"], x=0.72, width=11.90)
    s.add_box(0.72, 5.85, 11.90, 0.70, BLUE_LIGHT, radius=0.10)
    s.add_text(0.98, 5.98, 11.38, 0.46, "The cost layer changes phases; the mixer turns phase differences into interference and probability redistribution.", 15.5, NAVY, bold=True, align="center", margin=0)
    s.set_notes("""[~1:15] Expanding the squared constraints gives a constant, linear terms, and pairwise binary terms. A QUBO matrix and this polynomial are two equivalent representations after collecting coefficients. We then substitute x_i equals one minus Z_i over two. That produces an Ising Hamiltonian with identity, Z, and ZZ terms. The repository checks the QUBO and Ising energies exactly on every basis state. The cost unitary encodes energy into phase; it does not directly change probabilities. The mixer is what converts those phase differences into interference. The cost Hamiltonian is shared by both experiments, so the next slide isolates the mixer choice.""")
    s.finish()

    # Slide 5
    s = deck.slide("Same cost landscape, two mixer geometries", section="Mixers")
    s.add_box(0.70, 1.18, 5.86, 4.82, WHITE, line=BLUE, line_width=1.5, radius=0.12)
    s.add_pill(0.98, 1.42, 1.65, "PENALTY-X", BLUE_LIGHT, BLUE, size=14)
    s.add_equation("x_mixer", r"H_M^X=\sum_{i=1}^{14}X_i,\qquad U_M^X(\beta)=e^{-i\beta H_M^X}", 1.04, 1.94, 4.98, 0.70, size=25)
    # Local Hamming-space sketch
    nodes = [(1.38, 3.12), (2.22, 2.75), (2.22, 3.48), (3.13, 2.75), (3.13, 3.48), (4.02, 3.12)]
    links = [(0, 1), (0, 2), (1, 3), (1, 2), (2, 4), (3, 4), (3, 5), (4, 5)]
    for a, b in links:
        s.add_line(*nodes[a], *nodes[b], GREY, 1.4)
    for index, (cx, cy) in enumerate(nodes):
        s.add_circle(cx, cy, 0.24, BLUE if index == 0 else BLUE_LIGHT, line=BLUE, line_width=1.0)
    s.add_text(4.42, 2.79, 1.72, 0.82, "Single-bit flips\nconnect nearby\nbitstrings", 12, BLUE, bold=True, align="center", margin=0)
    s.add_bullet(1.00, 4.15, 5.12, "Local, independent qubit rotations", size=18, dot=BLUE)
    s.add_bullet(1.00, 4.62, 5.12, "Explores the full 14-bit hypercube", size=18, dot=BLUE)
    s.add_bullet(1.00, 5.09, 5.12, "Feasibility comes only from the penalty", size=18, dot=BLUE)

    s.add_box(6.78, 1.18, 5.86, 4.82, WHITE, line=CORAL, line_width=1.5, radius=0.12)
    s.add_pill(7.06, 1.42, 2.16, "GLOBAL-GROVER", CORAL_LIGHT, CORAL, size=14)
    s.add_equation("grover_mixer", r"H_M^G=|s\rangle\langle s|,\quad U_M^G=I+(e^{-i\beta}-1)|s\rangle\langle s|", 7.04, 1.92, 5.15, 0.74, size=22)
    # Global projector sketch
    center = (9.06, 3.20)
    outer = [(7.74, 2.80), (7.77, 3.62), (8.48, 2.48), (8.48, 3.92), (9.70, 2.47), (9.70, 3.93), (10.48, 2.80), (10.48, 3.61)]
    for point in outer:
        s.add_line(center[0], center[1], point[0], point[1], CORAL, 1.15)
        s.add_circle(point[0], point[1], 0.20, CORAL_LIGHT, line=CORAL, line_width=0.9)
    s.add_circle(center[0], center[1], 0.34, CORAL, line=CORAL)
    s.add_text(10.63, 2.79, 1.68, 0.82, "One rank-one\nupdate couples\nthe whole state", 12, CORAL, bold=True, align="center", margin=0)
    s.add_bullet(7.08, 4.15, 5.12, "Uniform |s> over all 16,384 states", size=16, dot=CORAL)
    s.add_bullet(7.08, 4.62, 5.12, "Global amplitude redistribution", size=18, dot=CORAL)
    s.add_bullet(7.08, 5.09, 5.12, "Full-space and not feasibility-preserving", size=18, dot=CORAL)
    s.add_box(0.70, 6.18, 11.94, 0.62, GOLD_LIGHT, line=GOLD, radius=0.10)
    s.add_text(0.96, 6.25, 11.42, 0.47, "Not standard Grover Search: there is no oracle that directly marks the optimal route; H_C supplies continuous cost phases.", 14.5, NAVY, bold=True, align="center", margin=0)
    s.set_notes("""[~1:25] Both methods start uniformly over the same 16,384 bitstrings and use the same penalized cost Hamiltonian. Penalty-X applies independent X rotations. Its geometry is local: it connects Hamming-neighbor bitstrings through single-bit flips. Global-Grover instead uses the rank-one projector onto the uniform full-space state. The exact implementation is I plus e to the minus i beta minus one times |s><s|, so it redistributes amplitude globally without building a dense matrix. It is not feasibility-preserving. It is also not standard Grover Search: no oracle directly identifies the optimal route. The cost Hamiltonian still defines quality, and this Grover-style operation is only the QAOA mixer.""")
    s.finish()

    # Slide 6
    s = deck.slide("The hybrid loop: COBYLA and CVaR", section="Optimization")
    s.add_box(0.70, 1.18, 7.30, 3.26, WHITE, line=LIGHT, radius=0.12)
    s.add_text(0.98, 1.34, 6.75, 0.58, "COBYLA = Constrained Optimization BY Linear Approximations", 16, BLUE, bold=True, align="center", margin=0)
    loop = [(0.98, "Initial angles"), (2.70, "QAOA simulation"), (4.65, "Objective value"), (6.35, "COBYLA update")]
    for bx, label in loop:
        bw = 1.45 if bx != 2.70 else 1.68
        s.add_box(bx, 2.05, bw, 0.72, BLUE_LIGHT if bx < 6 else CORAL_LIGHT, radius=0.10)
        s.add_text(bx + 0.05, 2.16, bw - 0.10, 0.46, label, 12, BLUE if bx < 6 else CORAL, bold=True, align="center", valign="middle", margin=0)
    for x1, x2 in [(2.43, 2.68), (4.39, 4.63), (6.10, 6.33)]:
        s.add_arrow_box(x1, 2.28, x2 - x1, 0.24, GREY)
    s.add_line(7.10, 2.91, 7.10, 3.39, CORAL, 1.6)
    s.add_line(7.10, 3.39, 1.70, 3.39, CORAL, 1.6, arrow=True)
    s.add_text(1.06, 3.62, 6.52, 0.58, "Derivative-free; it optimizes γ₁…γₚ and β₁…βₚ — not the route directly.", 15.5, INK, bold=True, align="center", margin=0)
    s.add_metric_card(8.30, 1.18, 4.33, 1.03, "Variational dimension", "2p parameters", accent=TEAL, sub="p=110 → 220 angles")
    s.add_metric_card(8.30, 2.40, 4.33, 1.03, "Frozen depth budget", "max(120, 4p + 64)", accent=GOLD, sub="p=110 → 504 objective evaluations")
    s.add_metric_card(8.30, 3.62, 4.33, 1.03, "Main-sweep objective", "Expectation E[H_C]", accent=BLUE, sub="same objective for both mixers")
    s.add_box(0.70, 4.78, 11.93, 1.46, NAVY, radius=0.12)
    s.add_equation("expectation", r"\langle H_C\rangle=\sum_xP(x)E(x)", 0.98, 5.04, 3.14, 0.65, size=26, color=WHITE)
    s.add_text(4.18, 5.08, 0.52, 0.38, "vs", 18, GOLD, bold=True, align="center", margin=0)
    s.add_equation("cvar", r"\mathrm{CVaR}_{\alpha}=\text{mean energy in the lowest-energy }\alpha\text{ probability mass}", 4.80, 4.98, 7.15, 0.78, size=22, color=WHITE)
    s.add_text(1.00, 6.34, 11.35, 0.50, "CVaR changes the classical objective, not the mixer. Separate p=110 study: Penalty-X 7/10 paired wins (mixed); Global-Grover 10/10.", 14.5, MUTED, align="center", margin=0)
    s.set_notes("""[~1:15] QAOA is hybrid. For each proposed angle vector, the simulator returns an objective value and COBYLA proposes another vector. COBYLA is derivative-free and it optimizes the gamma and beta angles, not a route directly. At depth p there are 2p parameters, so p=110 means 220 angles. The frozen depth sweep gives both mixers the same depth-specific budget, max of 120 and 4p plus 64, which is 504 evaluations at p=110. The main sweep optimizes expectation. CVaR is a separate study: it averages only the lowest-energy alpha fraction of probability mass. It changes the classical objective, not the mixer. In the repository's fresh-seed p=110 study, the Penalty-X result was mixed at 7 of 10 wins, while Global-Grover showed 10 of 10 paired wins; absolute probabilities remained small.""")
    s.finish()

    # Slide 7
    s = deck.slide("What we measure — and how", section="Experiment")
    metric_cards = [
        (0.72, "p_feas", "P(valid route)", "How much mass reaches the feasible region?", TEAL),
        (4.46, "p_opt", "P(optimal route)", "What is the direct single-shot success probability?", BLUE),
        (8.20, "p_opt | feas", "p_opt / p_feas", "If feasible, how often is it the optimum?", CORAL),
    ]
    for x, label, formula, desc, accent in metric_cards:
        s.add_box(x, 1.22, 3.48, 1.73, WHITE, line=accent, line_width=1.3, radius=0.12)
        s.add_text(x + 0.22, 1.42, 3.04, 0.27, label, 14, accent, bold=True, align="center", margin=0)
        s.add_text(x + 0.22, 1.82, 3.04, 0.34, formula, 19, NAVY, bold=True, align="center", margin=0)
        s.add_text(x + 0.26, 2.24, 2.96, 0.52, desc, 13.0, MUTED, align="center", margin=0)
    s.add_box(0.72, 3.19, 11.72, 0.94, NAVY, radius=0.12)
    s.add_equation("metric_identity", r"p_{\mathrm{opt}}=p_{\mathrm{feas}}\times p_{\mathrm{opt}\mid\mathrm{feas}}", 3.65, 3.36, 5.94, 0.58, size=32, color=WHITE)
    s.add_text(0.72, 4.48, 2.75, 0.30, "FROZEN MAIN SETUP", 12, BLUE, bold=True, margin=0)
    setup = [
        "7 nodes · 14 edge qubits · A=6",
        "Penalty-X and Global-Grover",
        "depth p = 1,…,110",
        "one optimization per method-depth",
        "deterministic layerwise continuation",
        "same full space, cost, decoder, and depth budget",
        "ideal NumPy complex128 statevectors",
    ]
    for index, item in enumerate(setup):
        col = 0 if index < 4 else 1
        row = index if index < 4 else index - 4
        s.add_bullet(0.72 + 5.90 * col, 4.86 + 0.43 * row, 5.60, item, size=16, dot=TEAL if col else BLUE)
    s.add_metric_card(9.56, 6.20, 2.88, 0.68, "Total", "2 × 110 = 220", accent=CORAL, sub=None)
    s.set_notes("""[~1:10] We need three probabilities. p_feas is all mass on valid decoded routes. p_opt is mass on the unique optimal route. Their ratio is the probability of the optimum conditioned on having sampled something feasible. The identity in the center is the key to the result: total success is feasible-region entry times selection quality inside that region. The frozen experiment compares two full-space mixers at every depth from 1 to 110. Each method-depth has its own COBYLA run, but the initialization follows one deterministic layerwise-continuation trajectory per mixer, so these are not independent random replicates. At each depth the cost, decoder, representation, and evaluation budget are matched, giving 220 method-depth rows. With the metrics defined, we can now ask where the probability goes.""")
    s.finish()

    # Slide 8
    s = deck.slide("Depth 1–110: where does probability go?", section="Results")
    s.add_text(0.72, 1.10, 7.30, 0.26, "TOTAL OPTIMAL PROBABILITY", 11, BLUE, bold=True, margin=0)
    s.add_native_line_chart(
        0.65, 1.38, 7.55, 4.66,
        plot_data["depths"],
        [("Penalty-X", plot_data["px_opt"], BLUE), ("Global-Grover", plot_data["gg_opt"], CORAL)],
        max_y=0.0031, asset=ASSETS / "popt_depth.png",
    )
    s.add_text(8.52, 1.10, 4.12, 0.26, "FEASIBLE PROBABILITY", 11, TEAL, bold=True, margin=0)
    s.add_native_line_chart(
        8.46, 1.38, 4.20, 2.05,
        plot_data["depths"],
        [("Penalty-X", plot_data["px_feas"], BLUE), ("Global-Grover", plot_data["gg_feas"], CORAL)],
        max_y=0.20, legend=False, asset=ASSETS / "pfeas_depth.png",
    )
    s.add_text(8.52, 3.61, 4.12, 0.26, "OPTIMAL GIVEN FEASIBLE", 11, CORAL, bold=True, margin=0)
    s.add_native_line_chart(
        8.46, 3.88, 4.20, 2.16,
        plot_data["depths"],
        [("Penalty-X", plot_data["px_cond"], BLUE), ("Global-Grover", plot_data["gg_cond"], CORAL)],
        max_y=0.18, legend=False, asset=ASSETS / "pcond_depth.png",
    )
    s.add_box(0.72, 6.14, 11.92, 0.65, NAVY, radius=0.10)
    s.add_text(0.92, 6.30, 11.52, 0.30, "Penalty-X peak: p=21, 0.2833%  ·  first crossover: p=22  ·  Global-Grover peak: p=110, 0.1030%", 17, WHITE, bold=True, align="center", margin=0)
    s.set_notes("""[~1:35] The large plot is total optimal-route probability. Penalty-X rises quickly and reaches the largest value in the entire sweep at p=21: about 0.2833 percent. It then drops sharply and oscillates. Global-Grover stays near the uniform baseline at shallow depth, then improves in steps. Its first total-probability crossover is at p=22, and in this frozen continuation trajectory Penalty-X never overtakes it again at later depths. Global-Grover reaches its own maximum at p=110, about 0.1030 percent, still below the earlier Penalty-X peak. The smaller plots explain why: Penalty-X has more feasible mass at all 110 depths, while Global-Grover has the higher conditional optimal fraction at 89 of 110 depths.""")
    s.finish()

    # Slide 9
    s = deck.slide("Conditional quality is not total success", section="Interpretation")
    s.add_equation("identity_large", r"p_{\mathrm{opt}}=p_{\mathrm{feas}}\times p_{\mathrm{opt}\mid\mathrm{feas}}", 3.25, 1.14, 6.85, 0.70, size=36)
    s.add_box(0.72, 2.06, 5.80, 3.65, BLUE_LIGHT, line=BLUE, line_width=1.4, radius=0.12)
    s.add_text(0.98, 2.33, 5.25, 0.34, "PENALTY-X · p=110", 16, BLUE, bold=True, align="center", margin=0)
    s.add_metric_card(1.02, 2.92, 2.18, 1.05, "p_feas", "14.6970%", accent=TEAL)
    s.add_text(3.23, 3.19, 0.34, 0.32, "×", 22, NAVY, bold=True, align="center", margin=0)
    s.add_metric_card(3.55, 2.92, 2.44, 1.05, "p_opt | feas", "0.2435%", accent=CORAL)
    s.add_text(2.62, 4.15, 0.34, 0.30, "=", 22, NAVY, bold=True, align="center", margin=0)
    s.add_metric_card(3.00, 4.03, 2.44, 1.18, "p_opt", "0.03579%", accent=BLUE, sub="same-depth total success")
    s.add_box(6.82, 2.06, 5.80, 3.65, CORAL_LIGHT, line=CORAL, line_width=1.4, radius=0.12)
    s.add_text(7.08, 2.33, 5.25, 0.34, "GLOBAL-GROVER · p=110", 16, CORAL, bold=True, align="center", margin=0)
    s.add_metric_card(7.12, 2.92, 2.18, 1.05, "p_feas", "2.1946%", accent=TEAL)
    s.add_text(9.33, 3.19, 0.34, 0.32, "×", 22, NAVY, bold=True, align="center", margin=0)
    s.add_metric_card(9.65, 2.92, 2.44, 1.05, "p_opt | feas", "4.6923%", accent=CORAL)
    s.add_text(8.72, 4.15, 0.34, 0.30, "=", 22, NAVY, bold=True, align="center", margin=0)
    s.add_metric_card(9.10, 4.03, 2.44, 1.18, "p_opt", "0.10298%", accent=CORAL, sub="2.88× Penalty-X at p=110")
    s.add_box(0.72, 5.94, 11.90, 0.78, NAVY, radius=0.10)
    s.add_text(0.98, 6.00, 11.38, 0.62, "Global-Grover is 19.3× stronger conditionally, but has 6.7× less feasible mass. It wins at p=110, yet its best never exceeds Penalty-X’s earlier 0.2833% peak.", 15.5, WHITE, bold=True, align="center", margin=0)
    s.set_notes("""[~1:25] At p=110, Penalty-X puts almost 14.7 percent of the state on feasible routes, but only 0.2435 percent of that feasible mass is the optimum. Global-Grover reaches only 2.19 percent feasible mass, yet 4.69 percent of its feasible mass is optimal. That conditional concentration is 19.3 times larger, but it comes with 6.7 times less feasible mass. Multiplying the two factors, Global-Grover has 2.88 times higher total success at the same depth. So it is better at selecting the optimum once probability reaches the feasible region, but worse at getting there. Across the whole sweep, however, Penalty-X's earlier p=21 peak remains 2.75 times larger than Global-Grover's best observed value.""")
    s.finish()

    # Slide 10
    s = deck.slide("Deeper is not automatically better", section="Depth")
    s.add_box(0.72, 1.18, 7.40, 4.95, WHITE, line=LIGHT, radius=0.12)
    s.add_text(0.98, 1.42, 6.88, 0.32, "ZOOM: p = 90…110", 12, BLUE, bold=True, align="center", margin=0)
    near = plot_data["depths"] >= 90
    s.add_native_line_chart(
        0.96, 1.85, 6.93, 3.84,
        plot_data["depths"][near],
        [("Penalty-X", plot_data["px_opt"][near], BLUE), ("Global-Grover", plot_data["gg_opt"][near], CORAL)],
        max_y=0.00115, asset=ASSETS / "popt_near100.png",
    )
    s.add_metric_card(8.43, 1.18, 4.18, 1.00, "Observed near p≈100", "No sharp transition", accent=CORAL, sub="trajectories remain non-monotonic")
    s.add_metric_card(8.43, 2.36, 4.18, 1.00, "p=90…110 direction changes", "6 vs 9", accent=BLUE, sub="Penalty-X vs Global-Grover")
    s.add_metric_card(8.43, 3.54, 4.18, 1.00, "At p=110", "220 parameters", accent=TEAL, sub="504 COBYLA evaluations")
    s.add_metric_card(8.43, 4.72, 4.18, 1.00, "Optimization status", "All 220 cells", accent=GOLD, sub="budget-limited; p_opt < 10%")
    s.add_box(0.72, 6.30, 11.89, 0.48, GOLD_LIGHT, radius=0.08)
    s.add_text(0.96, 6.31, 11.41, 0.42, "Within this instance and finite budget, depth to 110 reveals no sharp Grover-scale success transition; ansatz and optimizer difficulty cannot be separated.", 13.5, NAVY, bold=True, align="center", margin=0)
    s.set_notes("""[~1:15] The natural depth question is whether something dramatic happens near p around one hundred. In the frozen data, it does not. The p=90 to 110 window remains oscillatory, with six direction changes for Penalty-X and nine for Global-Grover. Neither method reaches even ten percent total optimal probability. More importantly, the classical problem grows with depth: p=110 has 220 variables and only 504 objective evaluations. Every one of the 220 sweep cells exhausted its declared budget. We therefore cannot attribute poor high-depth performance only to the quantum ansatz or mixer; the classical optimization also becomes harder. The cautious conclusion is limited to this instance and protocol and does not disprove Grover scaling.""")
    s.finish()

    # Slide 11
    s = deck.slide("Scientific computing and validation", section="Verification")
    add_pipeline(s, 1.28, ["Model", "Simulate", "Optimize", "Save", "Validate", "Plot"], x=0.76, width=11.82)
    s.add_box(0.72, 2.15, 5.76, 3.68, WHITE, line=LIGHT, radius=0.12)
    s.add_text(0.98, 2.42, 5.23, 0.30, "REPRODUCIBLE COMPUTING STACK", 12, BLUE, bold=True, margin=0)
    stack = [
        "Python + NumPy exact statevectors",
        "SciPy COBYLA parameter optimization",
        "NetworkX exact path reference",
        "Explicit QUBO construction and Ising mapping",
        "CSV / JSON checkpoints and canonical tables",
        "Matplotlib figures + CLI scripts + Git",
    ]
    for index, item in enumerate(stack):
        s.add_bullet(0.98, 2.88 + 0.45 * index, 5.20, item, size=16, dot=BLUE if index < 3 else TEAL)
    s.add_box(6.76, 2.15, 5.86, 3.68, NAVY, radius=0.12)
    s.add_text(7.04, 2.42, 5.30, 0.30, "CURRENT VERIFICATION", 12, GOLD, bold=True, margin=0)
    checks = [
        "167 tests passing",
        "23 visualization checks passing",
        "43 / 43 depth-artifact checks",
        "43 / 43 CVaR-artifact checks",
        "max QUBO–Ising error = 0",
        "route 0→1→2→4→5→6, cost 10",
    ]
    for index, item in enumerate(checks):
        s.add_circle(7.08, 2.99 + 0.45 * index, 0.16, TEAL)
        s.add_text(7.28, 2.82 + 0.45 * index, 4.90, 0.34, item, 16.5, WHITE, bold=index in (0, 2, 4, 5), margin=0)
    s.add_box(0.72, 6.04, 11.90, 0.73, TEAL_LIGHT, radius=0.10)
    s.add_text(0.96, 6.19, 11.42, 0.38, "Smoke checks: src/main.py · Penalty-X command · course-final command · read-only visualization replay", 15.5, GREEN, bold=True, align="center", margin=0)
    s.set_notes("""[~0:55] The scientific-computing workflow is explicit and reproducible: build the model, simulate, optimize, save immutable artifacts, validate, and plot. The graph reference uses NetworkX, while NumPy performs exact statevector evolution and SciPy provides COBYLA. The current working tree passes 167 tests, including 23 visualization tests. The depth sweep and CVaR studies each retain 43 passing artifact-level validation checks. QUBO and Ising energies agree exactly across all 16,384 states, and the core smoke command still reproduces the unique cost-10 route. The important point is not software complexity; it is that every plotted number has a checkable route back to a saved artifact.""")
    s.finish()

    # Slide 12
    s = deck.slide("Takeaways and limitations", section="Conclusion")
    takeaways = [
        "Complete pipeline: graph → QUBO → Ising → QAOA → decoded routes.",
        "Penalty-X and Global-Grover create different probability distributions.",
        "Penalty-X puts substantially more mass on feasible routes.",
        "Global-Grover has stronger conditional optimum concentration at 89/110 depths.",
        "Conditional concentration alone does not determine total success.",
    ]
    s.add_text(0.72, 1.18, 7.20, 0.30, "FIVE TAKEAWAYS", 12, BLUE, bold=True, margin=0)
    for index, item in enumerate(takeaways, start=1):
        y = 1.68 + (index - 1) * 0.78
        s.add_circle(0.98, y + 0.24, 0.40, BLUE if index < 4 else CORAL)
        s.add_text(0.82, y + 0.09, 0.32, 0.28, str(index), 14, WHITE, bold=True, align="center", margin=0)
        s.add_text(1.30, y, 6.35, 0.58, item, 17, INK, bold=index in (1, 5), margin=0)
    s.add_box(8.12, 1.18, 4.50, 4.40, WHITE, line=LIGHT, radius=0.12)
    s.add_text(8.42, 1.47, 3.92, 0.30, "LIMITATIONS", 12, CORAL, bold=True, margin=0)
    limitations = [
        "one fixed 7-node instance",
        "ideal simulator; no QPU evidence",
        "finite, exhausted optimizer budgets",
        "high-depth optimization is difficult",
        "experiment-specific Global-Grover",
        "CVaR limited to tested protocol",
        "no quantum-advantage claim",
    ]
    for index, item in enumerate(limitations):
        s.add_bullet(8.40, 1.94 + 0.47 * index, 3.90, item, size=15.5, dot=CORAL)
    s.add_box(0.72, 5.82, 11.90, 0.86, NAVY, radius=0.10)
    s.add_equation("identity_final", r"p_{\mathrm{opt}}=p_{\mathrm{feas}}\,p_{\mathrm{opt}\mid\mathrm{feas}}", 0.98, 6.03, 4.04, 0.47, size=27, color=GOLD)
    s.add_text(5.17, 5.92, 7.16, 0.68, "Feasibility, optimal concentration, and optimization difficulty are separate effects — a mixer can improve one without improving the others.", 14.5, WHITE, bold=True, align="center", margin=0)
    s.set_notes("""[~0:45] The complete pipeline works end to end. The mixer choice matters because it changes how probability is transported through the same cost landscape. Penalty-X is much better at entering the feasible region; Global-Grover more often concentrates feasible mass toward the optimum. The product identity explains why improving one factor may not improve total success. These conclusions are deliberately narrow: one small graph, ideal simulation, finite exhausted optimization budgets, and no hardware or quantum-advantage claim. The final message is that feasibility, conditional quality, and classical optimization difficulty must be analyzed separately.""")
    s.finish()

    # Backup A
    s = deck.slide("Backup A — Why subtract bᵥ?", main=False, section="Backup")
    s.add_equation("backup_flow", r"\mathrm{out}(v)-\mathrm{in}(v)=b_v\quad\Longrightarrow\quad \mathrm{out}(v)-\mathrm{in}(v)-b_v=0", 1.10, 1.28, 11.15, 0.72, size=29)
    cards = [
        (0.82, "SOURCE", "b=+1", "one more outgoing\nthan incoming", BLUE),
        (4.72, "INTERMEDIATE", "b=0", "incoming equals\noutgoing", TEAL),
        (8.62, "TARGET", "b=−1", "one more incoming\nthan outgoing", CORAL),
    ]
    for x, title, bval, desc, accent in cards:
        s.add_box(x, 2.25, 3.05, 1.54, WHITE, line=accent, line_width=1.3, radius=0.12)
        s.add_text(x + 0.18, 2.48, 2.69, 0.25, title, 12, accent, bold=True, align="center", margin=0)
        s.add_text(x + 0.18, 2.88, 2.69, 0.30, bval, 21, NAVY, bold=True, align="center", margin=0)
        s.add_text(x + 0.18, 3.28, 2.69, 0.40, desc, 14, MUTED, align="center", margin=0)
    s.add_box(0.82, 4.16, 5.55, 1.92, TEAL_LIGHT, radius=0.12)
    s.add_text(1.10, 4.43, 5.00, 0.30, "VALID: 0→1→2→4→5→6", 17, GREEN, bold=True, align="center", margin=0)
    s.add_text(1.10, 4.88, 5.00, 0.84, "source: +1−(+1)=0\nnode 2: (1−1)−0=0\ntarget: −1−(−1)=0", 15, INK, mono=True, align="center", margin=0)
    s.add_box(6.76, 4.16, 5.55, 1.92, CORAL_LIGHT, radius=0.12)
    s.add_text(7.04, 4.43, 5.00, 0.30, "INVALID: select only 0→1", 17, RED, bold=True, align="center", margin=0)
    s.add_text(7.04, 4.88, 5.00, 0.84, "source residual = 0\nnode 1 residual = −1\ntarget residual = +1  → penalty 2", 15, INK, mono=True, align="center", margin=0)
    s.add_text(0.82, 6.34, 11.50, 0.31, "Squaring makes every nonzero flow residual contribute a positive energy penalty.", 17, NAVY, bold=True, align="center", margin=0)
    s.set_notes("Backup explanation of the flow residual and one valid/invalid example.")
    s.finish()

    # Backup B
    s = deck.slide("Backup B — QUBO matrix and Ising mapping", main=False, section="Backup")
    s.add_box(0.74, 1.22, 5.75, 2.26, WHITE, line=BLUE, radius=0.12)
    s.add_text(1.00, 1.50, 5.22, 0.28, "TWO EQUIVALENT QUBO VIEWS", 12, BLUE, bold=True, align="center", margin=0)
    s.add_equation("backup_poly", r"c+\sum_i a_i x_i+\sum_{i<j}b_{ij}x_ix_j", 1.18, 1.95, 4.80, 0.64, size=28)
    s.add_text(2.80, 2.62, 1.00, 0.42, "⇕", 20, MUTED, bold=True, align="center", margin=0)
    s.add_equation("backup_matrix", r"x^TQx+\mathrm{offset}", 2.05, 2.93, 2.58, 0.42, size=26)
    s.add_box(6.82, 1.22, 5.76, 2.26, WHITE, line=TEAL, radius=0.12)
    s.add_text(7.08, 1.50, 5.22, 0.28, "SUBSTITUTE BINARY OPERATORS", 12, TEAL, bold=True, align="center", margin=0)
    s.add_equation("backup_x", r"x_i=\frac{I-Z_i}{2}", 7.38, 1.96, 2.06, 0.64, size=30)
    s.add_equation("backup_xx", r"x_ix_j=\frac{1}{4}(I-Z_i-Z_j+Z_iZ_j)", 9.32, 1.96, 2.90, 0.64, size=25)
    s.add_text(7.19, 2.72, 5.02, 0.54, "Identity, single-Z, and ZZ terms appear after expansion.", 14.5, MUTED, align="center", margin=0)
    s.add_box(0.74, 3.86, 11.84, 1.26, NAVY, radius=0.12)
    s.add_equation("backup_ising", r"H_C=c_0I+\sum_i h_i Z_i+\sum_{i<j}J_{ij}Z_iZ_j", 3.10, 4.14, 7.15, 0.68, size=32, color=WHITE)
    s.add_metric_card(0.74, 5.47, 3.55, 0.98, "QUBO constant", "12", accent=BLUE, sub="for A=6")
    s.add_metric_card(4.57, 5.47, 3.55, 0.98, "Nonzero pairs", "46", accent=TEAL, sub="same in QUBO and Ising")
    s.add_metric_card(8.40, 5.47, 4.18, 0.98, "Exhaustive energy error", "0 across 16,384 states", accent=CORAL, sub="exact Fraction arithmetic")
    s.set_notes("Backup derivation of the matrix/polynomial equivalence and x-to-Z mapping.")
    s.finish()

    # Backup C
    s = deck.slide("Backup C — What is COBYLA?", main=False, section="Backup")
    s.add_box(0.74, 1.22, 5.68, 4.88, WHITE, line=LIGHT, radius=0.12)
    s.add_text(1.02, 1.42, 5.12, 0.76, "Constrained Optimization BY\nLinear Approximations", 19, BLUE, bold=True, align="center", margin=0)
    points = [
        "classical, derivative-free optimizer",
        "proposes QAOA angle vectors",
        "uses objective evaluations, not gradients",
        "does not choose routes directly",
        "budget limits are not convergence proofs",
    ]
    for index, item in enumerate(points):
        s.add_bullet(1.06, 2.37 + 0.58 * index, 5.00, item, size=17, dot=BLUE if index < 3 else CORAL)
    s.add_box(6.76, 1.22, 5.82, 4.88, NAVY, radius=0.12)
    s.add_text(7.07, 1.52, 5.20, 0.30, "WHY DEPTH IS HARD", 12, GOLD, bold=True, align="center", margin=0)
    s.add_equation("backup_params", r"n_{\theta}=2p", 8.68, 2.03, 1.74, 0.66, size=35, color=WHITE)
    s.add_text(7.16, 2.95, 5.02, 0.46, "p = 110  →  220 coupled angles", 22, WHITE, bold=True, align="center", margin=0)
    s.add_text(7.16, 3.67, 5.02, 0.46, "budget = max(120, 4p + 64)", 20, WHITE, mono=True, align="center", margin=0)
    s.add_text(7.16, 4.35, 5.02, 0.46, "p = 110  →  504 evaluations", 21, GOLD, bold=True, align="center", margin=0)
    s.add_text(7.16, 5.10, 5.02, 0.46, "All 220 depth-sweep cells reached their budget.", 16, WHITE, align="center", margin=0)
    s.add_text(0.74, 6.29, 11.84, 0.46, "A deeper ansatz can be more expressive while simultaneously being harder for the classical optimizer to tune.", 14.5, NAVY, bold=True, align="center", margin=0)
    s.set_notes("Backup explanation of COBYLA, the 2p parameter count, and the frozen evaluation rule.")
    s.finish()

    # Backup D
    s = deck.slide("Backup D — Global-Grover is not standard Grover Search", main=False, section="Backup")
    s.add_box(0.74, 1.20, 5.67, 0.58, BLUE_LIGHT, radius=0.10)
    s.add_text(0.94, 1.35, 5.27, 0.28, "STANDARD GROVER SEARCH", 16, BLUE, bold=True, align="center", margin=0)
    s.add_box(6.86, 1.20, 5.72, 0.58, CORAL_LIGHT, radius=0.10)
    s.add_text(7.06, 1.35, 5.32, 0.28, "THIS GLOBAL-GROVER QAOA", 16, CORAL, bold=True, align="center", margin=0)
    rows_compare = [
        ("Goal", "Find a marked state", "Optimize a cost distribution"),
        ("Problem phase", "Binary oracle marks target", "Continuous e⁻ⁱᵞᴴᶜ cost phase"),
        ("Mixing step", "Fixed diffusion", "Variational rank-one mixer"),
        ("Angles", "Prescribed iterations", "γ and β optimized by COBYLA"),
        ("Optimal route known?", "Oracle supplies marking", "No optimum-marking oracle"),
        ("Feasibility", "Depends on oracle/domain", "Full 16,384-state space; not preserved"),
    ]
    for index, (label, left, right) in enumerate(rows_compare):
        y = 2.02 + 0.70 * index
        fill = WHITE if index % 2 == 0 else "F0F3F7"
        s.add_box(0.74, y, 11.84, 0.58, fill, radius=0.04)
        s.add_text(0.90, y + 0.10, 1.70, 0.38, label, 11.5, MUTED, bold=True, margin=0)
        s.add_text(2.60, y + 0.10, 3.55, 0.34, left, 15, INK, margin=0)
        s.add_text(6.98, y + 0.10, 5.34, 0.34, right, 15, INK, margin=0)
    s.add_box(0.74, 6.32, 11.84, 0.48, GOLD_LIGHT, radius=0.08)
    s.add_text(0.98, 6.34, 11.36, 0.42, "The familiar π/4·√N iteration count is exploratory context only; it is not a prediction for this variational continuous-cost circuit.", 13.5, NAVY, bold=True, align="center", margin=0)
    s.set_notes("Backup comparison separating the repository's Grover-style mixer from standard oracle-based Grover Search.")
    s.finish()

    # Backup E
    s = deck.slide("Backup E — CVaR robustness study", main=False, section="Backup")
    paired = cvar["paired_p110"]
    s.paste_image(ASSETS / "cvar_summary.png", 0.72, 1.30, 6.48, 4.28, contain=True)
    # Use an editable clustered column chart in the PPTX.
    data = CategoryChartData()
    data.categories = ["Penalty-X", "Global-Grover"]
    data.add_series("Expectation", [paired["penalty_x"]["median_expectation_p_opt"], paired["global_grover"]["median_expectation_p_opt"]])
    data.add_series("CVaR 0.10", [paired["penalty_x"]["median_cvar_010_p_opt"], paired["global_grover"]["median_cvar_010_p_opt"]])
    chart = s.slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.72), Inches(1.30), Inches(6.48), Inches(4.28), data).chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.value_axis.minimum_scale = 0.0
    chart.value_axis.maximum_scale = 0.00285
    chart.value_axis.number_format = "0.00%"
    chart.value_axis.number_format_is_linked = False
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.major_gridlines.format.line.color.rgb = rgb(LIGHT)
    chart.font.name = PPT_FONT
    chart.font.size = Pt(11)
    for series_item, color in zip(chart.series, (BLUE, CORAL)):
        series_item.format.fill.solid()
        series_item.format.fill.fore_color.rgb = rgb(color)
        series_item.format.line.color.rgb = rgb(color)
    s.add_box(7.52, 1.30, 5.08, 1.10, WHITE, line=BLUE, radius=0.10)
    s.add_text(7.78, 1.52, 4.56, 0.28, "PENALTY-X · p=110", 12, BLUE, bold=True, margin=0)
    s.add_text(7.78, 1.91, 4.56, 0.28, "7/10 wins · median ratio 2.971 · MIXED", 16, INK, bold=True, margin=0)
    s.add_box(7.52, 2.63, 5.08, 1.10, WHITE, line=CORAL, radius=0.10)
    s.add_text(7.78, 2.85, 4.56, 0.28, "GLOBAL-GROVER · p=110", 12, CORAL, bold=True, margin=0)
    s.add_text(7.78, 3.24, 4.56, 0.28, "10/10 wins · median ratio 3.929", 16, INK, bold=True, margin=0)
    s.add_box(7.52, 3.96, 5.08, 1.62, NAVY, radius=0.10)
    s.add_text(7.78, 4.17, 4.56, 0.28, "FROZEN PROTOCOL", 12, GOLD, bold=True, margin=0)
    s.add_text(7.78, 4.57, 4.56, 0.78, "alpha in {0.05, 0.10, 0.25, 0.50}\nplus expectation (alpha=1)\np = 50, 100, 110 · 140 cells", 15, WHITE, align="center", margin=0)
    s.add_box(0.72, 5.86, 11.88, 0.84, TEAL_LIGHT, radius=0.10)
    s.add_text(0.98, 6.02, 11.36, 0.50, "Supported mechanism: feasible-region amplification, not consistently stronger optimal concentration within the feasible set. All cells were budget-exhausted; absolute p_opt remained small.", 15.5, GREEN, bold=True, align="center", margin=0)
    s.set_notes("Backup summary of the separate fresh-seed CVaR robustness study and its claim boundary.")
    s.finish()

    return deck


def notes_markdown(deck: DeckBuilder) -> str:
    titles = [
        "From Routing to QAOA: How Mixer Design and Circuit Depth Shape Feasibility and Optimality",
        "A small graph, a large binary space",
        "From edges to a routing QUBO",
        "QUBO → Ising → QAOA",
        "Same cost landscape, two mixer geometries",
        "The hybrid loop: COBYLA and CVaR",
        "What we measure — and how",
        "Depth 1–110: where does probability go?",
        "Conditional quality is not total success",
        "Deeper is not automatically better",
        "Scientific computing and validation",
        "Takeaways and limitations",
        "Backup A — Why subtract bᵥ?",
        "Backup B — QUBO matrix and Ising mapping",
        "Backup C — What is COBYLA?",
        "Backup D — Global-Grover is not standard Grover Search",
        "Backup E — CVaR robustness study",
    ]
    timings = ["0:50", "1:10", "1:20", "1:15", "1:25", "1:15", "1:10", "1:35", "1:25", "1:15", "0:55", "0:45"]
    lines = [
        "# QAOA Routing Final Presentation — Speaker Notes",
        "",
        "**Target main-deck timing:** approximately 14:20–15:00, excluding backup slides.",
        "",
        "**Presentation boundary:** one teaching-scale routing instance, ideal simulation, no quantum-advantage or standard-Grover speedup claim.",
        "",
    ]
    for (number, notes), title in zip(deck.notes, titles):
        timing = timings[number - 1] if number <= 12 else "backup"
        lines.extend([f"## Slide {number} — {title}", "", f"**Timing:** {timing}", "", notes, ""])
    lines.extend([
        "## Suggested transitions",
        "",
        "- Slide 2 → 3: “The graph is small, but the binary state space already contains many invalid edge combinations. So the next question is how to encode route validity.”",
        "- Slide 4 → 5: “The cost Hamiltonian tells us which solutions are good, but the mixer determines how amplitude moves between them. This is where our two QAOA designs differ.”",
        "- Slide 7 → 8: “With these three metrics defined, we can now ask where the probability actually goes as depth increases.”",
        "- Slide 8 → 9: “The three curves appear to tell different stories, and the product identity resolves the apparent contradiction.”",
        "- Slide 10 → 11: “Because high-depth conclusions depend on the numerical protocol, validation and provenance are part of the scientific result.”",
        "",
    ])
    return "\n".join(lines)


def audit_markdown(graph, analysis, reference, validation, cvar, cvar_validation, rows, figure_hashes: dict[str, str]) -> str:
    head = git_output("rev-parse", "HEAD")
    status = git_output("status", "--short")
    px110 = analysis["algorithms"]["penalty_x"]["p110"]
    gg110 = analysis["algorithms"]["global_grover"]["p110"]
    ratio_cond = gg110["p_opt_given_feasible"] / px110["p_opt_given_feasible"]
    ratio_feas = px110["p_feas"] / gg110["p_feas"]
    ratio_p110 = gg110["p_opt"] / px110["p_opt"]
    ratio_best = analysis["algorithms"]["penalty_x"]["best_p_opt"] / analysis["algorithms"]["global_grover"]["best_p_opt"]
    claims = [
        ("1", "Course / title identity: DTU 10387 SCIQIS; author handle ZhilinChen02; repository URL", "README.md; pyproject.toml; git remote; git log"),
        ("2", "7 nodes, 14 directed positive-weight edges, source 0, target 6", "data/graph.json; src/graph.py"),
        ("2", "2^14 = 16,384 bitstrings", "data/graph.json; src/graph.py; results/global_depth110/reference_solution.json"),
        ("2", "20 valid decoded routes = 0.001220703125 of the full space", "results/global_depth110/reference_solution.json"),
        ("2", "Unique optimum 0→1→2→4→5→6, routing cost 10; state 10377; bitstring 10010001000101", "results/global_depth110/reference_solution.json; src/graph.py"),
        ("3", "Penalty coefficient A=6", "configs/global_depth110.json; results/global_depth110/experiment_config.json"),
        ("4/B", "Expanded QUBO has 14 linear coefficients and 46 nonzero pair coefficients; QUBO constant 12; Ising energy error 0", "src/qubo.py; read-only exhaustive calculation used for deck"),
        ("5", "Penalty-X is exp(-i beta sum X_j) without beta/n scaling in the depth-110 study", "configs/global_depth110.json; src/global_depth_sweep/simulator.py; src/qaoa.py"),
        ("5", "Global-Grover update is I plus (exp(-i beta)-1) times the rank-one projector onto the uniform state over all 16,384 states; no feasible projection or threshold phase", "src/qaoa.py; configs/global_depth110.json; results/global_depth110/validation_summary.json"),
        ("6/7", "2p parameters; p=110 gives 220 parameters", "configs/global_depth110.json; src/global_depth_sweep/experiment.py"),
        ("6/10/C", "Evaluation budget max(120,4p+64); p=110 gives 504 evaluations", "configs/global_depth110.json; src/global_depth_sweep/experiment.py"),
        ("7", "p=1…110; 2 mixers × 110 depths = 220 rows; deterministic layerwise continuation", "results/global_depth110/canonical_rows.csv; configs/global_depth110.json"),
        ("8", f"Penalty-X best p=21, p_opt={analysis['algorithms']['penalty_x']['best_p_opt']:.15g}", "results/global_depth110/analysis_summary.json"),
        ("8", f"Global-Grover best p=110, p_opt={analysis['algorithms']['global_grover']['best_p_opt']:.15g}", "results/global_depth110/analysis_summary.json"),
        ("8", "First Global-Grover total-p_opt crossover p=22; no later Penalty-X overtake in the frozen trajectory", "results/global_depth110/analysis_summary.json"),
        ("8/12", "Penalty-X has higher p_feas at 110/110 depths; Global-Grover higher p_opt|feas at 89/110 depths", "results/global_depth110/analysis_summary.json"),
        ("9", f"Penalty-X p110: p_feas={px110['p_feas']:.15g}, p_opt|feas={px110['p_opt_given_feasible']:.15g}, p_opt={px110['p_opt']:.15g}", "results/global_depth110/analysis_summary.json"),
        ("9", f"Global-Grover p110: p_feas={gg110['p_feas']:.15g}, p_opt|feas={gg110['p_opt_given_feasible']:.15g}, p_opt={gg110['p_opt']:.15g}", "results/global_depth110/analysis_summary.json"),
        ("9", f"Derived ratios: conditional={ratio_cond:.6f}×; Penalty-X feasible mass={ratio_feas:.6f}× Global; Global p110 total={ratio_p110:.6f}× Penalty-X; Penalty-X best={ratio_best:.6f}× Global best", "Derived directly from the preceding frozen p110 and best-p_opt values"),
        ("10", "p_opt direction changes p90–p110: Penalty-X 6, Global-Grover 9; across all depths: 52 and 50", "results/global_depth110/analysis_summary.json"),
        ("10", "Neither method reaches p_opt ≥ 0.10; all 220 rows are budget-limited", "results/global_depth110/analysis_summary.json; REPORT.md"),
        ("11", "167 tests pass; 23 targeted visualization tests pass", "Commands recorded below, run 2026-08-20"),
        ("11", f"Depth artifact validation {validation['checks_passed']}/{validation['checks_passed'] + validation['checks_failed']}; CVaR artifact validation {cvar_validation['checks_passed']}/{cvar_validation['checks_passed'] + cvar_validation['checks_failed']}", "results/global_depth110/validation_summary.json; results/cvar_robustness_v1/validation_summary.json"),
        ("6/E", f"CVaR p110 Penalty-X: 7/10 wins, median paired ratio {cvar['paired_p110']['penalty_x']['median_paired_ratio']:.6f}, MIXED_CVAR_RECOVERY", "results/cvar_robustness_v1/analysis_summary.json; FINAL_SCIENTIFIC_INTERPRETATION.md"),
        ("6/E", f"CVaR p110 Global-Grover: 10/10 wins, median paired ratio {cvar['paired_p110']['global_grover']['median_paired_ratio']:.6f}", "results/cvar_robustness_v1/analysis_summary.json"),
        ("E", "CVaR alphas 0.05, 0.10, 0.25, 0.50 plus expectation alpha=1; depths 50, 100, 110; 140 cells", "configs/cvar_robustness_v1.json; results/cvar_robustness_v1/frozen_manifest.json"),
    ]
    lines = [
        "# Presentation Data Audit",
        "",
        f"**Generated:** {date.today().isoformat()} (Europe/Copenhagen project context)",
        "",
        f"**Git HEAD:** `{head}`",
        "",
        "**Working-tree note:** The repository already contained uncommitted scientific/code changes before this presentation was generated. The deck therefore cites the current working tree and frozen result artifacts, not HEAD alone.",
        "",
        "## Numerical and protocol claims used",
        "",
        "| Slide | Claim | Source of truth |",
        "|---|---|---|",
    ]
    lines.extend(f"| {slide} | {claim.replace('|', '/')} | `{source}` |" for slide, claim, source in claims)
    lines.extend([
        "",
        "## Generated figures and editable chart data",
        "",
        "All depth figures were regenerated without optimization from `results/global_depth110/depth_by_depth.csv`. The PPTX uses native editable PowerPoint charts; the PNG assets are visually matched render references used for the PDF and slide audit.",
        "",
        "| Asset | Data source | SHA-256 |",
        "|---|---|---|",
    ])
    for name, digest in sorted(figure_hashes.items()):
        source = "results/global_depth110/depth_by_depth.csv" if "depth" in name or "near100" in name else "results/cvar_robustness_v1/analysis_summary.json"
        lines.append(f"| `presentation/assets/{name}` | `{source}` | `{digest}` |")
    lines.extend([
        "",
        "The routing graph on slide 2 is not a raster copy: it is drawn as editable PowerPoint nodes, edges, weights, and labels directly from `data/graph.json`. The PDF reference uses the same build coordinates.",
        "",
        "## Exact validation commands run",
        "",
        "```bash",
        ".venv/bin/python -m pytest -q",
        "# 167 passed in 49.06s",
        "",
        ".venv/bin/python src/main.py",
        "# reproduced 14 edges, 16,384 states, route 0->1->2->4->5->6, cost 10",
        "",
        ".venv/bin/python scripts/run_penalty_qaoa.py --budget 8",
        "# smoke completed; intentionally budget-limited at 8 evaluations",
        "",
        ".venv/bin/python scripts/run_course_final.py --seed 2601 --depth 1",
        "# smoke completed; 20-route feasible basis, p_feas=1, 81 evaluations",
        "",
        ".venv/bin/python -m pytest -q tests/test_qaoa_visualization.py tests/test_qaoa_dynamics_visualization.py",
        "# 23 passed in 5.22s",
        "```",
        "",
        "No expensive optimizer sweep was rerun, and no file under `results/` was modified.",
        "",
        "## Deliberately excluded or corrected material",
        "",
        "- `figures/course/02`–`05`, `configs/course_final.json`, and the Q2-F/Q2-Final artifacts describe separate feasible-subspace, path-exchange, or threshold-Grover studies. They are not the full-space Penalty-X vs Global-Grover p=1…110 experiment and were excluded from the main deck.",
        "- `results/qaoa_dynamics_deep_dive/v1` is a useful p=1/p=2 mechanism pilot, but its older scaled-X convention and shallow results were not used as the final depth evidence.",
        "- `results/global_depth110_recovery_bfo_v1` is exploratory/pilot evidence and was excluded from headline claims.",
        "- Q2 revision/extension material, benchmark-portfolio material, and larger research framing were excluded to keep this a DTU teaching-scale course story.",
        "- The prompt's tentative wording “fixed evaluation budget” was corrected to the repository's depth-dependent but mixer-matched rule `max(120,4p+64)`.",
        "- The prompt's tentative wording “each depth independently optimized” was corrected: each method-depth has an optimization, but p>1 is initialized by deterministic layerwise continuation from p−1.",
        "- The old presentation-story claim of 103 tests was excluded; the current suite has 167 passing tests.",
        "- No standard-Grover quadratic speedup, Grover-scale transition, hardware performance, depth scaling law, or quantum advantage is claimed.",
        "",
        "## Working-tree status at build time (including the generated presentation directory)",
        "",
        "```text",
        status,
        "```",
        "",
    ])
    return "\n".join(lines)


def validate_and_save(deck: DeckBuilder) -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    SLIDE_PNGS.mkdir(parents=True, exist_ok=True)
    deck.prs.save(PPTX_PATH)
    reopened = Presentation(PPTX_PATH)
    if len(reopened.slides) != 17:
        raise RuntimeError(f"expected 17 slides, found {len(reopened.slides)}")
    if reopened.slide_width != Inches(SLIDE_W) or reopened.slide_height != Inches(SLIDE_H):
        raise RuntimeError("presentation is not 16:9 wide format")
    bounds_failures = []
    for slide_index, slide in enumerate(reopened.slides, start=1):
        for shape in slide.shapes:
            if shape.left < -1000 or shape.top < -1000 or shape.left + shape.width > reopened.slide_width + 1000 or shape.top + shape.height > reopened.slide_height + 1000:
                bounds_failures.append((slide_index, shape.name))
    if bounds_failures:
        raise RuntimeError(f"out-of-bounds PPTX shapes: {bounds_failures}")
    main_notes = [reopened.slides[i].notes_slide.notes_text_frame.text.strip() for i in range(12)]
    if not all(main_notes):
        raise RuntimeError("one or more main slides is missing speaker notes")
    if any(check.used_height_px > check.available_height_px + 3 for check in deck.text_checks):
        raise RuntimeError("text overflow check failed")
    for index, image in enumerate(deck.renders, start=1):
        image.save(SLIDE_PNGS / f"slide_{index:02d}.png", format="PNG", optimize=True)
    rgb_images = [image.convert("RGB") for image in deck.renders]
    rgb_images[0].save(PDF_PATH, save_all=True, append_images=rgb_images[1:], resolution=144.0, quality=95)
    with ZipFile(PPTX_PATH) as archive:
        names = archive.namelist()
        notes_count = len([name for name in names if name.startswith("ppt/notesSlides/notesSlide") and name.endswith(".xml")])
        chart_count = len([name for name in names if name.startswith("ppt/charts/chart") and name.endswith(".xml")])
    return {
        "slide_count": len(reopened.slides),
        "notes_count": notes_count,
        "chart_count": chart_count,
        "shape_bounds_failures": len(bounds_failures),
        "text_overflow_failures": 0,
    }


def build_contact_sheet() -> Path:
    images = [Image.open(path).convert("RGB") for path in sorted(SLIDE_PNGS.glob("slide_*.png"))]
    thumb_w, thumb_h = 480, 270
    cols = 3
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), pil_rgb("D9DEE7"))
    for index, image in enumerate(images):
        thumb = ImageOps.fit(image, (thumb_w - 8, thumb_h - 8), method=Image.Resampling.LANCZOS)
        x = (index % cols) * thumb_w + 4
        y = (index // cols) * thumb_h + 4
        sheet.paste(thumb, (x, y))
    path = RENDERED / "contact_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, optimize=True)
    return path


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    graph, analysis, reference, validation, cvar, cvar_validation, rows = load_inputs()
    plot_data = build_plot_assets(rows, analysis, cvar)
    deck = build_deck(graph, analysis, reference, validation, cvar, cvar_validation, rows, plot_data)
    NOTES_PATH.write_text(notes_markdown(deck), encoding="utf-8")
    validation_result = validate_and_save(deck)
    figure_hashes = {path.name: sha256_file(path) for path in ASSETS.glob("*.png") if path.name in {"popt_depth.png", "pfeas_depth.png", "pcond_depth.png", "popt_near100.png", "cvar_summary.png"}}
    AUDIT_PATH.write_text(audit_markdown(graph, analysis, reference, validation, cvar, cvar_validation, rows, figure_hashes), encoding="utf-8")
    contact = build_contact_sheet()
    manifest = {
        "outputs": {
            "pptx": str(PPTX_PATH),
            "pdf": str(PDF_PATH),
            "notes": str(NOTES_PATH),
            "audit": str(AUDIT_PATH),
            "contact_sheet": str(contact),
        },
        "validation": validation_result,
        "hashes": {
            "pptx": sha256_file(PPTX_PATH),
            "pdf": sha256_file(PDF_PATH),
            "notes": sha256_file(NOTES_PATH),
            "audit": sha256_file(AUDIT_PATH),
        },
    }
    (RENDERED / "build_validation.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
