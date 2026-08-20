#!/usr/bin/env python3
"""Build the final course-first QAOA routing presentation.

Only tracked course figures are read. No optimizer or scientific experiment is
run. Rendered slide PNGs are local validation outputs and are ignored by Git.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation"
RENDERED = OUT / "rendered"
SLIDES = RENDERED / "slides"
PPTX = OUT / "QAOA_Routing_Final_Presentation.pptx"
PDF = OUT / "QAOA_Routing_Final_Presentation.pdf"
NOTES = OUT / "QAOA_Routing_Final_Presentation_Notes.md"
AUDIT = OUT / "PRESENTATION_DATA_AUDIT.md"

SLIDE_W = 13.333
SLIDE_H = 7.5
CANVAS = (1600, 900)

NAVY = "14213D"
INK = "26364D"
MUTED = "617087"
LIGHT = "E8EDF4"
BG = "F7F9FC"
BLUE = "20639B"
TEAL = "2A9D8F"
GOLD = "E9B949"
CORAL = "D14D5A"
WHITE = "FFFFFF"


def rgb(value: str) -> RGBColor:
    return RGBColor.from_string(value)


def pil_color(value: str) -> tuple[int, int, int]:
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def font_path(*names: str) -> Path:
    roots = (Path("/usr/share/fonts"), Path.home() / ".local/share")
    for name in names:
        for root in roots:
            matches = list(root.rglob(name)) if root.exists() else []
            if matches:
                return matches[0]
    raise FileNotFoundError(f"font not found: {names}")


REGULAR_FONT = font_path(
    "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Inter-Regular.otf"
)
BOLD_FONT = font_path(
    "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Inter-SemiBold.otf"
)


def pil_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD_FONT if bold else REGULAR_FONT), size)


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


@dataclass(frozen=True)
class SlideSpec:
    title: str
    kicker: str
    bullets: tuple[str, ...]
    images: tuple[Path, ...] = ()
    notes: str = ""
    hero: bool = False


COURSE = ROOT / "figures/course"
DYNAMICS = ROOT / "figures/qaoa_dynamics_deep_dive/v1"
Q2F = next((ROOT / "results/q2f_course_extension").glob("q2f-*"))


SPECS = (
    SlideSpec(
        "QAOA Routing",
        "DTU SCIQIS COURSE PROJECT",
        (
            "From a weighted graph to QUBO, Ising, QAOA, and decoded routes",
            "Ideal statevector simulation · teaching-scale instance · no quantum-advantage claim",
        ),
        notes="We show the complete scientific chain for one transparent routing instance. The aim is to understand the encoding and probability dynamics, not to outperform classical shortest-path algorithms.",
        hero=True,
    ),
    SlideSpec(
        "The routing problem",
        "GRAPH AND EXACT REFERENCE",
        (
            "7 nodes and 14 directed weighted edges",
            "Source 0, target 6; one binary variable per edge",
            "Unique optimum: 0 → 1 → 2 → 4 → 5 → 6",
            "Exact route cost: 10",
        ),
        (COURSE / "01_routing_graph_and_routes.png",),
        "The graph is fixed in data/graph.json. NetworkX shortest path and exhaustive simple-route enumeration agree on the same unique cost-10 route.",
    ),
    SlideSpec(
        "Edge variables become a routing QUBO",
        "CONSTRAINED BINARY OPTIMIZATION",
        (
            "x_e = 1 means that directed edge e is selected",
            "Routing cost is the weighted sum of selected edges",
            "Squared flow residuals enforce source, conservation, and target constraints",
            "The frozen penalty coefficient is A = 6",
        ),
        (OUT / "assets/eq_final_qubo.png",),
        "The objective says which route is cheap, while the flow penalty says whether the selected edges form a valid source-to-target route.",
    ),
    SlideSpec(
        "QUBO → Ising → QAOA",
        "QUANTUM MODEL",
        (
            "Substitute x_j = (I − Z_j) / 2",
            "The diagonal cost Hamiltonian contains I, Z, and ZZ terms",
            "A cost phase changes phases; the mixer converts phase differences into probability changes",
            "All 16,384 QUBO and Ising basis energies agree exactly",
        ),
        (DYNAMICS / "02_qubo_ising_qaoa_pipeline.png",),
        "The cost layer alone does not change measurement probabilities. Interference appears after the mixer, so the mixer geometry is scientifically important.",
    ),
    SlideSpec(
        "Two search spaces, three mixers",
        "REPRESENTATION MATTERS",
        (
            "Penalty-X: 2^14 = 16,384 edge bitstrings",
            "Global-Grover: the same full bitstring space",
            "Feasible-Grover: only 20 enumerated valid routes",
            "Global-Grover and Feasible-Grover are QAOA mixers, not standard Grover search",
        ),
        (DYNAMICS / "03_three_search_spaces_and_mixers.png",),
        "Restricting the logical basis to valid routes structurally removes infeasible probability mass, but it does not by itself guarantee concentration on the optimum.",
    ),
    SlideSpec(
        "What happens to probability mass?",
        "SHALLOW QAOA DYNAMICS",
        (
            "Full-space initial p_feas = 20 / 16,384 ≈ 0.00122",
            "At p=2, Global-Grover reaches p_opt ≈ 0.000213",
            "At p=2, Feasible-Grover keeps p_feas = 1 and reaches p_opt ≈ 0.170927",
            "Feasible restriction solves validity; phase and mixer still determine optimality",
        ),
        (DYNAMICS / "06_probability_mass_decomposition.png",),
        "The full-space methods leave most probability on infeasible bitstrings. The feasible basis removes that mass by construction and lets us study concentration among valid routes.",
    ),
    SlideSpec(
        "Final course method comparison",
        "THREE SEEDS · MEDIAN p_opt",
        (
            "Q2-F Expectation, p=3: 0.264901",
            "BSP / path exchange, p=3: 0.444656",
            "GM-QAOA Expectation, p=3: 0.182410",
            "GM-Th-QAOA, p=3: 0.998712",
            "All four methods use the 20-route feasible basis, so p_feas ≈ 1",
        ),
        (COURSE / "02_popt_versus_depth.png",),
        "The table uses the saved three-seed median. Evaluation caps differ between the historical Q2-F baseline and the final matrix, so this is a course-result summary rather than a strict equal-budget leaderboard.",
    ),
    SlideSpec(
        "The final distribution is concentrated on the optimum",
        "GM-Th-QAOA · DEPTH 3",
        (
            "Median p_opt = 0.998712",
            "Seed range: 0.816034–0.999804",
            "Median expected route cost = 10.002983",
            "The highlighted state is the unique cost-10 route",
        ),
        (COURSE / "04_final_probability_distribution.png",),
        "This is an ideal statevector probability distribution, not finite-shot frequency. The optimum label is used for evaluation and presentation, not supplied directly to COBYLA.",
    ),
    SlideSpec(
        "Why GM-Th works on this instance",
        "FEASIBLE BASIS + THRESHOLD PHASE + MIXER",
        (
            "The logical basis contains only valid routes: p_feas = 1",
            "A deterministic greedy incumbent has cost 11",
            "The threshold marks routes with cost < 11",
            "On this graph, that marked set contains exactly the unique optimum",
            "The feasible Grover mixer amplifies the marked probability",
        ),
        (COURSE / "05_grover_amplification.png",),
        "The near-unit success probability depends on instance-specific threshold information. This is a clear mechanism demonstration, not a scalable quantum-routing or quantum-advantage result.",
    ),
    SlideSpec(
        "CVaR course extension: a useful negative result",
        "EXPECTATION · CVaR · ASCENDING CVaR",
        (
            "CVaR changes the classical objective, not the mixer",
            "At depth 3: Expectation 0.264901, CVaR 0.258176, Ascending CVaR 0.263190",
            "CVaR did not provide a clear improvement in this course experiment",
            "The fractional cutoff implementation remains tested and available",
        ),
        (Q2F / "figures/03_popt_versus_depth.png",),
        "Some shallow cells improve, but the saved three-seed depth-three comparison does not show stable superiority over expectation. We report this as a legitimate negative result.",
    ),
    SlideSpec(
        "Run it and watch it",
        "REPRODUCIBLE COURSE DEMO",
        (
            "python -m src.main",
            "python scripts/run_course_final.py --seed 2601 --depth 1",
            "python scripts/run_qaoa_visualizer.py",
            "python scripts/run_qaoa_dynamics_visualizer.py",
            "Both browser demos replay saved data and do not rerun historical experiments",
        ),
        (DYNAMICS / "07_layer_by_layer_expected_hc.png",),
        "The circuit replay focuses on the final feasible methods. The dynamics dashboard compares Penalty-X, Global-Grover, and Feasible-Grover layer by layer.",
    ),
    SlideSpec(
        "What we learned",
        "COURSE TAKEAWAYS AND LIMITATIONS",
        (
            "Graph → QUBO → Ising → QAOA → decoded routes works end to end",
            "Full-space QAOA can waste most probability on infeasible bitstrings",
            "Feasibility preservation does not automatically imply optimal-state concentration",
            "Mixer and phase/objective design strongly change the final distribution",
            "One fixed graph, ideal simulation, enumerated feasible basis, finite optimizer budgets",
            "No hardware, scalability, or quantum-advantage claim",
        ),
        notes="The main scientific message is the separation of validity from optimality. Representation controls where amplitude is allowed; the phase and mixer control where probability concentrates inside that representation.",
    ),
    SlideSpec(
        "Backup: a two-qubit Ising gate example",
        "COURSE IMPLEMENTATION DETAIL",
        (
            "Local Z terms become RZ rotations",
            "A ZZ coupling becomes CX–RZ–CX",
            "The X mixer becomes RX rotations",
            "This is a small teaching decomposition, not the full 14-qubit compiled circuit",
        ),
        (COURSE / "simple_ising_circuit.png",),
        "This optional Qiskit example connects the Ising formula to familiar gates without claiming that the logical feasible-space operators have a shallow hardware decomposition.",
    ),
)


def add_text(slide, x, y, w, h, text, size, color=INK, *, bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.margin_left = frame.margin_right = Inches(0.02)
    frame.margin_top = frame.margin_bottom = Inches(0.01)
    frame.vertical_anchor = MSO_ANCHOR.TOP
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.alignment = align
    paragraph.font.name = "Aptos"
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = rgb(color)
    return shape


def add_bullets(slide, bullets: tuple[str, ...], x=0.75, y=1.55, w=4.6, h=5.25):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.margin_left = Inches(0.05)
    for index, text in enumerate(bullets):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = text
        paragraph.level = 0
        paragraph.font.name = "Aptos"
        paragraph.font.size = Pt(18 if len(bullets) <= 4 else 16)
        paragraph.font.color.rgb = rgb(INK)
        paragraph.space_after = Pt(13)
        paragraph.text = f"•  {text}"
    return shape


def contain_box(path: Path, x, y, w, h):
    with Image.open(path) as image:
        ratio = image.width / image.height
    box_ratio = w / h
    if ratio >= box_ratio:
        height = w / ratio
        return x, y + (h - height) / 2, w, height
    width = h * ratio
    return x + (w - width) / 2, y, width, h


def build_pptx() -> None:
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)
    blank = prs.slide_layouts[6]
    for number, spec in enumerate(SPECS, start=1):
        slide = prs.slides.add_slide(blank)
        background = slide.background.fill
        background.solid()
        background.fore_color.rgb = rgb(NAVY if spec.hero else BG)
        if spec.hero:
            add_text(slide, 0.95, 1.35, 11.4, 1.25, spec.title, 44, WHITE, bold=True)
            add_text(slide, 0.98, 0.72, 8.0, 0.35, spec.kicker, 14, GOLD, bold=True)
            add_text(slide, 1.0, 3.05, 10.9, 1.35, spec.bullets[0], 26, WHITE)
            add_text(slide, 1.0, 5.45, 10.9, 0.55, spec.bullets[1], 15, LIGHT)
        else:
            add_text(slide, 0.72, 0.34, 9.8, 0.3, spec.kicker, 11, BLUE, bold=True)
            add_text(slide, 0.70, 0.72, 11.9, 0.55, spec.title, 28, NAVY, bold=True)
            add_bullets(slide, spec.bullets)
            if spec.images:
                x, y, w, h = contain_box(spec.images[0], 5.45, 1.42, 7.25, 5.45)
                slide.shapes.add_picture(str(spec.images[0]), Inches(x), Inches(y), Inches(w), Inches(h))
            add_text(slide, 12.15, 7.02, 0.55, 0.25, str(number), 10, MUTED, align=PP_ALIGN.RIGHT)
        if spec.notes:
            notes_frame = slide.notes_slide.notes_text_frame
            notes_frame.text = spec.notes
    prs.core_properties.title = "QAOA Routing — Final SCIQIS Course Presentation"
    prs.core_properties.subject = "Graph to QUBO, Ising, QAOA, and decoded routes"
    prs.core_properties.keywords = "QAOA, routing, QUBO, Ising, CVaR, SCIQIS"
    prs.save(PPTX)


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines = []
    for paragraph in text.splitlines() or [""]:
        words = paragraph.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if draw.textbbox((0, 0), candidate, font=font)[2] <= width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def paste_contained(canvas: Image.Image, path: Path, box: tuple[int, int, int, int]) -> None:
    x, y, w, h = box
    with Image.open(path) as source:
        image = ImageOps.contain(source.convert("RGB"), (w, h), Image.Resampling.LANCZOS)
    canvas.paste(image, (x + (w - image.width) // 2, y + (h - image.height) // 2))


def render_slide(spec: SlideSpec, number: int) -> Image.Image:
    canvas = Image.new("RGB", CANVAS, pil_color(NAVY if spec.hero else BG))
    draw = ImageDraw.Draw(canvas)
    if spec.hero:
        draw.text((115, 90), spec.kicker, font=pil_font(24, bold=True), fill=pil_color(GOLD))
        draw.text((115, 175), spec.title, font=pil_font(66, bold=True), fill=pil_color(WHITE))
        y = 370
        for line in wrap(draw, spec.bullets[0], pil_font(34), 1300):
            draw.text((120, y), line, font=pil_font(34), fill=pil_color(WHITE))
            y += 48
        draw.text((120, 715), spec.bullets[1], font=pil_font(20), fill=pil_color(LIGHT))
        return canvas

    draw.text((84, 42), spec.kicker, font=pil_font(18, bold=True), fill=pil_color(BLUE))
    draw.text((82, 88), spec.title, font=pil_font(38, bold=True), fill=pil_color(NAVY))
    draw.line((82, 153, 1515, 153), fill=pil_color(LIGHT), width=3)
    y = 190
    bullet_font = pil_font(23 if len(spec.bullets) <= 4 else 20)
    for bullet in spec.bullets:
        lines = wrap(draw, bullet, bullet_font, 510)
        draw.ellipse((92, y + 10, 104, y + 22), fill=pil_color(TEAL))
        for line in lines:
            draw.text((122, y), line, font=bullet_font, fill=pil_color(INK))
            y += 31
        y += 20
    if spec.images:
        paste_contained(canvas, spec.images[0], (655, 175, 850, 650))
    draw.text((1490, 852), str(number), font=pil_font(15), fill=pil_color(MUTED))
    return canvas


def build_pdf_and_validation() -> list[Path]:
    SLIDES.mkdir(parents=True, exist_ok=True)
    images = []
    paths = []
    for number, spec in enumerate(SPECS, start=1):
        image = render_slide(spec, number)
        path = SLIDES / f"slide_{number:02d}.png"
        image.save(path, optimize=True)
        images.append(image)
        paths.append(path)
    images[0].save(PDF, "PDF", resolution=144, save_all=True, append_images=images[1:])
    thumbs = [ImageOps.contain(image, (380, 214)) for image in images]
    sheet = Image.new("RGB", (1200, ((len(thumbs) + 2) // 3) * 245), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(thumb, (15 + (index % 3) * 395, 15 + (index // 3) * 245))
    RENDERED.mkdir(parents=True, exist_ok=True)
    sheet.save(RENDERED / "contact_sheet.png", optimize=True)
    validation = {
        "course_first": True,
        "scientific_experiments_rerun": False,
        "slide_count": len(SPECS),
        "all_source_images_exist": all(path.is_file() for spec in SPECS for path in spec.images),
        "pptx_bytes": PPTX.stat().st_size,
        "pdf_bytes": PDF.stat().st_size,
        "pptx_sha256": digest(PPTX),
        "pdf_sha256": digest(PDF),
    }
    (RENDERED / "build_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    return paths


def build_notes() -> None:
    lines = [
        "# QAOA Routing Final Presentation — Speaker Notes",
        "",
        "**Boundary:** course experiments only; one teaching-scale graph; ideal simulation; no quantum-advantage claim.",
        "",
    ]
    for number, spec in enumerate(SPECS, start=1):
        lines.extend((f"## Slide {number} — {spec.title}", "", spec.notes or "Backup slide.", ""))
    NOTES.write_text("\n".join(lines), encoding="utf-8")


def build_audit() -> None:
    sources = sorted({path.relative_to(ROOT) for spec in SPECS for path in spec.images})
    lines = [
        "# Presentation Data Audit",
        "",
        "The final deck is course-first and reads only tracked course figures. It does not read or present depth-110, recovery, robustness, or search-control campaigns as course results.",
        "",
        "| Claim | Frozen course source |",
        "| --- | --- |",
        "| 7 nodes, 14 edge variables, exact route cost 10 | `data/graph.json`; `src/graph.py` |",
        "| QUBO/Ising equality over 16,384 states | `tests/test_encoding.py` |",
        "| 20 feasible routes and structural p_feas=1 | `results/q2f_final_improvement/`; `tests/test_feasible_qaoa.py` |",
        "| GM-Th depth-3 median p_opt=0.998712 | `results/q2f_final_improvement/*/summary/aggregate_by_method_depth.csv` |",
        "| CVaR did not clearly improve depth-3 expectation | `results/q2f_course_extension/*/summary/aggregate_by_objective_depth.csv` |",
        "",
        "## Figure inputs",
        "",
    ]
    lines.extend(f"- `{path}`" for path in sources)
    lines.extend(("", "Generated outputs: PPTX, PDF, Notes, and this audit. Local rendered slide PNGs are ignored by Git."))
    AUDIT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    missing = [path for spec in SPECS for path in spec.images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing presentation sources: {missing}")
    OUT.mkdir(parents=True, exist_ok=True)
    build_pptx()
    build_notes()
    build_audit()
    slide_paths = build_pdf_and_validation()
    print(f"built {PPTX}")
    print(f"built {PDF}")
    print(f"validated {len(slide_paths)} slides from tracked course figures")


if __name__ == "__main__":
    main()
