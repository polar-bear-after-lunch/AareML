"""
Build AareML — Project Effort Log PDF with ReportLab.
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor, white, Color
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether,
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import os

# ── Register fonts ──────────────────────────────────────────────
pdfmetrics.registerFont(TTFont("DMSans-Bold", "/tmp/fonts/DMSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("Inter", "/tmp/fonts/Inter-Regular.ttf"))

# ── Palette ─────────────────────────────────────────────────────
TEAL_DARK  = HexColor("#0C4E54")
TEAL       = HexColor("#01696F")
TEAL_LIGHT = HexColor("#E0F0F1")
TEXT       = HexColor("#28251D")
MUTED      = HexColor("#7A7974")
BG_ALT     = HexColor("#F7F6F2")
BORDER     = HexColor("#D4D1CA")
WHITE      = white

# Category colours
CAT_PLANNING_RESEARCH = HexColor("#EDE5F5")   # light purple
CAT_PLANNING_WRITING  = HexColor("#DEEBF7")   # light blue
CAT_DATA_EDA          = HexColor("#E2F0D9")   # light green
CAT_MODELLING         = TEAL_LIGHT             # teal light
CAT_ENGINEERING       = HexColor("#FBE5D0")   # light orange
CAT_WRITING           = HexColor("#FFF8D6")   # light yellow
CAT_INFRASTRUCTURE    = HexColor("#EAEAEA")   # light grey

CATEGORY_COLORS = {
    "Planning & Research": CAT_PLANNING_RESEARCH,
    "Planning & Writing":  CAT_PLANNING_WRITING,
    "Data & EDA":          CAT_DATA_EDA,
    "Modelling":           CAT_MODELLING,
    "Engineering":         CAT_ENGINEERING,
    "Writing":             CAT_WRITING,
    "Infrastructure":      CAT_INFRASTRUCTURE,
}

# ── Page setup ──────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

OUTPUT = "/home/user/workspace/AareML-effort-log.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    title="AareML — Project Effort Log",
    author="Perplexity Computer",
    leftMargin=MARGIN,
    rightMargin=MARGIN,
    topMargin=MARGIN + 14 * mm,   # room for header bar
    bottomMargin=MARGIN + 10 * mm, # room for footer bar
)

# ── Styles ──────────────────────────────────────────────────────
title_style = ParagraphStyle(
    "Title",
    fontName="DMSans-Bold",
    fontSize=22,
    leading=26,
    textColor=TEAL_DARK,
    spaceAfter=4,
)
subtitle_style = ParagraphStyle(
    "Subtitle",
    fontName="Inter",
    fontSize=10,
    leading=13,
    textColor=MUTED,
    spaceAfter=2,
)
info_style = ParagraphStyle(
    "Info",
    fontName="Inter",
    fontSize=10,
    leading=13,
    textColor=TEXT,
    spaceAfter=10,
)
body_style = ParagraphStyle(
    "Body",
    fontName="Inter",
    fontSize=9,
    leading=12,
    textColor=TEXT,
)
body_small = ParagraphStyle(
    "BodySmall",
    fontName="Inter",
    fontSize=8,
    leading=10.5,
    textColor=TEXT,
)
heading2 = ParagraphStyle(
    "H2",
    fontName="DMSans-Bold",
    fontSize=13,
    leading=16,
    textColor=TEAL_DARK,
    spaceBefore=14,
    spaceAfter=6,
)
heading3 = ParagraphStyle(
    "H3",
    fontName="DMSans-Bold",
    fontSize=11,
    leading=14,
    textColor=TEAL_DARK,
    spaceBefore=10,
    spaceAfter=4,
)
summary_value = ParagraphStyle(
    "SummaryValue",
    fontName="DMSans-Bold",
    fontSize=18,
    leading=22,
    textColor=TEAL_DARK,
    alignment=1,
)
summary_label = ParagraphStyle(
    "SummaryLabel",
    fontName="Inter",
    fontSize=8,
    leading=10,
    textColor=MUTED,
    alignment=1,
)
notes_style = ParagraphStyle(
    "Notes",
    fontName="Inter",
    fontSize=9,
    leading=12,
    textColor=MUTED,
    leftIndent=6,
)
planned_style = ParagraphStyle(
    "Planned",
    fontName="Inter",
    fontSize=8.5,
    leading=11.5,
    textColor=MUTED,
)

# ── Data ────────────────────────────────────────────────────────
sessions = [
    (1, "17 Mar 2026", "Project scoping & dataset research", "Planning & Research", 4.0),
    (2, "5 Apr 2026",  "Project definition & proposal", "Planning & Writing", 2.0),
    (3, "5 Apr 2026",  "Data acquisition & EDA", "Data & EDA", 5.0),
    (4, "6 Apr 2026",  "Baseline models", "Modelling", 4.0),
    (5, "6 Apr 2026",  "LSTM implementation", "Modelling", 4.0),
    (6, "6 Apr 2026",  "Code improvements & refactoring", "Engineering", 4.0),
    (7, "6 Apr 2026",  "Multi-site & SHAP notebooks", "Modelling", 3.0),
    (8, "7 Apr 2026",  "Report writing", "Writing", 3.0),
    (9,  "7 Apr 2026",  "GitHub upload & environment setup", "Infrastructure", 1.5),
    (10, "8 Apr 2026",  "Cross-ecosystem transfer (lakes nb06, nb07)", "Modelling", 3.0),
    (11, "8 Apr 2026",  "USGS cross-continental transfer (nb08)", "Modelling", 3.0),
    (12, "9 Apr 2026",  "Canton Zurich analysis (nb09)", "Modelling", 3.0),
    (13, "9 Apr 2026",  "Swiss lakes LSTM (nb10)", "Modelling", 3.0),
    (14, "10 Apr 2026", "Report writing v1.0-v1.10", "Writing", 4.0),
    (15, "11 Apr 2026", "Full code audit & bug fixes (3-model review)", "Engineering", 5.0),
    (16, "12 Apr 2026", "UBELIX HPC setup & job submission", "Infrastructure", 3.0),
    (17, "13 Apr 2026", "Results integration & report v1.11-v1.15", "Writing", 3.0),
    (18, "14 Apr 2026", "Presentation design (PPTX, fish boxes)", "Writing", 3.0),
    (19, "15 Apr 2026", "UBELIX reruns (seed bug, EA-LSTM scaler fix)", "Modelling", 4.0),
    (20, "16 Apr 2026", "Report shortening v1.19 + TOC v1.20", "Writing", 2.5),
    (21, "17 Apr 2026", "Fact-check & corrections v1.21", "Writing", 2.5),
    (22, "18 Apr 2026", "Professor feedback integration v1.22", "Writing", 2.0),
    (23, "19 Apr 2026", "Innovation Sandbox application draft", "Writing", 3.0),
    (24, "20 Apr 2026", "nb11 Ablation study", "Modelling", 2.0),
    (25, "21 Apr 2026", "nb12 Error analysis + nb13 Seasonal analysis", "Modelling", 2.5),
    (26, "22 Apr 2026", "nb14 AR baseline + professor feedback Q&A", "Modelling", 2.0),
    (27, "23 Apr 2026", "nb15 Scientific rigor (Granger, ACF, threshold)", "Modelling", 2.5),
    (28, "1 May 2026",  "Thiago Nascimento feedback — EA-LSTM updates", "Modelling", 3.0),
    (29, "5 May 2026",  "CAMELS-CH static attributes integration", "Modelling", 2.5),
    (30, "6 May 2026",  "EA-LSTM temperature (nb04b) + 86-gauge expansion", "Modelling", 3.0),
    (31, "7 May 2026",  "nb04c EC precipitation proxy", "Modelling", 2.0),
    (32, "8 May 2026",  "Scientific restraint rewrites v1.23", "Writing", 2.0),
    (33, "11 May 2026", "UBELIX reruns (Q bug fix, nb04/04b/15)", "Infrastructure", 3.0),
    (34, "12 May 2026", "Presentation training Day 1-3 (LSTM, training, transfer)", "Writing", 2.5),
    (35, "13 May 2026", "Ridge zero-shot + nb16 cross-validation design", "Modelling", 3.0),
    (36, "14 May 2026", "Presentation script + slides update", "Writing", 2.0),
    (37, "15 May 2026", "CAMELS-CH base attributes (Höge 2023) integration", "Modelling", 3.0),
    (38, "16 May 2026", "Literature review (Zhi 2023, Padrón 2025, Baste 2025)", "Planning & Research", 2.5),
    (39, "16 May 2026", "Innovation Sandbox v3 (Eawag partnership strategy)", "Planning & Writing", 2.0),
    (40, "16 May 2026", "Short lookback ablation (4-7 days) + report v1.25", "Modelling", 2.0),
]

total_hours = sum(s[4] for s in sessions)
budget = 120.0
remaining = budget - total_hours
pct = total_hours / budget * 100

# Build cumulative column
cumulative = []
running = 0.0
for s in sessions:
    running += s[4]
    cumulative.append(running)

# ── Header / Footer ────────────────────────────────────────────
def header_footer(canvas_obj, doc_obj):
    canvas_obj.saveState()
    w, h = A4

    # Header bar
    bar_h = 11 * mm
    y_bar = h - MARGIN + 2 * mm
    canvas_obj.setFillColor(TEAL_DARK)
    canvas_obj.rect(MARGIN, y_bar, w - 2 * MARGIN, bar_h, fill=1, stroke=0)

    # Header text
    canvas_obj.setFillColor(WHITE)
    canvas_obj.setFont("DMSans-Bold", 9)
    canvas_obj.drawString(MARGIN + 6 * mm, y_bar + 3.2 * mm, "AareML")

    canvas_obj.setFont("Inter", 8)
    canvas_obj.drawRightString(
        w - MARGIN - 6 * mm, y_bar + 3.2 * mm,
        "CAS Advanced Machine Learning — University of Bern",
    )

    # Footer bar
    footer_h = 8 * mm
    y_footer = MARGIN - 6 * mm
    canvas_obj.setFillColor(BORDER)
    canvas_obj.rect(MARGIN, y_footer, w - 2 * MARGIN, footer_h, fill=1, stroke=0)

    canvas_obj.setFillColor(MUTED)
    canvas_obj.setFont("Inter", 7.5)
    canvas_obj.drawString(MARGIN + 6 * mm, y_footer + 2.4 * mm, "Effort Log")
    canvas_obj.drawRightString(
        w - MARGIN - 6 * mm, y_footer + 2.4 * mm,
        f"Page {doc_obj.page}",
    )

    canvas_obj.restoreState()

# ── Build story ─────────────────────────────────────────────────
story = []

# Title section
story.append(Paragraph("AareML — Project Effort Log", title_style))
story.append(Paragraph(
    "CAS in Advanced Machine Learning · University of Bern", subtitle_style
))
story.append(Paragraph(
    "Total budget: 120 hours (4 ECTS) · Deadline: 15 June 2026", info_style
))
story.append(Spacer(1, 6))

# ── Summary box ─────────────────────────────────────────────────
summary_data = [
    [
        Paragraph(f"{total_hours:.1f}", summary_value),
        Paragraph(f"{remaining:.1f}", summary_value),
        Paragraph(f"{pct:.0f}%", summary_value),
    ],
    [
        Paragraph("Hours Logged", summary_label),
        Paragraph("Hours Remaining", summary_label),
        Paragraph("Complete", summary_label),
    ],
]

avail_w = PAGE_W - 2 * MARGIN
col_w_summary = avail_w / 3.0

summary_table = Table(summary_data, colWidths=[col_w_summary] * 3, rowHeights=[30, 16])
summary_table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, -1), TEAL_LIGHT),
    ("BOX",           (0, 0), (-1, -1), 1, TEAL),
    ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",    (0, 0), (-1, 0), 10),
    ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
    ("TOPPADDING",    (0, 1), (-1, 1), 0),
    ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
    # Vertical dividers between columns
    ("LINEAFTER",     (0, 0), (1, -1), 0.5, TEAL),
]))
story.append(summary_table)
story.append(Spacer(1, 14))

# ── Main effort table ──────────────────────────────────────────
story.append(Paragraph("Effort Log", heading2))

# Column widths
col_widths = [
    48,   # Session
    62,   # Date
    avail_w - 48 - 62 - 82 - 48 - 60,  # Activity (flexible)
    82,   # Category
    48,   # Hours
    60,   # Cumulative
]

# Header row
header_row = [
    Paragraph('<font name="DMSans-Bold" color="white" size="8">Session</font>', body_style),
    Paragraph('<font name="DMSans-Bold" color="white" size="8">Date</font>', body_style),
    Paragraph('<font name="DMSans-Bold" color="white" size="8">Activity</font>', body_style),
    Paragraph('<font name="DMSans-Bold" color="white" size="8">Category</font>', body_style),
    Paragraph('<font name="DMSans-Bold" color="white" size="8">Hours</font>', body_style),
    Paragraph('<font name="DMSans-Bold" color="white" size="8">Cumul.</font>', body_style),
]

table_data = [header_row]
for i, (sess, date, activity, category, hours) in enumerate(sessions):
    table_data.append([
        Paragraph(f'<font name="DMSans-Bold" size="8">{sess}</font>', body_small),
        Paragraph(f'<font name="Inter" size="8">{date}</font>', body_small),
        Paragraph(f'<font name="Inter" size="8">{activity}</font>', body_small),
        Paragraph(f'<font name="Inter" size="8">{category}</font>', body_small),
        Paragraph(f'<font name="DMSans-Bold" size="8">{hours:.1f}</font>', body_small),
        Paragraph(f'<font name="Inter" size="8">{cumulative[i]:.1f}</font>', body_small),
    ])

# Totals row
table_data.append([
    Paragraph('<font name="DMSans-Bold" size="8"></font>', body_small),
    Paragraph('<font name="DMSans-Bold" size="8"></font>', body_small),
    Paragraph('<font name="DMSans-Bold" size="8">TOTAL</font>', body_small),
    Paragraph('<font name="DMSans-Bold" size="8"></font>', body_small),
    Paragraph(f'<font name="DMSans-Bold" size="9">{total_hours:.1f}</font>', body_small),
    Paragraph(f'<font name="DMSans-Bold" size="9">{total_hours:.1f}</font>', body_small),
])

effort_table = Table(table_data, colWidths=col_widths, repeatRows=1)

# Build style commands
style_cmds = [
    # Header
    ("BACKGROUND",    (0, 0), (-1, 0), TEAL_DARK),
    ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
    ("FONTNAME",      (0, 0), (-1, 0), "DMSans-Bold"),
    ("FONTSIZE",      (0, 0), (-1, 0), 8),

    # Grid
    ("GRID",          (0, 0), (-1, -1), 0.4, BORDER),
    ("BOX",           (0, 0), (-1, -1), 0.6, BORDER),

    # Alignment
    ("ALIGN",         (0, 0), (0, -1), "CENTER"),  # Session col centered
    ("ALIGN",         (4, 0), (5, -1), "CENTER"),   # Hours & Cumul centered
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),

    # Padding
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),

    # Totals row
    ("BACKGROUND",    (0, -1), (-1, -1), TEAL_LIGHT),
    ("FONTNAME",      (0, -1), (-1, -1), "DMSans-Bold"),
    ("LINEABOVE",     (0, -1), (-1, -1), 1.2, TEAL_DARK),
]

# Alternating row backgrounds (skip header row 0 and totals row -1)
for i in range(1, len(table_data) - 1):
    bg = WHITE if (i - 1) % 2 == 0 else BG_ALT
    style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))

# Category column color coding
for i, (sess, date, activity, category, hours) in enumerate(sessions):
    row_idx = i + 1  # +1 for header
    cat_color = CATEGORY_COLORS.get(category)
    if cat_color:
        style_cmds.append(("BACKGROUND", (3, row_idx), (3, row_idx), cat_color))

effort_table.setStyle(TableStyle(style_cmds))
story.append(effort_table)
story.append(Spacer(1, 16))

# ── Planned work section ───────────────────────────────────────
story.append(Paragraph("Planned Work", heading3))

planned_items = [
    ("Run notebook 03 full Optuna tuning", "3 h"),
    ("Run notebook 04 multi-site evaluation", "2 h"),
    ("Run notebook 05 SHAP attribution", "2 h"),
    ("Update report with full LSTM + multi-site + SHAP results", "4 h"),
    ("Discussion sections 6.3–6.4", "2 h"),
    ("Final polish, figures, proofreading", "3 h"),
    ("GitHub final push + submission", "1 h"),
    ("Buffer / unexpected issues", "10 h"),
]

planned_header = [
    Paragraph('<font name="DMSans-Bold" color="white" size="8">Task</font>', body_small),
    Paragraph('<font name="DMSans-Bold" color="white" size="8">Est.</font>', body_small),
]
planned_data = [planned_header]
for task, est in planned_items:
    planned_data.append([
        Paragraph(f'<font name="Inter" size="8" color="#7A7974">{task}</font>', planned_style),
        Paragraph(f'<font name="Inter" size="8" color="#7A7974">{est}</font>', planned_style),
    ])
# Total row
planned_data.append([
    Paragraph('<font name="DMSans-Bold" size="8" color="#7A7974">Total planned remaining</font>', planned_style),
    Paragraph('<font name="DMSans-Bold" size="8" color="#7A7974">27 h + buffer</font>', planned_style),
])

planned_table = Table(planned_data, colWidths=[avail_w - 70, 70])
planned_style_cmds = [
    ("BACKGROUND",    (0, 0), (-1, 0), TEAL),
    ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
    ("GRID",          (0, 0), (-1, -1), 0.3, BORDER),
    ("BOX",           (0, 0), (-1, -1), 0.5, BORDER),
    ("ALIGN",         (1, 0), (1, -1), "CENTER"),
    ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ("BACKGROUND",    (0, -1), (-1, -1), BG_ALT),
    ("LINEABOVE",     (0, -1), (-1, -1), 0.8, MUTED),
]
# Alternate rows
for i in range(1, len(planned_data) - 1):
    bg = WHITE if (i - 1) % 2 == 0 else BG_ALT
    planned_style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))

planned_table.setStyle(TableStyle(planned_style_cmds))
story.append(planned_table)
story.append(Spacer(1, 18))

# ── Notes section ──────────────────────────────────────────────
story.append(Paragraph("Notes", heading3))

# Draw a light bordered area with some placeholder lines
notes_lines = [
    "• Hours are rounded to the nearest 0.5 h.",
    "• Sessions may span multiple calendar days; the date shown is the primary work date.",
    "• Budget includes all project work: planning, coding, writing, and infrastructure.",
    "• ECTS conversion: 4 ECTS = 120 hours (30 h / ECTS).",
]
for line in notes_lines:
    story.append(Paragraph(line, notes_style))
    story.append(Spacer(1, 2))

# ── Build PDF ──────────────────────────────────────────────────
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

file_size = os.path.getsize(OUTPUT)
print(f"PDF created: {OUTPUT}")
print(f"File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
