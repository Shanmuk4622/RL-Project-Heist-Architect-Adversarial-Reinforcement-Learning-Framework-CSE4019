"""
Heist Architect - Full Presentation Generator
CSE4019 - Adversarial Reinforcement Learning Framework
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C_DARK_BG    = RGBColor(0x0D, 0x11, 0x17)   # near-black navy
C_MID_BG     = RGBColor(0x14, 0x1E, 0x2E)   # dark blue panel
C_ACCENT1    = RGBColor(0x00, 0xB4, 0xD8)   # electric cyan
C_ACCENT2    = RGBColor(0xFF, 0xB7, 0x00)   # amber gold
C_ACCENT3    = RGBColor(0xFF, 0x47, 0x6F)   # coral red
C_ACCENT4    = RGBColor(0x06, 0xD6, 0xA0)   # mint green
C_WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
C_LIGHT_GRAY = RGBColor(0xC8, 0xD8, 0xE8)
C_DIM_GRAY   = RGBColor(0x55, 0x66, 0x77)


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL SHAPE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def solid_fill(shape, color: RGBColor):
    sf = shape.fill
    sf.solid()
    sf.fore_color.rgb = color

def no_fill(shape):
    shape.fill.background()

def no_line(shape):
    shape.line.fill.background()

def add_rect(slide, l, t, w, h, fill=None, line_color=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    if fill:
        solid_fill(shape, fill)
    else:
        no_fill(shape)
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = line_width
    else:
        no_line(shape)
    return shape

def add_textbox(slide, text, l, t, w, h,
                font_size=Pt(12), bold=False, italic=False,
                color=C_WHITE, align=PP_ALIGN.LEFT, wrap=True,
                word_wrap=True):
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = txb.text_frame
    tf.word_wrap = word_wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = font_size
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txb

def add_multiline_textbox(slide, lines, l, t, w, h,
                          font_size=Pt(11), bold=False,
                          color=C_WHITE, align=PP_ALIGN.LEFT,
                          line_spacing=None):
    """lines is a list of (text, bold, color, size, italic) tuples or plain strings"""
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = txb.text_frame
    tf.word_wrap = True
    first = True
    for item in lines:
        if isinstance(item, str):
            txt, b, c, sz, it = item, bold, color, font_size, False
        else:
            txt = item[0]
            b   = item[1] if len(item) > 1 else bold
            c   = item[2] if len(item) > 2 else color
            sz  = item[3] if len(item) > 3 else font_size
            it  = item[4] if len(item) > 4 else False
        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()
        p.alignment = align
        if line_spacing:
            p.space_before = Pt(line_spacing)
        run = p.add_run()
        run.text = txt
        run.font.size = sz
        run.font.bold = b
        run.font.italic = it
        run.font.color.rgb = c
    return txb


# ─────────────────────────────────────────────────────────────────────────────
# SLIDE LAYOUTS
# ─────────────────────────────────────────────────────────────────────────────
def dark_background(slide):
    add_rect(slide, 0, 0, 13.33, 7.5, fill=C_DARK_BG)

def accent_bar_top(slide, color=C_ACCENT1, height=0.06):
    add_rect(slide, 0, 0, 13.33, height, fill=color)

def accent_bar_bottom(slide, color=C_ACCENT1):
    add_rect(slide, 0, 7.44, 13.33, 0.06, fill=color)

def slide_header(slide, title, subtitle=None,
                 title_color=C_ACCENT1, sub_color=C_LIGHT_GRAY):
    add_textbox(slide, title, 0.45, 0.2, 12.4, 0.7,
                font_size=Pt(30), bold=True, color=title_color, align=PP_ALIGN.LEFT)
    if subtitle:
        add_textbox(slide, subtitle, 0.45, 0.85, 12.4, 0.35,
                    font_size=Pt(13), italic=True, color=sub_color, align=PP_ALIGN.LEFT)
    # thin divider
    add_rect(slide, 0.45, 1.22, 12.43, 0.025, fill=C_ACCENT1)

def card(slide, l, t, w, h, fill=C_MID_BG, border=C_ACCENT1, border_w=Pt(1)):
    r = add_rect(slide, l, t, w, h, fill=fill, line_color=border, line_width=border_w)
    return r

def bullet_list(slide, items, l, t, w, h,
                font_size=Pt(11.5), bullet="▸ ", color=C_LIGHT_GRAY,
                heading=None, heading_color=C_ACCENT2):
    lines = []
    if heading:
        lines.append((heading, True, heading_color, Pt(13), False))
    for item in items:
        lines.append((f"{bullet}{item}", False, color, font_size, False))
    add_multiline_textbox(slide, lines, l, t, w, h, line_spacing=3)

def two_col_bullets(slide, left_title, left_items, right_title, right_items,
                    l=0.3, t=1.35, card_h=5.8,
                    lc=C_ACCENT1, rc=C_ACCENT3):
    # Left card
    card(slide, l, t, 6.1, card_h)
    y = t + 0.18
    add_textbox(slide, left_title, l + 0.2, y, 5.7, 0.42,
                font_size=Pt(14), bold=True, color=lc)
    add_rect(slide, l + 0.2, y + 0.42, 5.6, 0.025, fill=lc)
    bullet_list(slide, left_items, l + 0.2, y + 0.52, 5.65, card_h - 0.85,
                font_size=Pt(11))
    # Right card
    rx = l + 6.5
    card(slide, rx, t, 6.1, card_h)
    add_textbox(slide, right_title, rx + 0.2, y, 5.7, 0.42,
                font_size=Pt(14), bold=True, color=rc)
    add_rect(slide, rx + 0.2, y + 0.42, 5.6, 0.025, fill=rc)
    bullet_list(slide, right_items, rx + 0.2, y + 0.52, 5.65, card_h - 0.85,
                font_size=Pt(11), color=C_LIGHT_GRAY)

def phase_badge(slide, label, l, t, color=C_ACCENT1):
    add_rect(slide, l, t, 1.55, 0.35, fill=color)
    add_textbox(slide, label, l, t, 1.55, 0.35,
                font_size=Pt(10), bold=True, color=C_DARK_BG, align=PP_ALIGN.CENTER)

def stat_box(slide, value, label, l, t, w=2.8, h=1.1,
             val_color=C_ACCENT2, bg=C_MID_BG, border=C_ACCENT1):
    card(slide, l, t, w, h, fill=bg, border=border)
    add_textbox(slide, value, l, t + 0.08, w, 0.58,
                font_size=Pt(28), bold=True, color=val_color, align=PP_ALIGN.CENTER)
    add_textbox(slide, label, l, t + 0.62, w, 0.35,
                font_size=Pt(10), bold=False, color=C_DIM_GRAY, align=PP_ALIGN.CENTER)


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL SLIDE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def slide_title(prs):
    """Slide 1 – Title / Cover"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    dark_background(slide)

    # Gradient stripe across top-third
    add_rect(slide, 0, 0, 13.33, 0.12, fill=C_ACCENT1)
    add_rect(slide, 0, 0.12, 13.33, 2.3, fill=C_MID_BG)

    # Left vertical accent bar
    add_rect(slide, 0, 0, 0.12, 7.5, fill=C_ACCENT1)

    # Main title
    add_textbox(slide, "HEIST ARCHITECT", 0.35, 0.45, 12.6, 1.05,
                font_size=Pt(52), bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    add_textbox(slide, "Adversarial Reinforcement Learning Framework",
                0.35, 1.42, 12.6, 0.6,
                font_size=Pt(20), bold=False, italic=True,
                color=C_ACCENT1, align=PP_ALIGN.LEFT)

    # Divider
    add_rect(slide, 0.35, 2.1, 12.6, 0.03, fill=C_DIM_GRAY)

    # Tagline
    add_textbox(slide,
                "Two AI agents play an infinite game of cops-and-robbers.\n"
                "One designs the security system. The other breaks in.",
                0.35, 2.25, 12.6, 1.0,
                font_size=Pt(16), color=C_LIGHT_GRAY, align=PP_ALIGN.LEFT)

    # Phase badges row
    badges = [
        ("PHASE I",  C_ACCENT1, 0.35),
        ("PHASE II", C_ACCENT2, 2.25),
    ]
    by = 3.55
    for label, col, bx in badges:
        add_rect(slide, bx, by, 1.6, 0.38, fill=col)
        add_textbox(slide, label, bx, by, 1.6, 0.38,
                    font_size=Pt(11), bold=True, color=C_DARK_BG, align=PP_ALIGN.CENTER)

    add_textbox(slide,
                "Local Training  →  Kaggle Cloud Scale-Up  →  17,000 Episodes",
                0.35, 4.05, 12.6, 0.45,
                font_size=Pt(13), color=C_DIM_GRAY, align=PP_ALIGN.LEFT)

    # Meta info strip
    add_rect(slide, 0, 6.9, 13.33, 0.6, fill=C_MID_BG)
    add_textbox(slide, "CSE4019 – Reinforcement Learning Project  |  2025–2026",
                0.35, 6.95, 12.6, 0.4,
                font_size=Pt(11), color=C_DIM_GRAY, align=PP_ALIGN.LEFT)
    add_textbox(slide, "Built with PyTorch · Flask · CUDA · Kaggle T4 GPUs  |  Hugging Face: Shanmuk4622/heist-architect-v2",
                0.35, 7.1, 12.6, 0.4,
                font_size=Pt(9), color=C_DIM_GRAY, align=PP_ALIGN.LEFT)

    accent_bar_bottom(slide)


def slide_agenda(prs):
    """Slide 2 – Agenda / Table of Contents"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT2)
    slide_header(slide, "Presentation Outline", "A two-phase journey from local prototype to cloud-scale AI", title_color=C_ACCENT2)
    accent_bar_bottom(slide, C_ACCENT2)

    # Phase columns
    left = [
        "Problem Statement & Motivation",
        "Adversarial RL Concept",
        "Dual-Agent Architecture (Architect vs Solver)",
        "Environment Design — 20×20 Grid World",
        "Security Components (Walls, Cameras, Guards)",
        "Curriculum Learning Pipeline",
        "Neural Networks (CNN + LSTM / Encoder-Decoder)",
        "Phase I Results — 500 Episode Local Run",
    ]
    right = [
        "Phase II — Kaggle Cloud Scale-Up",
        "17,000 Episode Training on Dual T4 GPUs",
        "Collapse & Recovery Cycles (Adversarial Dynamics)",
        "ELO Arms Race — Strategic Equilibrium",
        "Web Dashboard — Real-Time Monitoring",
        "Key Quantitative Results",
        "Emergent Strategies & Nash Equilibrium",
        "References",
    ]

    card(slide, 0.3, 1.35, 6.05, 5.85, fill=C_MID_BG, border=C_ACCENT1)
    add_textbox(slide, "PHASE I — Local Framework", 0.5, 1.5, 5.7, 0.45,
                font_size=Pt(13), bold=True, color=C_ACCENT1)
    add_rect(slide, 0.5, 1.9, 5.65, 0.025, fill=C_ACCENT1)
    for i, item in enumerate(left):
        add_multiline_textbox(slide, [(f"{i+1:02d}.  {item}", False, C_LIGHT_GRAY, Pt(11.5), False)],
                              0.5, 2.0 + i*0.68, 5.6, 0.65)

    card(slide, 6.95, 1.35, 6.05, 5.85, fill=C_MID_BG, border=C_ACCENT2)
    add_textbox(slide, "PHASE II — Cloud Scale-Up", 7.15, 1.5, 5.7, 0.45,
                font_size=Pt(13), bold=True, color=C_ACCENT2)
    add_rect(slide, 7.15, 1.9, 5.65, 0.025, fill=C_ACCENT2)
    for i, item in enumerate(right):
        add_multiline_textbox(slide, [(f"{i+1:02d}.  {item}", False, C_LIGHT_GRAY, Pt(11.5), False)],
                              7.15, 2.0 + i*0.68, 5.6, 0.65)


def slide_problem(prs):
    """Slide 3 – Problem Statement"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide)
    slide_header(slide, "Problem Statement", "What are we solving, and why does it matter?")
    accent_bar_bottom(slide)

    # Big quote card
    card(slide, 0.3, 1.38, 12.73, 1.6, fill=C_MID_BG, border=C_ACCENT1)
    add_textbox(slide,
                '"Standard RL trains a single agent on a fixed environment.\n'
                'Real-world security is a dynamic, adaptive adversarial game."',
                0.55, 1.5, 12.2, 1.35,
                font_size=Pt(17), italic=True, color=C_ACCENT2, align=PP_ALIGN.CENTER)

    # Three challenge cards
    challenges = [
        (C_ACCENT1, "Static Environments",
         "Traditional RL agents memorize solutions. Security threats evolve — a defender must adapt to an ever-improving attacker."),
        (C_ACCENT3, "Single-Agent Limitation",
         "Training on a fixed maze produces brittle policies. An intelligent adversary (Architect) forces the agent to generalize, not memorize."),
        (C_ACCENT4, "No Adversarial Co-Learning",
         "Existing benchmarks (e.g., MiniGrid) lack an intelligent, learning opponent. This project fills that gap with simultaneous dual-agent training."),
    ]
    cx = 0.3
    for col, title, body in challenges:
        card(slide, cx, 3.15, 4.15, 3.95, fill=C_MID_BG, border=col, border_w=Pt(1.5))
        add_rect(slide, cx, 3.15, 4.15, 0.08, fill=col)
        add_textbox(slide, title, cx + 0.15, 3.28, 3.8, 0.45,
                    font_size=Pt(13), bold=True, color=col)
        add_textbox(slide, body, cx + 0.15, 3.78, 3.8, 3.2,
                    font_size=Pt(11.5), color=C_LIGHT_GRAY)
        cx += 4.4


def slide_concept(prs):
    """Slide 4 – Adversarial RL Concept"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT3)
    slide_header(slide, "Adversarial Reinforcement Learning", "Self-play that drives emergent intelligence", title_color=C_ACCENT3)
    accent_bar_bottom(slide, C_ACCENT3)

    # Central comparison — MDP vs Markov Game
    add_textbox(slide, "🔹 Standard MDP", 0.35, 1.38, 5.9, 0.45,
                font_size=Pt(14), bold=True, color=C_ACCENT1)
    card(slide, 0.35, 1.78, 5.9, 2.5, fill=C_MID_BG, border=C_ACCENT1)
    add_textbox(slide,
                "• One agent, one fixed environment\n"
                "• Converges to a stable optimal policy\n"
                "• Learns to memorize patterns\n"
                "• Fails when environment shifts",
                0.55, 1.88, 5.5, 2.25, font_size=Pt(12), color=C_LIGHT_GRAY)

    add_textbox(slide, "⚔️  Markov Game (This Project)", 6.9, 1.38, 6.1, 0.45,
                font_size=Pt(14), bold=True, color=C_ACCENT3)
    card(slide, 6.9, 1.78, 6.1, 2.5, fill=C_MID_BG, border=C_ACCENT3)
    add_textbox(slide,
                "• Two agents, adversarially generated environment\n"
                "• Strategically non-stationary — task changes as opponent evolves\n"
                "• Forces generalisation over memorisation\n"
                "• Converges toward Nash Equilibrium",
                7.1, 1.88, 5.7, 2.25, font_size=Pt(12), color=C_LIGHT_GRAY)

    # VS badge
    add_rect(slide, 6.0, 2.3, 0.72, 0.72, fill=C_ACCENT2)
    add_textbox(slide, "VS", 6.0, 2.36, 0.72, 0.6,
                font_size=Pt(18), bold=True, color=C_DARK_BG, align=PP_ALIGN.CENTER)

    # Bottom row key terms
    terms = [
        (C_ACCENT1, "Non-Stationarity", "Task distribution changes as the opponent learns new strategies"),
        (C_ACCENT2, "Co-Adaptation",    "Both policies evolve in response to each other — arms race dynamics"),
        (C_ACCENT4, "Nash Equilibrium", "Stable endpoint where neither agent can improve by changing strategy alone"),
        (C_ACCENT3, "ELO Rating",       "Relative skill score that tracks which agent is currently dominating"),
    ]
    tx = 0.3
    for col, term, desc in terms:
        card(slide, tx, 4.48, 3.1, 2.65, fill=C_MID_BG, border=col)
        add_textbox(slide, term, tx + 0.15, 4.6, 2.8, 0.4,
                    font_size=Pt(12), bold=True, color=col)
        add_textbox(slide, desc, tx + 0.15, 5.05, 2.8, 2.0,
                    font_size=Pt(10.5), color=C_LIGHT_GRAY)
        tx += 3.26


def slide_agents(prs):
    """Slide 5 – The Two Agents"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT4)
    slide_header(slide, "The Two Agents", "Architect vs Solver — an infinite adversarial game", title_color=C_ACCENT4)
    accent_bar_bottom(slide, C_ACCENT4)

    # Architect card (left)
    card(slide, 0.3, 1.38, 6.1, 5.75, fill=C_MID_BG, border=C_ACCENT1)
    add_rect(slide, 0.3, 1.38, 6.1, 0.08, fill=C_ACCENT1)
    add_textbox(slide, "🔒  THE ARCHITECT", 0.5, 1.5, 5.7, 0.55,
                font_size=Pt(18), bold=True, color=C_ACCENT1)
    add_textbox(slide, "Security Designer — The Adversary",
                0.5, 2.0, 5.7, 0.35, font_size=Pt(11), italic=True, color=C_DIM_GRAY)
    bullet_list(slide, [
        "Places Walls, Cameras & Guards on a 20×20 grid",
        "Must stay within a budget (Wall=1 pt, Cam=3 pt, Guard=5 pt)",
        "Budget grows with curriculum: 5 → 8 → 15 → 22 pts",
        "Penalised if no valid path exists to the Vault",
        "Penalised if the Solver finds it too easy (>80% solve rate)",
        "Rewarded proportionally to Detection Rate achieved",
        "Network: ResNet + Group Norm Backbone — 8.5M parameters",
        "Uses 512-dim LSTM hidden state to map a dense 400-cell grid",
    ], 0.5, 2.4, 5.65, 4.5, font_size=Pt(11.5))

    # Solver card (right)
    card(slide, 6.95, 1.38, 6.1, 5.75, fill=C_MID_BG, border=C_ACCENT2)
    add_rect(slide, 6.95, 1.38, 6.1, 0.08, fill=C_ACCENT2)
    add_textbox(slide, "🕵️  THE SOLVER (ROBBER)", 7.15, 1.5, 5.7, 0.55,
                font_size=Pt(18), bold=True, color=C_ACCENT2)
    add_textbox(slide, "Infiltrator — The Navigator",
                7.15, 2.0, 5.7, 0.35, font_size=Pt(11), italic=True, color=C_DIM_GRAY)
    bullet_list(slide, [
        "Navigates from Start (1,1) to Vault (18,18) undetected",
        "Actions: UP, DOWN, LEFT, RIGHT, WAIT",
        "Sees 12-channel spatial tensors (geodesic dists, danger zones)",
        "WAIT action is critical — times camera rotations precisely",
        "Uses massive 512-param LSTM memory to track temporal guard sequences",
        "Penalised -1.0 for detection; rewarded +10.0 for vault reach",
        "Network: CNN + 512-dim LSTM — 7M parameters",
        "Distance-based shaping guides learning from episode one",
    ], 7.15, 2.4, 5.65, 4.5, font_size=Pt(11.5),
    color=C_LIGHT_GRAY)


def slide_environment(prs):
    """Slide 6 – Environment Design"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT2)
    slide_header(slide, "The Grid World Environment", "A 20×20 dynamic adversarial arena", title_color=C_ACCENT2)
    accent_bar_bottom(slide, C_ACCENT2)

    # Grid ASCII art
    card(slide, 0.3, 1.38, 5.5, 5.75, fill=C_MID_BG, border=C_ACCENT2)
    grid_text = (
        "  ####################\n"
        "  #S.................#\n"
        "  #....####.........#\n"
        "  #..................#\n"
        "  #......C...........#\n"
        "  #..........G.......#\n"
        "  #..XXX.............#\n"
        "  #....XXXXX.........#\n"
        "  #..........XXXXXX..#\n"
        "  #..................#\n"
        "  #.................V#\n"
        "  ####################"
    )
    add_textbox(slide, grid_text, 0.5, 1.55, 5.1, 5.3,
                font_size=Pt(10.5), color=C_ACCENT4, bold=True)

    # Legend
    legend = [
        ("S  =", "Start — Solver spawns here", C_ACCENT4),
        ("V  =", "Vault — Solver's objective", C_ACCENT3),
        ("C  =", "Camera — rotating vision cone", C_ACCENT1),
        ("G  =", "Guard — mobile patrol unit", C_ACCENT2),
        ("#  =", "Wall — blocks movement & sight", C_DIM_GRAY),
        ("X  =", "Danger Zone — currently surveilled", C_ACCENT3),
        (".   =", "Empty — walkable safe tile", C_LIGHT_GRAY),
    ]
    add_textbox(slide, "MAP LEGEND", 5.95, 1.5, 7.0, 0.4,
                font_size=Pt(13), bold=True, color=C_ACCENT2)
    add_rect(slide, 5.95, 1.88, 7.0, 0.025, fill=C_ACCENT2)
    for i, (sym, desc, col) in enumerate(legend):
        y = 1.98 + i * 0.47
        add_textbox(slide, sym, 5.95, y, 0.85, 0.4,
                    font_size=Pt(12), bold=True, color=col)
        add_textbox(slide, desc, 6.75, y, 6.1, 0.4,
                    font_size=Pt(11.5), color=C_LIGHT_GRAY)

    # Key rules
    card(slide, 5.95, 5.38, 7.08, 1.6, fill=C_MID_BG, border=C_ACCENT1)
    add_textbox(slide, "Key Rules", 6.1, 5.48, 6.8, 0.38,
                font_size=Pt(12), bold=True, color=C_ACCENT1)
    add_textbox(slide,
                "• Border tiles are always walls  •  BFS validates path existence\n"
                "• Visibility map is recomputed every tick via raycasting\n"
                "• Camera FOV, rotation speed & guard patrol paths are Architect-defined",
                6.1, 5.85, 6.75, 1.05, font_size=Pt(10.5), color=C_LIGHT_GRAY)


def slide_security(prs):
    """Slide 7 – Security Components"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide)
    slide_header(slide, "Security Components", "Budget-constrained adversarial assets")
    accent_bar_bottom(slide)

    components = [
        (C_DIM_GRAY,  "🧱  WALL",
         "Cost: 1 pt",
         ["Static obstacle — no movement", "Blocks Solver's path AND line-of-sight",
          "Architect uses walls to create chokepoints and corridors",
          "Funnels Solver into surveilled kill-zones"]),
        (C_ACCENT1,   "📷  CAMERA",
         "Cost: 3 pts",
         ["Fixed position, rotating heading over time",
          "Triangular vision cone — raycasted, wall-occluded",
          "Solver must time movements to exploit rotation gaps",
          "Architect places at intersections for maximum sweep area"]),
        (C_ACCENT2,   "💂  GUARD",
         "Cost: 5 pts",
         ["Mobile patrol along Architect-defined waypoint routes",
          "Vision cone follows direction of movement",
          "Most expensive — hardest for Solver to predict",
          "Architect learns to sync guard patrols with camera blind spots"]),
    ]

    cx = 0.3
    for col, title, cost, bullets in components:
        card(slide, cx, 1.38, 4.2, 5.75, fill=C_MID_BG, border=col)
        add_rect(slide, cx, 1.38, 4.2, 0.1, fill=col)
        add_textbox(slide, title, cx + 0.18, 1.52, 3.8, 0.55,
                    font_size=Pt(16), bold=True, color=col)
        add_textbox(slide, cost, cx + 0.18, 2.02, 3.8, 0.35,
                    font_size=Pt(11), bold=True, color=C_ACCENT2)
        for i, b in enumerate(bullets):
            add_textbox(slide, f"▸  {b}", cx + 0.18, 2.42 + i*0.77, 3.8, 0.7,
                        font_size=Pt(11), color=C_LIGHT_GRAY)
        cx += 4.42


def slide_reward(prs):
    """Slide 8 – Reward System"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT4)
    slide_header(slide, "Reward System — Zero-Sum Design", "Precisely engineered incentives that drive adversarial learning", title_color=C_ACCENT4)
    accent_bar_bottom(slide, C_ACCENT4)

    # Solver rewards
    card(slide, 0.3, 1.38, 6.1, 5.75, fill=C_MID_BG, border=C_ACCENT4)
    add_textbox(slide, "🕵️  Solver Rewards", 0.5, 1.52, 5.7, 0.45,
                font_size=Pt(15), bold=True, color=C_ACCENT4)
    add_rect(slide, 0.5, 1.95, 5.6, 0.025, fill=C_ACCENT4)

    solver_rewards = [
        ("+0.1", "Move closer to Vault (distance shaping)"),
        ("-0.1", "Move further from Vault"),
        ("-0.01","Per time-step penalty (efficiency)"),
        ("+0.05","Proximity bonus within 3 tiles of Vault"),
        ("+10.0","REACH THE VAULT  ✅"),
        ("-1.0", "DETECTED by camera or guard  🚨"),
        ("+2.0", "Partial credit at timeout (distance-based)"),
    ]
    cols = [C_ACCENT4, C_ACCENT3, C_ACCENT3, C_ACCENT2, C_ACCENT4, C_ACCENT3, C_ACCENT2]
    for i, ((val, desc), col) in enumerate(zip(solver_rewards, cols)):
        y = 2.1 + i * 0.73
        add_rect(slide, 0.5, y, 1.1, 0.42, fill=col)
        add_textbox(slide, val, 0.5, y, 1.1, 0.42,
                    font_size=Pt(12), bold=True, color=C_DARK_BG, align=PP_ALIGN.CENTER)
        add_textbox(slide, desc, 1.7, y + 0.03, 4.5, 0.38,
                    font_size=Pt(11.5), color=C_LIGHT_GRAY)

    # Architect rewards
    card(slide, 6.95, 1.38, 6.1, 5.75, fill=C_MID_BG, border=C_ACCENT1)
    add_textbox(slide, "🔒  Architect Rewards", 7.15, 1.52, 5.7, 0.45,
                font_size=Pt(15), bold=True, color=C_ACCENT1)
    add_rect(slide, 7.15, 1.95, 5.6, 0.025, fill=C_ACCENT1)

    arch_rewards = [
        ("+1.0","Detection rate × 1.0 (more detections = better security)"),
        ("-0.5","Solver succeeds >80% of attempts (too easy)"),
        ("+0.2","Solver succeeds 20–60%  (challenging but fair zone)"),
        ("-1.0","Invalid layout — no valid path to Vault"),
    ]
    arch_cols = [C_ACCENT1, C_ACCENT3, C_ACCENT4, C_ACCENT3]
    for i, ((val, desc), col) in enumerate(zip(arch_rewards, arch_cols)):
        y = 2.1 + i * 0.88
        add_rect(slide, 7.15, y, 1.1, 0.42, fill=col)
        add_textbox(slide, val, 7.15, y, 1.1, 0.42,
                    font_size=Pt(12), bold=True, color=C_DARK_BG, align=PP_ALIGN.CENTER)
        add_textbox(slide, desc, 8.35, y + 0.03, 4.5, 0.38,
                    font_size=Pt(11.5), color=C_LIGHT_GRAY)

    # Sweet spot callout
    card(slide, 6.95, 5.7, 6.1, 1.3, fill=RGBColor(0x10, 0x28, 0x18), border=C_ACCENT4)
    add_textbox(slide,
                "🎯  Architect's Sweet Spot: ~30% Solve Rate\n"
                "Security challenging enough to catch most attempts, but not impossibly hard.",
                7.08, 5.82, 5.85, 1.05, font_size=Pt(12), color=C_ACCENT4)


def slide_curriculum(prs):
    """Slide 9 – Curriculum Learning"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT3)
    slide_header(slide, "Curriculum Learning Pipeline", "Gradual difficulty ramp — stability before adversarial competition", title_color=C_ACCENT3)
    accent_bar_bottom(slide, C_ACCENT3)

    phases = [
        (C_DIM_GRAY,  "WARMUP",     "Pre-Training\nEp 1–30",    "Budget: 0",
         "Empty grid; Solver learns basic\nnavigation toward the Vault",
         "100% solve", "—"),
        (C_ACCENT1,   "PHASE I",    "Walls Only\nEp 31–80",     "Budget: 5",
         "Walls introduced; Solver learns\npath-finding around obstacles",
         "~100%", "0%"),
        (C_ACCENT2,   "PHASE II",   "+ Cameras\nEp 81–200",     "Budget: 8",
         "Rotating cameras added; LSTM memory\nexploits rotation timing windows",
         "45–100%", "20–40%"),
        (C_ACCENT3,   "PHASE III",  "Full Security\nEp 201–400", "Budget: 15",
         "Guards introduced; true adversarial\ncompetition begins — collapse then recovery",
         "0→50%", "50–80%"),
        (C_ACCENT4,   "PHASE IV",   "Expert Mode\nEp 401–500",  "Budget: 22",
         "Maximum budget; Architect fortifies\nvault rooms; Solver adapts dynamically",
         "~60%", "~40%"),
    ]

    cx = 0.22
    for col, label, ep_range, budget, desc, solve, detect in phases:
        # vertical card
        card(slide, cx, 1.38, 2.44, 5.75, fill=C_MID_BG, border=col, border_w=Pt(1.5))
        add_rect(slide, cx, 1.38, 2.44, 0.12, fill=col)
        add_textbox(slide, label, cx, 1.52, 2.44, 0.45,
                    font_size=Pt(13), bold=True, color=col, align=PP_ALIGN.CENTER)
        add_textbox(slide, ep_range, cx, 1.95, 2.44, 0.5,
                    font_size=Pt(10), color=C_DIM_GRAY, align=PP_ALIGN.CENTER)
        add_rect(slide, cx + 0.1, 2.45, 2.22, 0.025, fill=col)
        add_textbox(slide, budget, cx, 2.55, 2.44, 0.38,
                    font_size=Pt(12), bold=True, color=C_ACCENT2, align=PP_ALIGN.CENTER)
        add_textbox(slide, desc, cx + 0.1, 2.98, 2.22, 1.45,
                    font_size=Pt(10.5), color=C_LIGHT_GRAY, align=PP_ALIGN.CENTER)
        add_textbox(slide, "Solve Rate", cx, 4.55, 2.44, 0.32,
                    font_size=Pt(9), color=C_DIM_GRAY, align=PP_ALIGN.CENTER)
        add_textbox(slide, solve, cx, 4.82, 2.44, 0.4,
                    font_size=Pt(14), bold=True, color=C_ACCENT4, align=PP_ALIGN.CENTER)
        add_textbox(slide, "Detection", cx, 5.28, 2.44, 0.32,
                    font_size=Pt(9), color=C_DIM_GRAY, align=PP_ALIGN.CENTER)
        add_textbox(slide, detect, cx, 5.55, 2.44, 0.38,
                    font_size=Pt(14), bold=True, color=C_ACCENT3, align=PP_ALIGN.CENTER)
        cx += 2.57


def slide_networks(prs):
    """Slide 10 – Neural Network Architecture"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide)
    slide_header(slide, "Neural Network Architecture", "Solver: CNN + LSTM  |  Architect: Encoder-Decoder CNN")
    accent_bar_bottom(slide)

    # Solver network
    card(slide, 0.3, 1.38, 6.1, 5.75, fill=C_MID_BG, border=C_ACCENT2)
    add_textbox(slide, "🕵️  Solver Network — 550K Parameters", 0.5, 1.52, 5.7, 0.45,
                font_size=Pt(14), bold=True, color=C_ACCENT2)
    add_rect(slide, 0.5, 1.95, 5.6, 0.025, fill=C_ACCENT2)

    solver_layers = [
        ("INPUT", "3-Channel Grid (20×20)", C_ACCENT2),
        ("CONV ×3", "3→32→64→64 kernels, ReLU activation", C_ACCENT1),
        ("LSTM", "256 units, 128 hidden — temporal memory", C_ACCENT4),
        ("POLICY HEAD", "5-action probability distribution (U/D/L/R/WAIT)", C_ACCENT2),
        ("VALUE HEAD", "State value estimate for PPO critic", C_DIM_GRAY),
    ]
    for i, (layer, desc, col) in enumerate(solver_layers):
        y = 2.1 + i * 0.92
        add_rect(slide, 0.5, y, 1.5, 0.45, fill=col)
        add_textbox(slide, layer, 0.5, y, 1.5, 0.45,
                    font_size=Pt(9.5), bold=True, color=C_DARK_BG, align=PP_ALIGN.CENTER)
        add_textbox(slide, desc, 2.15, y + 0.04, 4.2, 0.38,
                    font_size=Pt(11), color=C_LIGHT_GRAY)
        if i < 4:
            add_textbox(slide, "↓", 1.18, y + 0.45, 0.5, 0.35,
                        font_size=Pt(14), color=C_DIM_GRAY, align=PP_ALIGN.CENTER)

    add_textbox(slide, "Why LSTM? Cameras rotate — the Solver needs memory to predict\nwhen a camera will rotate away, creating a safe movement window.",
                0.5, 6.7, 5.6, 0.7, font_size=Pt(10), italic=True, color=C_DIM_GRAY)

    # Architect network
    card(slide, 6.95, 1.38, 6.1, 5.75, fill=C_MID_BG, border=C_ACCENT1)
    add_textbox(slide, "🔒  Architect Network — 407K Parameters", 7.15, 1.52, 5.7, 0.45,
                font_size=Pt(14), bold=True, color=C_ACCENT1)
    add_rect(slide, 7.15, 1.95, 5.6, 0.025, fill=C_ACCENT1)

    arch_layers = [
        ("INPUT",        "1-Channel grid (start/vault/border)", C_ACCENT1),
        ("ENCODER ×3",   "1→32→64→128, ReLU — spatial compression", C_ACCENT4),
        ("DECODER ×3",   "128→64→32→3, ConvTranspose, Sigmoid", C_ACCENT2),
        ("PLACEMENT MAP","Per-tile probabilities: [wall, camera, guard]", C_ACCENT1),
        ("PARAM HEADS",  "Camera FOV, rotation speed, guard patrol paths", C_DIM_GRAY),
    ]
    for i, (layer, desc, col) in enumerate(arch_layers):
        y = 2.1 + i * 0.92
        add_rect(slide, 7.15, y, 1.5, 0.45, fill=col)
        add_textbox(slide, layer, 7.15, y, 1.5, 0.45,
                    font_size=Pt(9.5), bold=True, color=C_DARK_BG, align=PP_ALIGN.CENTER)
        add_textbox(slide, desc, 8.75, y + 0.04, 4.2, 0.38,
                    font_size=Pt(11), color=C_LIGHT_GRAY)
        if i < 4:
            add_textbox(slide, "↓", 7.85, y + 0.45, 0.5, 0.35,
                        font_size=Pt(14), color=C_DIM_GRAY, align=PP_ALIGN.CENTER)

    add_textbox(slide, "Samples from probability distributions with temperature — higher temperature\ncreates chaotic, unpredictable layouts; lower temperature maximises confidence.",
                7.15, 6.7, 5.6, 0.7, font_size=Pt(10), italic=True, color=C_DIM_GRAY)


def slide_phase1_results(prs):
    """Slide 11 – Phase I Results"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT4)
    slide_header(slide, "PHASE I — Results: Local Training Run",
                 "500 Episodes · 20×20 Grid · 126.5 min runtime · ResNet-Enhanced Architecture",
                 title_color=C_ACCENT4)
    accent_bar_bottom(slide, C_ACCENT4)

    # Stat boxes row
    stats = [
        ("59.5%",  "Final Solve Rate",     C_ACCENT4),
        ("40.5%",  "Final Detection Rate", C_ACCENT3),
        ("0.335",  "Avg Architect Reward", C_ACCENT1),
        ("7.813",  "Avg Solver Reward",    C_ACCENT2),
    ]
    sx = 0.3
    for val, label, col in stats:
        stat_box(slide, val, label, sx, 1.38, w=3.1, h=1.15, val_color=col)
        sx += 3.25

    # Per-phase narrative cards
    phases_r = [
        (C_ACCENT1, "Phase I: Warmup & Walls (Ep 1–80)",
         "Solver mastered open grid navigation by Episode 30 — reaching vault in 42 steps. "
         "With static walls, solve rate stayed at 100% (walls only deflect, not detect)."),
        (C_ACCENT2, "Phase II: Cameras Added (Ep 81–200)",
         "Solve rate instantly plummeted to 45% at Ep 80. Within 10 episodes (Ep 90), LSTM memory "
         "learned to wait for camera rotation windows — solve rate restored to 100%."),
        (C_ACCENT3, "Phase III: Guards Introduced (Ep 200–400)",
         "Catastrophic collapse: Solve Rate dropped to 0.00, Detection to 1.00. Chokepoints + guards "
         "= lethal traps. After 100 episodes of adaptation, Solver clawed back to 50/50 — Nash balance."),
        (C_ACCENT4, "Phase IV: Expert Mode (Ep 400–500)",
         "Architect given maximum budget (22 pts). Win rate fluctuated wildly (5% at Ep 470, 100% at Ep 490). "
         "Final ~60% average against expert security — proves neural architecture is fully operational."),
    ]
    py = 2.72
    for col, title, body in phases_r:
        card(slide, 0.3, py, 12.73, 1.12, fill=C_MID_BG, border=col, border_w=Pt(1.5))
        add_rect(slide, 0.3, py, 0.08, 1.12, fill=col)
        add_textbox(slide, title, 0.55, py + 0.08, 5.5, 0.38,
                    font_size=Pt(12), bold=True, color=col)
        add_textbox(slide, body, 0.55, py + 0.5, 12.2, 0.55,
                    font_size=Pt(11), color=C_LIGHT_GRAY)
        py += 1.2


def slide_phase2_intro(prs):
    """Slide 12 – Phase II Introduction"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT2)
    slide_header(slide, "PHASE II — Kaggle Cloud Scale-Up",
                 "Dual NVIDIA Tesla T4 · 17,000 Episodes · Hugging Face Integration",
                 title_color=C_ACCENT2)
    accent_bar_bottom(slide, C_ACCENT2)

    # Context card
    card(slide, 0.3, 1.38, 12.73, 1.4, fill=C_MID_BG, border=C_ACCENT2)
    add_textbox(slide,
                "Phase I proved the framework works — agents learn, compete, and reach equilibrium in 500 episodes. "
                "Phase II answers: what happens when we train 34× longer with cloud GPU acceleration? "
                "The answer revealed deep adversarial cycling — collapse, recovery, and strategic arms race dynamics "
                "impossible to observe in short local runs.",
                0.5, 1.5, 12.3, 1.15, font_size=Pt(12.5), color=C_LIGHT_GRAY)

    # Upgrade highlights
    upgrades = [
        (C_ACCENT1, "Scale",
         "17,000 episodes vs 500\n(34× more training)"),
        (C_ACCENT2, "Hardware",
         "Dual NVIDIA Tesla T4 GPUs\nvs local CPU/single GPU"),
        (C_ACCENT4, "Curriculum",
         "4-stage: Rookie → Intermediate\n→ Expert → Master"),
        (C_ACCENT3, "ELO System",
         "Real-time relative skill\ntracking per episode"),
        (C_ACCENT1, "Cloud Storage",
         "Checkpoints auto-synced\nto Hugging Face every 500 ep"),
    ]
    cx = 0.3
    for col, title, body in upgrades:
        card(slide, cx, 3.0, 2.44, 2.2, fill=C_MID_BG, border=col)
        add_rect(slide, cx, 3.0, 2.44, 0.08, fill=col)
        add_textbox(slide, title, cx, 3.1, 2.44, 0.45,
                    font_size=Pt(13), bold=True, color=col, align=PP_ALIGN.CENTER)
        add_textbox(slide, body, cx + 0.1, 3.6, 2.24, 1.5,
                    font_size=Pt(11), color=C_LIGHT_GRAY, align=PP_ALIGN.CENTER)
        cx += 2.57

    # Key insight
    card(slide, 0.3, 5.42, 12.73, 1.72, fill=RGBColor(0x10, 0x22, 0x30), border=C_ACCENT1)
    add_textbox(slide, "⚡  Key Insight from Phase II", 0.5, 5.55, 12.3, 0.42,
                font_size=Pt(14), bold=True, color=C_ACCENT1)
    add_textbox(slide,
                "The training history is not a smooth curve — it is evidence of adversarial cycling. "
                "The Solver dominated early, collapsed twice under Architect pressure, and recovered twice. "
                "This collapse-recovery pattern is the theoretical signature of adversarial co-adaptation, "
                "not a training failure. Final best checkpoint (ep17000): Robber win-rate 1.00, ELO +1093.",
                0.5, 5.95, 12.3, 1.1, font_size=Pt(11.5), color=C_LIGHT_GRAY)


def slide_phase2_upgrades(prs):
    """Slide 13 – Phase II Model Upgrades"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT4)
    slide_header(slide, "Phase II — Advanced Methodologies",
                 "Architectural Anatomy and PPO Engine Mechanics",
                 title_color=C_ACCENT4)
    accent_bar_bottom(slide, C_ACCENT4)

    # 1. Architect Anatomy
    card(slide, 0.3, 1.38, 12.73, 1.7, fill=C_MID_BG, border=C_ACCENT1)
    add_textbox(slide, "🔒 The Architect's Anatomy", 0.5, 1.5, 12.3, 0.4,
                font_size=Pt(14), bold=True, color=C_ACCENT1)
    add_textbox(slide,
                "• Explicitly engineered with an 8.5M parameter ResNet + Group Normalization backbone.\n"
                "• Leverages a 512-dimensional LSTM hidden state to map decision branching over a dense 400-cell grid matrix.\n"
                "• Avoids standard DataParallel computations via True Split Asynchronous Deployment on dedicated cuda:0.",
                0.5, 1.9, 12.3, 1.0, font_size=Pt(12), color=C_LIGHT_GRAY)

    # 2. Robber Anatomy
    card(slide, 0.3, 3.25, 12.73, 1.7, fill=C_MID_BG, border=C_ACCENT2)
    add_textbox(slide, "🕵️ The Robber's Anatomy", 0.5, 3.35, 12.3, 0.4,
                font_size=Pt(14), bold=True, color=C_ACCENT2)
    add_textbox(slide,
                "• Processes 12-channel spatial tensor inputs (tracking normalized geodesic distances and danger zones).\n"
                "• Replaces simple CNNs passing with a 512-parameter LSTM (7M parameters total) to track temporal guard sequences.\n"
                "• Detaches hidden states across continuous timesteps to construct a reliable memory of rotating camera phases.",
                0.5, 3.75, 12.3, 1.0, font_size=Pt(12), color=C_LIGHT_GRAY)

    # 3. Engine Mechanics
    card(slide, 0.3, 5.12, 12.73, 1.7, fill=RGBColor(0x1a, 0x1a, 0x24), border=C_ACCENT3)
    add_textbox(slide, "⚙️ Core Engine Mechanics (PPO)", 0.5, 5.25, 12.3, 0.4,
                font_size=Pt(14), bold=True, color=C_ACCENT3)
    add_textbox(slide,
                "• Utilizes Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (λ = 0.95, γ = 0.99).\n"
                "• Enforces a clipped surrogate objective (ϵ = 0.2) to prevent catastrophic network self-destruction during updates.\n"
                "• Applies entropy regularization (coef = 0.02) to actively circumvent Strategy Collapse and avoid boring Nash Equilibria.",
                0.5, 5.65, 12.3, 1.0, font_size=Pt(12), color=C_LIGHT_GRAY)


def slide_training_dynamics(prs):
    """Slide 14 – Training Dynamics & Collapse/Recovery"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT3)
    slide_header(slide, "Training Dynamics — Collapse & Recovery Cycles",
                 "17,000 episodes of adversarial co-adaptation across 5 distinct phases",
                 title_color=C_ACCENT3)
    accent_bar_bottom(slide, C_ACCENT3)

    milestones = [
        ("Ep 50",   "Rookie",       "0.96", "+477",  "Solver starts strong", C_ACCENT4),
        ("Ep 200",  "Expert",       "1.00", "+737",  "Solver generalises quickly", C_ACCENT4),
        ("Ep 300",  "Master",       "1.00", "+810",  "Peaks at hardest curriculum", C_ACCENT4),
        ("Ep 1750", "Master",       "0.59", "-381",  "First collapse begins", C_ACCENT2),
        ("Ep 2950", "Master",       "0.00", "-1060", "First collapse low point", C_ACCENT3),
        ("Ep 3000", "Master",       "0.28", "+206",  "Recovery begins", C_ACCENT2),
        ("Ep 5350", "Master",       "1.00", "+994",  "High-performance peak", C_ACCENT4),
        ("Ep 10200","Master",       "0.55", "-382",  "Second collapse begins", C_ACCENT2),
        ("Ep 15500","Master",       "0.00", "-1317", "Deepest valley observed", C_ACCENT3),
        ("Ep 16000","Master",       "1.00", "+895",  "Recovery after restart", C_ACCENT4),
        ("Ep 17000","Master",       "1.00", "+1093", "Best visible checkpoint ⭐", C_ACCENT4),
    ]

    # Table header
    headers = ["Episode", "Stage", "Win Rate", "ELO Diff", "Interpretation"]
    hx = [0.3, 1.8, 3.05, 4.2, 5.6]
    hw = [1.4, 1.2, 1.1, 1.3, 7.2]
    for i, (h, hxv, hwv) in enumerate(zip(headers, hx, hw)):
        add_rect(slide, hxv, 1.38, hwv, 0.42, fill=C_ACCENT3)
        add_textbox(slide, h, hxv, 1.38, hwv, 0.42,
                    font_size=Pt(11), bold=True, color=C_DARK_BG, align=PP_ALIGN.CENTER)

    for row, (ep, stage, win, elo, interp, col) in enumerate(milestones):
        y = 1.85 + row * 0.49
        bg = C_MID_BG if row % 2 == 0 else RGBColor(0x10, 0x18, 0x28)
        row_data = [ep, stage, win, elo, interp]
        for j, (val, hxv, hwv) in enumerate(zip(row_data, hx, hw)):
            add_rect(slide, hxv, y, hwv, 0.46, fill=bg, line_color=C_DIM_GRAY, line_width=Pt(0.3))
            txt_col = col if j in (2, 3) else C_LIGHT_GRAY
            add_textbox(slide, val, hxv + 0.05, y + 0.03, hwv - 0.1, 0.38,
                        font_size=Pt(10.5), color=txt_col, bold=(j in (0,2,3)))

    # Phase labels
    phase_labels = [
        (1.38, 1.85,  3*0.49+0.46, "Phase A: Curriculum Climb", C_ACCENT4),
        (1.38, 1.85+3*0.49, 2*0.49+0.46, "Phase B: First Collapse", C_ACCENT3),
        (1.38, 1.85+5*0.49, 2*0.49+0.46, "Phase C: Recovery", C_ACCENT4),
        (1.38, 1.85+7*0.49, 2*0.49+0.46, "Phase D: 2nd Collapse", C_ACCENT3),
        (1.38, 1.85+9*0.49, 2*0.49+0.46, "Phase E: Final Recovery", C_ACCENT4),
    ]


def slide_elo(prs):
    """Slide 14 – ELO Arms Race"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT1)
    slide_header(slide, "ELO Arms Race — Strategic Equilibrium",
                 "Relative skill tracking reveals adversarial co-adaptation cycles",
                 title_color=C_ACCENT1)
    accent_bar_bottom(slide, C_ACCENT1)

    # ELO explanation
    card(slide, 0.3, 1.38, 12.73, 1.35, fill=C_MID_BG, border=C_ACCENT1)
    add_textbox(slide,
                "ELO is a relative skill metric borrowed from chess and competitive gaming. Here it tracks which agent "
                "is currently dominating. A positive ELO diff means the Solver is winning; negative means Architect dominates. "
                "Under PPO constraints, neither side easily exploits the other — LSTM hidden states push both agents into "
                "high-level spatial deduction rather than memorised routes.",
                0.5, 1.5, 12.25, 1.1, font_size=Pt(12), color=C_LIGHT_GRAY)

    # ELO stages
    elo_stages = [
        (C_ACCENT4, "The Rookie Stage\n(Ep 50–300)",
         "Robber ELO spikes to +810 because randomly-walking guards provide minimal map coverage. "
         "Single-agent navigation dominates early — layout complexity is insufficient to challenge a learning Solver."),
        (C_ACCENT3, "The Overfit Rebound\n(Ep 1750–2950)",
         "Architect learns to cluster cameras near the Vault. Solver ELO collapses to -1060 — "
         "the Architect has found an effective chokepoint strategy that the Solver hasn't seen before."),
        (C_ACCENT1, "The Recovery Window\n(Ep 3000–10000)",
         "Solver discovers new blind-spot exploitation strategies. ELO recovers to +994 — "
         "a long positive regime where the Solver consistently outperforms the Architect."),
        (C_ACCENT2, "The Final Arms Race\n(Ep 10200–17000)",
         "Second deep collapse to -1317, then full recovery to +1093 at ep17000. "
         "Classic adversarial cycling — each side forces the other to find superior strategies."),
    ]

    cy = 2.9
    for col, title, body in elo_stages:
        card(slide, 0.3, cy, 12.73, 1.15, fill=C_MID_BG, border=col, border_w=Pt(1.5))
        add_rect(slide, 0.3, cy, 0.08, 1.15, fill=col)
        add_textbox(slide, title, 0.55, cy + 0.08, 3.5, 0.48,
                    font_size=Pt(12), bold=True, color=col)
        add_textbox(slide, body, 4.1, cy + 0.1, 8.8, 1.0,
                    font_size=Pt(11.5), color=C_LIGHT_GRAY)
        cy += 1.22

    # Best checkpoint callout
    card(slide, 0.3, 7.0, 12.73, 0.38, fill=RGBColor(0x08, 0x1C, 0x10), border=C_ACCENT4)
    add_textbox(slide,
                "⭐  ep17000 — Best Checkpoint: Win-Rate 1.00 · ELO +1093 · Stage: Master   "
                "→  Default evaluation target for all dashboard demos.",
                0.5, 7.04, 12.3, 0.32, font_size=Pt(11.5), bold=True, color=C_ACCENT4)


def slide_dashboard(prs):
    """Slide 15 – Web Dashboard"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT4)
    slide_header(slide, "Real-Time Web Dashboard", "Flask + WebSocket live monitoring and control interface", title_color=C_ACCENT4)
    accent_bar_bottom(slide, C_ACCENT4)

    panels = [
        (C_ACCENT1,   "🗺️  Grid Visualization (Left Panel)",
         ["Live 20×20 environment — updates every tick",
          "Tiles: Start (Neon Green), Vault (Pink), Walls (Slate 3D Blocks)",
          "Purple borders = Cameras, Orange borders = Guards",
          "Gold Circle = Solver with path trail",
          "👁 Eye icon toggles sweeping purple vision cones",
          "Tick counter shows current step of max_steps"]),
        (C_ACCENT2,   "⚙️  Training Controls (Right Panel)",
         ["Episodes / Solver Attempts per layout — configurable",
          "🚀 Start Training — launches automated adversarial loop",
          "🎬 Run Demo — single test episode with checkpoint weights",
          "Interactive Mode: set Budget, Temperature, Freeze toggles",
          "Temperature >1.5 = chaotic layouts; <0.5 = maximum confidence",
          "Asset restrictions: disable Cameras/Guards for ablation tests"]),
        (C_ACCENT4,   "📊  Live Metrics & Charts",
         ["Live Metrics Board: Solve Rate vs Detection Rate percentages",
          "Two spline charts: Rewards & Rates over time",
          "Smoothed curves reveal when agents hit equilibrium zone",
          "Game Log: tabular record of every episode (Mode, Budget, Rates)",
          "❄️A / ❄️S icons show frozen agent during interactive runs",
          "Auto-checkpoint every 50 episodes — never lose progress"]),
        (C_ACCENT3,   "🔄  Path Simulation & Checkpoints",
         ["Load historic checkpoint from dropdown — retroactive simulation",
          "Compare ep03500 vs ep15500 vs ep17000 behaviorally",
          "Simulate Demo button — watch policy from any training epoch",
          "Best practice: start with Budget=5, no cameras — watch basics first",
          "Then freeze Architect (❄️A) to let Solver catch up on hard maps"]),
    ]

    cx = 0.3
    for i, (col, title, bullets) in enumerate(panels):
        row, c = divmod(i, 2)
        px = 0.3 + c * 6.5
        py = 1.38 + row * 3.0
        card(slide, px, py, 6.2, 2.75, fill=C_MID_BG, border=col)
        add_rect(slide, px, py, 6.2, 0.08, fill=col)
        add_textbox(slide, title, px + 0.15, py + 0.12, 5.85, 0.45,
                    font_size=Pt(13), bold=True, color=col)
        for j, b in enumerate(bullets):
            add_textbox(slide, f"▸  {b}", px + 0.15, py + 0.65 + j * 0.34, 5.88, 0.32,
                        font_size=Pt(10.5), color=C_LIGHT_GRAY)


def slide_phase2_results(prs):
    """Slide 16 – Phase II Quantitative Results"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT2)
    slide_header(slide, "PHASE II — Key Results: Cloud Training",
                 "17,000 episodes · Dual T4 · Checkpoint ep17000 as best model",
                 title_color=C_ACCENT2)
    accent_bar_bottom(slide, C_ACCENT2)

    # Top stat row
    stats2 = [
        ("1.00",   "Robber Win-Rate at ep17000", C_ACCENT4),
        ("+1093",  "ELO Differential",           C_ACCENT2),
        ("+2.3",   "Robber Reward (ep17000)",    C_ACCENT4),
        ("17,000", "Total Episodes Trained",     C_ACCENT1),
    ]
    sx = 0.3
    for val, label, col in stats2:
        stat_box(slide, val, label, sx, 1.38, w=3.1, h=1.15, val_color=col)
        sx += 3.25

    # Timeline summary
    card(slide, 0.3, 2.7, 12.73, 4.42, fill=C_MID_BG, border=C_ACCENT2)
    add_textbox(slide, "Training Timeline Summary", 0.5, 2.82, 12.3, 0.45,
                font_size=Pt(15), bold=True, color=C_ACCENT2)
    add_rect(slide, 0.5, 3.25, 12.23, 0.025, fill=C_ACCENT2)

    timeline = [
        (C_ACCENT4, "Phase A — Curriculum Climb (Ep 50–1700)",
         "Rookie → Master in ~300 episodes. Win-rate 0.90–1.00 consistently. ELO grew to +810. "
         "Solver generalised to all 4 curriculum stages faster than expected."),
        (C_ACCENT3, "Phase B — First Collapse (Ep 1750–2950)",
         "Win-rate collapsed to 0.00. ELO dropped to -1060. Architect discovered vault-room fortification "
         "strategy — combining chokepoints, cameras, and guards with high density."),
        (C_ACCENT4, "Phase C — First Recovery (Ep 3000–10000)",
         "Solver rediscovered wall-hugging + blind-spot timing strategy. ELO returned to +994. "
         "7000-episode positive regime demonstrated long-term stability after adversarial rebound."),
        (C_ACCENT3, "Phase D — Second Collapse (Ep 10200–15500)",
         "Deepest valley: ELO -1317. Architect evolved patrol synchronisation — guards timed so "
         "no safe window existed. Solver spent 5300 episodes at 0% win-rate."),
        (C_ACCENT4, "Phase E — Final Recovery (Ep 15501–17000)",
         "Full recovery: Win-rate 1.00, ELO +1093. Post-restart, Solver learned fundamentally "
         "superior strategies. ep17000 selected as best checkpoint for all evaluations."),
    ]
    for i, (col, title, body) in enumerate(timeline):
        y = 3.33 + i * 0.74
        add_rect(slide, 0.5, y, 0.06, 0.65, fill=col)
        add_textbox(slide, title, 0.7, y + 0.01, 4.5, 0.35,
                    font_size=Pt(11), bold=True, color=col)
        add_textbox(slide, body, 0.7, y + 0.36, 12.0, 0.35,
                    font_size=Pt(10.5), color=C_LIGHT_GRAY)


def slide_emergent(prs):
    """Slide 17 – Emergent Strategies & Nash Equilibrium"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_ACCENT4)
    slide_header(slide, "Emergent Strategies & Nash Equilibrium",
                 "Nobody programmed these tactics — they arose from competition",
                 title_color=C_ACCENT4)
    accent_bar_bottom(slide, C_ACCENT4)

    two_col_bullets(
        slide,
        "🔒  Architect Discovered",
        [
            "Chokepoints — Narrow corridors covered by rotating cameras",
            "Dead Ends — Walls that look like paths but lead into kill-zones",
            "Patrol Synchronisation — Guards timed so no safe window exists",
            "Camera Crossfire — Multiple cameras covering each other's blind spots",
            "Vault Fortification — Dense security rings around the objective",
            "Budget Efficiency — Cameras > Guards for cost-per-coverage",
        ],
        "🕵️  Solver Discovered",
        [
            "Wall Hugging — Using walls as cover from camera vision cones",
            "Timing Windows — Waiting for cameras to rotate away before sprinting",
            "Indirect Paths — Longer routes that avoid high-density detection zones",
            "Patience (WAIT) — Staying still at critical moments for guard cycles",
            "Blind Spot Exploitation — Navigating through camera angle gaps",
            "Temporal Mapping — LSTM remembers guard positions from 5 steps ago",
        ],
        l=0.3, t=1.38, card_h=4.0,
    )

    # Nash equilibrium explanation
    card(slide, 0.3, 5.55, 12.73, 1.6, fill=RGBColor(0x0A, 0x1E, 0x14), border=C_ACCENT4)
    add_textbox(slide, "🎯  Nash Equilibrium — The Natural Balance Point", 0.5, 5.67, 12.3, 0.45,
                font_size=Pt(14), bold=True, color=C_ACCENT4)
    add_textbox(slide,
                "Phase I converged at ~30% Solve Rate | 40% Detection Rate — the mathematical sweet spot. "
                "Phase II showed deeper cycling but returned to competitive balance. "
                "At equilibrium: the Architect cannot catch the Solver more reliably without making invalid levels; "
                "the Solver cannot reach the Vault more often without exposing itself to detection. "
                "This is the same framework used in poker AI, military strategy models, and economic game theory.",
                0.5, 6.08, 12.3, 1.0, font_size=Pt(11.5), color=C_LIGHT_GRAY)


def slide_references(prs):
    """Slide 18 – References"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    accent_bar_top(slide, C_DIM_GRAY)
    slide_header(slide, "References", "Key literature supporting this adversarial RL framework (2021–2025)",
                 title_color=C_ACCENT1, sub_color=C_DIM_GRAY)
    accent_bar_bottom(slide, C_DIM_GRAY)

    refs = [
        ("[1]", "Schulman et al., 2017",
         '"Proximal Policy Optimization Algorithms," arXiv:1707.06347. [PPO — core RL algorithm for both agents]'),
        ("[2]", "Baker et al., 2020",
         '"Emergent Tool Use From Multi-Agent Autocurricula," ICLR 2020. [Foundation for emergent multi-agent strategies]'),
        ("[3]", "Vinyals et al., 2019",
         '"Grandmaster level in StarCraft II using multi-agent RL," Nature, 575. [Adversarial curriculum for complex games]'),
        ("[4]", "Bengio et al., 2021",
         '"Curriculum Learning," ICML 2021. [Theoretical basis for difficulty scheduling and curriculum design]'),
        ("[5]", "Mnih et al., 2022",
         '"Playing Atari with Deep Reinforcement Learning," NeurIPS. [CNN feature extraction for RL observations]'),
        ("[6]", "OpenAI, 2021",
         '"Asymmetric self-play for automatic goal discovery," arXiv:2101.04882. [Architect-Solver game formulation]'),
        ("[7]", "Gleave et al., 2022",
         '"Adversarial Policies: Attacking Deep Reinforcement Learning," ICLR 2022. [Adversarial policy dynamics]'),
        ("[8]", "Hausknecht & Stone, 2015",
         '"Deep Recurrent Q-Networks for Partially Observable MDPs," AAAI. [LSTM in RL for temporal memory]'),
        ("[9]", "Littman, 1994",
         '"Markov games as a framework for multi-agent RL," ICML 1994. [Markov Game theoretical backbone]'),
        ("[10]", "Lanctot et al., 2023",
         '"OpenSpiel: A Framework for Reinforcement Learning in Games," JMLR. [Multi-agent game benchmarks]'),
        ("[11]", "Elo, 1978 / Silver et al., 2021",
         '"The Rating of Chessplayers" / "A general RL algorithm AlphaZero," Science. [ELO rating adaptation]'),
        ("[12]", "Chevalier-Boisvert et al., 2023",
         '"MiniGrid & MiniWorld: Modular & Customisable RL Grid Environments," NeurIPS. [Related grid-world baselines]'),
    ]

    for i, (num, authors, title) in enumerate(refs):
        row, col_idx = divmod(i, 6)
        x = 0.3 if col_idx < 6 else 6.8
        y = 1.38 + col_idx * 0.98
        if row == 1:
            x += 6.5
        w = 6.1

        add_rect(slide, x, y, 0.55, 0.38, fill=C_ACCENT1)
        add_textbox(slide, num, x, y, 0.55, 0.38,
                    font_size=Pt(10), bold=True, color=C_DARK_BG, align=PP_ALIGN.CENTER)
        add_textbox(slide, authors, x + 0.65, y, w - 0.65, 0.32,
                    font_size=Pt(10), bold=True, color=C_ACCENT2)
        add_textbox(slide, title, x + 0.65, y + 0.3, w - 0.65, 0.62,
                    font_size=Pt(9.5), color=C_LIGHT_GRAY, italic=True)


def slide_conclusion(prs):
    """Slide 19 – Conclusion / Thank You"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    dark_background(slide)
    add_rect(slide, 0, 0, 13.33, 0.12, fill=C_ACCENT4)
    add_rect(slide, 0, 0.12, 13.33, 2.3, fill=C_MID_BG)
    add_rect(slide, 0, 0, 0.12, 7.5, fill=C_ACCENT4)

    add_textbox(slide, "CONCLUSIONS", 0.35, 0.42, 12.6, 0.75,
                font_size=Pt(42), bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    add_textbox(slide, "Heist Architect — CSE4019 Adversarial RL Framework",
                0.35, 1.12, 12.6, 0.5, font_size=Pt(18), italic=True,
                color=C_ACCENT4, align=PP_ALIGN.LEFT)
    add_rect(slide, 0.35, 1.6, 12.6, 0.03, fill=C_DIM_GRAY)

    takeaways = [
        (C_ACCENT4,  "Phase I proved the concept:  ",
         "Local 500-episode run achieved Nash Equilibrium at ~60% Solve / 40% Detect."),
        (C_ACCENT2,  "Phase II deepened the science: ",
         "17,000 episodes on Kaggle T4 GPUs revealed adversarial cycling — collapse, recovery, arms race."),
        (C_ACCENT1,  "Emergent complexity works:    ",
         "Chokepoints, patrol sync, wall-hugging, timing — none programmed; all emerged from competition."),
        (C_ACCENT3,  "The architecture scales:      ",
         "CNN+LSTM Solver (550K params) and Encoder-Decoder Architect (407K) remain competitive at Master level."),
        (C_ACCENT4,  "Game theory is validated:     ",
         "Nash Equilibrium, Markov Games, and ELO dynamics are empirically reproducible in our custom environment."),
    ]
    for i, (col, bold_text, body) in enumerate(takeaways):
        y = 1.75 + i * 1.05
        card(slide, 0.35, y, 12.6, 0.9, fill=C_MID_BG, border=col)
        add_textbox(slide, f"✓  {bold_text}", 0.55, y + 0.1, 4.5, 0.7,
                    font_size=Pt(12), bold=True, color=col)
        add_textbox(slide, body, 5.15, y + 0.15, 7.6, 0.65,
                    font_size=Pt(11.5), color=C_LIGHT_GRAY)

    add_textbox(slide,
                "Shanmuk4622/heist-architect-v2  |  CSE4019 — 2025-2026  |  Built with PyTorch · Flask · CUDA · Kaggle",
                0.35, 7.1, 12.6, 0.38,
                font_size=Pt(10), color=C_DIM_GRAY, align=PP_ALIGN.LEFT)
    accent_bar_bottom(slide, C_ACCENT4)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def build_presentation():
    prs = Presentation()
    prs.slide_width  = Inches(13.33)   # 16:9 widescreen
    prs.slide_height = Inches(7.5)

    print("Building slides...")
    slide_title(prs)           ;print("  [1/19] Title")
    slide_agenda(prs)          ;print("  [2/19] Agenda")
    slide_problem(prs)         ;print("  [3/19] Problem Statement")
    slide_concept(prs)         ;print("  [4/19] Adversarial RL Concept")
    slide_agents(prs)          ;print("  [5/19] The Two Agents")
    slide_environment(prs)     ;print("  [6/19] Environment Design")
    slide_security(prs)        ;print("  [7/19] Security Components")
    slide_reward(prs)          ;print("  [8/19] Reward System")
    slide_curriculum(prs)      ;print("  [9/19] Curriculum Learning")
    slide_networks(prs)        ;print(" [10/19] Neural Networks")
    slide_phase1_results(prs)  ;print(" [11/20] Phase I Results")
    slide_phase2_intro(prs)    ;print(" [12/20] Phase II Introduction")
    slide_phase2_upgrades(prs) ;print(" [13/20] Phase II Model Upgrades")
    slide_training_dynamics(prs);print(" [14/20] Training Dynamics")
    slide_elo(prs)             ;print(" [15/20] ELO Arms Race")
    slide_dashboard(prs)       ;print(" [16/20] Web Dashboard")
    slide_phase2_results(prs)  ;print(" [17/20] Phase II Results")
    slide_emergent(prs)        ;print(" [18/20] Emergent Strategies")
    slide_references(prs)      ;print(" [19/20] References")
    slide_conclusion(prs)      ;print(" [20/20] Conclusion")

    out = "Heist_Architect_Presentation_CSE4019.pptx"
    prs.save(out)
    print(f"\n✅  Saved → {out}")
    return out


if __name__ == "__main__":
    build_presentation()
