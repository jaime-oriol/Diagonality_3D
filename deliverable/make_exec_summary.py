"""
Genera el executive summary del hackathon: 5 slides 16:9.

Salidas en submission/:
  - executive_summary.pptx  (vidos Vision/PPCF/DOS embebidos en slides 3-5)
  - executive_summary.pdf   (version estatica, los videos salen como still)

Slides: 1 intro diagonalidad, 2 pipeline, 3 Vision, 4 PPCF, 5 DOS.
Ejecutar desde la raiz del repo:  python3 deliverable/make_exec_summary.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image
from pptx import Presentation
from pptx.util import Inches

ROOT = Path(__file__).resolve().parent.parent
FIG  = ROOT / "deliverable" / "figures"
VID  = ROOT / "figures" / "videos"
LOGO = ROOT / "figures" / "logos" / "Logo.png"
OUT  = ROOT / "submission"
SLD  = FIG / "exec_slides"
OUT.mkdir(exist_ok=True)
SLD.mkdir(exist_ok=True)

# --- paleta -----------------------------------------------------------
BG      = "#26282b"
PANEL   = "#313332"
WHITE   = "#f4f5f6"
MUTE    = "#a8adb3"
ACCENT  = "#3b82f6"   # corporate blue
AMBER   = "#f0b450"

plt.rcParams["font.family"] = "DejaVu Sans"

# slide en unidades de grid 16 x 9
GX, GY = 16.0, 9.0
SLIDE_W_IN, SLIDE_H_IN = 13.333, 7.5


def new_slide():
    fig = plt.figure(figsize=(SLIDE_W_IN, SLIDE_H_IN), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, GX); ax.set_ylim(0, GY)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), GX, GY, color=BG, zorder=0))
    # banda de acento superior
    ax.add_patch(plt.Rectangle((0, GY - 0.09), GX, 0.09, color=ACCENT, zorder=1))
    return fig, ax


def footer(ax, page):
    ax.text(0.55, 0.42, "Diagonality — The Best of Both Worlds",
            color=MUTE, fontsize=10, va="center", zorder=6)
    ax.text(GX - 0.55, 0.42, f"{page} / 5", color=MUTE, fontsize=10,
            ha="right", va="center", zorder=6)
    if LOGO.exists():
        img = plt.imread(str(LOGO))
        ar = img.shape[1] / img.shape[0]
        h = 0.62; w = h * ar
        ax.imshow(img, extent=[GX - 0.45 - w, GX - 0.45, GY - 0.88, GY - 0.88 + h],
                  zorder=6, aspect="auto")


def place_image(ax, path, x0, x1, y0, y1, z=5):
    """Encaja la imagen en la caja preservando aspecto. Devuelve el rect real."""
    img = plt.imread(str(path))
    ar_img = img.shape[1] / img.shape[0]
    bw, bh = x1 - x0, y1 - y0
    if ar_img > bw / bh:
        w = bw; h = w / ar_img
    else:
        h = bh; w = h * ar_img
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ex = [cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2]
    ax.imshow(img, extent=ex, zorder=z, aspect="auto")
    ax.add_patch(plt.Rectangle((ex[0], ex[2]), ex[1] - ex[0], ex[3] - ex[2],
                 fill=False, ec="#000000", lw=1.2, zorder=z + 1))
    return ex  # [left, right, bottom, top]


def kicker(ax, text):
    ax.text(0.7, GY - 0.62, text.upper(), color=ACCENT, fontsize=12.5,
            fontweight="bold", va="center", zorder=6)


def title(ax, text, y=GY - 1.35):
    ax.text(0.7, y, text, color=WHITE, fontsize=29, fontweight="bold",
            va="center", zorder=6)


def save(fig, name):
    p = SLD / name
    fig.savefig(str(p), facecolor=BG)
    plt.close(fig)
    return p


# ====================================================================
# SLIDE 1 -- portada: solo el core (por que + que hago), sin imagenes
# ====================================================================
def slide1():
    fig, ax = new_slide()
    cx = GX / 2

    # Kicker centrado — portada con composicion simetrica
    ax.text(cx, GY - 0.92,
            "AWS WORLD SPORTS INNOVATION CUP 2026     ·     CHALLENGE 2",
            color=ACCENT, fontsize=12.5, fontweight="bold",
            ha="center", va="center", zorder=6)

    # Titulo en dos niveles
    ax.text(cx, 6.85, "Diagonality", color=WHITE, fontsize=52,
            fontweight="bold", ha="center", va="center", zorder=6)
    ax.text(cx, 5.85, "The Best of Both Worlds", color=ACCENT,
            fontsize=24, ha="center", va="center", zorder=6)

    # Regla fina de acento
    ax.add_patch(plt.Rectangle((cx - 1.45, 5.18), 2.9, 0.03,
                               color=ACCENT, zorder=6))

    # POR QUE — el problema, parrafo apretado, en blanco
    why = [
        "Football's most-discussed tactical idea has never been measured.",
        "The diagonal is a claim about body orientation, and tracking only",
        "ever recorded positions. 3D skeleton tracking — 21 keypoints at",
        "50 Hz — measures it for the first time.",
    ]
    y = 4.45
    for ln in why:
        ax.text(cx, y, ln, color=WHITE, fontsize=14.5,
                ha="center", va="center", zorder=6)
        y -= 0.5

    # QUE HAGO — dos pilares, sin cajas, mucho aire
    pillars = [
        (4.3, ACCENT, "We TEST the theory",
         "6,923 on-ball actions scored with\n"
         "model-independent metrics — H1–H4 hold."),
        (11.7, AMBER, "We BUILD the tool",
         "the Diagonal Opportunity Surface —\n"
         "a real-time, orientation-aware map."),
    ]
    for px, ec, label, sub in pillars:
        ax.text(px, 2.05, label, color=ec, fontsize=16,
                fontweight="bold", ha="center", va="center", zorder=6)
        ax.text(px, 1.30, sub, color=WHITE, fontsize=11.5,
                ha="center", va="center", zorder=6, linespacing=1.55)

    footer(ax, 1)
    return save(fig, "slide1.png"), None


# ====================================================================
# SLIDE 2 -- pipeline
# ====================================================================
def slide2():
    fig, ax = new_slide()
    kicker(ax, "How it works")
    title(ax, "One pipeline, four orientation-aware stages")

    # chip de input
    ax.add_patch(FancyBboxPatch((0.7, 6.05), 14.6, 0.7,
                 boxstyle="round,pad=0.02,rounding_size=0.1",
                 fc=PANEL, ec=MUTE, lw=1.2, zorder=3))
    ax.text(8.0, 6.4, "INPUT   ·   3D skeleton, 21 keypoints @ 50 Hz   ·   "
            "measured head / shoulder / hip orientation of every player",
            color=WHITE, fontsize=13, ha="center", va="center", zorder=6)

    stages = [
        ("1  Vision", "Per-defender field of view.",
         "Bekkers 120° cone, speed\ndecay, torso occlusion."),
        ("2  Pitch Control", "What each player can reach.",
         "Anisotropic reach field;\nblob collapses in blind spot."),
        ("3  DOS Surface", "Danger from a blind defender.",
         "Control gained going\ndiagonal vs straight."),
        ("4  Scanning Gate", "Only what the player perceives.",
         "2.5 s decayed memory —\nreal-time, actionable."),
    ]
    n = len(stages)
    x0, x1 = 0.7, 15.3
    gap = 0.45
    bw = (x1 - x0 - gap * (n - 1)) / n
    top, bot = 5.55, 3.05
    centers = []
    for i, (h, sub, body) in enumerate(stages):
        bx = x0 + i * (bw + gap)
        centers.append(bx + bw / 2)
        ax.add_patch(FancyBboxPatch((bx, bot), bw, top - bot,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     fc=PANEL, ec=ACCENT, lw=1.7, zorder=4))
        ax.add_patch(plt.Rectangle((bx, top - 0.62), bw, 0.62, color=ACCENT,
                     zorder=5))
        ax.text(bx + bw / 2, top - 0.31, h, color="#10242e", fontsize=14.5,
                fontweight="bold", ha="center", va="center", zorder=6)
        ax.text(bx + bw / 2, top - 1.05, sub, color=WHITE, fontsize=11.3,
                fontweight="bold", ha="center", va="center", zorder=6)
        ax.text(bx + bw / 2, top - 1.85, body, color=MUTE, fontsize=10.3,
                ha="center", va="center", zorder=6, linespacing=1.45)
    # flechas entre cajas
    for i in range(n - 1):
        xa = x0 + i * (bw + gap) + bw
        ax.add_patch(FancyArrowPatch((xa + 0.03, (top + bot) / 2),
                     (xa + gap - 0.03, (top + bot) / 2),
                     arrowstyle="-|>", mutation_scale=16, color=ACCENT, lw=2,
                     zorder=6))

    # banda de validacion
    vy = 2.35
    ax.add_patch(FancyBboxPatch((0.7, vy - 0.62), 14.6, 1.18,
                 boxstyle="round,pad=0.02,rounding_size=0.1",
                 fc="#2c2620", ec=AMBER, lw=1.5, zorder=3))
    ax.text(0.95, vy + 0.2, "VALIDATION ARM", color=AMBER, fontsize=12.5,
            fontweight="bold", va="center", zorder=6)
    ax.text(0.95, vy - 0.22,
            "H1 perception  ·  H2 bi-axial disruption  ·  H3 receiver advantage"
            "  ·  H4 progression–safety trade-off",
            color=WHITE, fontsize=11.5, va="center", zorder=6)
    ax.text(15.05, vy, "6,923 actions\n2,354 off-ball runs\ntheory holds",
            color=AMBER, fontsize=11, ha="right", va="center",
            fontweight="bold", zorder=6, linespacing=1.4)

    ax.text(8.0, 1.05, "Skeleton geometry only — the DOS model never sees a goal, "
            "an xG, an xT or an outcome label.",
            color=MUTE, fontsize=11.5, ha="center", va="center", zorder=6)
    footer(ax, 2)
    return save(fig, "slide2.png"), None


# ====================================================================
# SLIDES 3-5 -- video stages
# ====================================================================
def video_slide(idx, kick, ttl, caption, fig_png, vid_mp4):
    fig, ax = new_slide()
    kicker(ax, kick)
    title(ax, ttl)
    rect = place_image(ax, FIG / fig_png, 0.7, 15.3, 1.5, GY - 2.05)
    ax.text(8.0, 1.05, caption, color=MUTE, fontsize=12.2, ha="center",
            va="center", zorder=6)
    footer(ax, idx)
    png = save(fig, f"slide{idx}.png")
    return png, {"rect": rect, "video": VID / vid_mp4, "poster": FIG / fig_png}


# ====================================================================
# build
# ====================================================================
def grid_to_pptx(rect):
    """rect = [left, right, bottom, top] en grid 16x9 -> (left,top,w,h) pulgadas."""
    sx = SLIDE_W_IN / GX
    sy = SLIDE_H_IN / GY
    left = rect[0] * sx
    width = (rect[1] - rect[0]) * sx
    top = (GY - rect[3]) * sy
    height = (rect[3] - rect[2]) * sy
    return Inches(left), Inches(top), Inches(width), Inches(height)


def main():
    slides = [
        slide1(),
        slide2(),
        video_slide(3, "Stage 1 — Vision",
                    "What every player can see",
                    "Probabilistic field of view: a 120° cone on the measured head "
                    "yaw, speed-dependent decay, torso occlusion from real shoulder widths.",
                    "Vision.png", "Vision_Video.mp4"),
        video_slide(4, "Stage 2 — Pitch Control",
                    "What every player can reach",
                    "Each player an anisotropic reach field — the blob collapses in a "
                    "defender's blind spot, a literal hole in his control of space.",
                    "PPCF.png", "PPCF_Video.mp4"),
        video_slide(5, "Stages 3–4 — DOS",
                    "The Diagonal Opportunity Surface",
                    "Extra control bought going diagonal vs straight, gated to what the "
                    "on-ball player perceives. Cyan: seen now. Amber: opportunity lost track of.",
                    "DOS.png", "DOS_Video.mp4"),
    ]

    # --- PDF (5 paginas, estatico) ------------------------------------
    imgs = [Image.open(str(p)).convert("RGB") for p, _ in slides]
    pdf = OUT / "executive_summary.pdf"
    imgs[0].save(str(pdf), save_all=True, append_images=imgs[1:])
    print("PDF  ->", pdf)

    # --- PPTX (videos embebidos en 3-5) -------------------------------
    prs = Presentation()
    prs.slide_width = Inches(SLIDE_W_IN)
    prs.slide_height = Inches(SLIDE_H_IN)
    blank = prs.slide_layouts[6]
    for png, vid in slides:
        s = prs.slides.add_slide(blank)
        s.shapes.add_picture(str(png), 0, 0, width=prs.slide_width,
                             height=prs.slide_height)
        if vid is not None:
            left, top, w, h = grid_to_pptx(vid["rect"])
            s.shapes.add_movie(str(vid["video"]), left, top, w, h,
                               poster_frame_image=str(vid["poster"]),
                               mime_type="video/mp4")
    pptx = OUT / "executive_summary.pptx"
    prs.save(str(pptx))
    print("PPTX ->", pptx)


if __name__ == "__main__":
    main()
