"""
Build the Telegram "working…" sticker: a small "Thinking" label with three
dots that bounce in a wave, on a transparent background, looping cleanly.

Telegram video-sticker rules (Bot API): WEBM / VP9, one side exactly 512 px
and the other at most 512, at most 3 s, at most 30 fps, at most 256 KB, no
audio. Telegram shows a sticker at a fixed width, so the drawing sits small
and left-aligned in a 512-wide canvas (the rest is transparent) to read
like a normal line of chat text.

    .venv/bin/python scripts/make_typing_sticker.py                 # needs ffmpeg with libvpx-vp9
    .venv/bin/python scripts/make_typing_sticker.py --preview /tmp/thinking.gif
    .venv/bin/python scripts/make_typing_sticker.py --style plain   # no pill behind the text

Output: static/telegram/typing-dots.webm
"""
from __future__ import annotations

import argparse, math, os, shutil, subprocess, sys, tempfile

from PIL import Image, ImageDraw, ImageFont

W, H = 512, 92                # one side exactly 512
FPS, PERIOD_S = 30, 1.2       # the file is exactly one period, so it loops cleanly
SS = 4                        # supersampling for smooth edges
LABEL = "Thinking"
FONT_PX = 40
TEXT = (60, 60, 67)           # dark grey label (pill style)
TEXT_PLAIN = (138, 138, 142)  # mid grey: readable on light and dark when there is no pill
DOT = (239, 61, 28)           # OpenTeddy red (the mascot)
DOT_D, DOT_GAP, DOT_LEAD = 10, 16, 12   # diameter, centre spacing, space after the label
BOUNCE = 8                    # px a dot rises at the top of its hop
STAGGER, HOP = 0.16, 0.5      # each dot starts 16 % of a cycle later; a hop lasts half a cycle
PILL_PAD_X, PILL_PAD_Y = 22, 14
FONTS = [  # first one found wins; the label is rasterised into the sticker
    ("/System/Library/Fonts/Avenir Next.ttc", 5),          # macOS: Avenir Next Medium
    ("/System/Library/Fonts/HelveticaNeue.ttc", 10),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "static", "telegram", "typing-dots.webm")


def _font(px: int) -> ImageFont.FreeTypeFont:
    for path, index in FONTS:
        if os.path.exists(path):
            return ImageFont.truetype(path, px, index=index)
    sys.exit("no usable font found; edit FONTS")


def hop(t: float, i: int) -> float:
    """0..1 height of dot i at time t: a smooth hop, then rest."""
    p = (t / PERIOD_S - i * STAGGER) % 1.0
    return math.sin(math.pi * p / HOP) ** 2 if p < HOP else 0.0


def frame(t: float, style: str = "pill") -> Image.Image:
    s = SS
    img = Image.new("RGBA", (W * s, H * s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = _font(FONT_PX * s)
    asc, desc = font.getmetrics()
    label_w = d.textlength(LABEL, font=font)
    dots_w = (DOT_LEAD + 2 * DOT_GAP + DOT_D) * s
    content_w, content_h = label_w + dots_w, asc + desc

    x0 = 4 * s
    if style == "pill":
        pw, ph = content_w + 2 * PILL_PAD_X * s, content_h + 2 * PILL_PAD_Y * s
        py = (H * s - ph) / 2
        d.rounded_rectangle((x0, py, x0 + pw, py + ph), radius=ph / 2,
                            fill=(255, 255, 255, 245), outline=(0, 0, 0, 22), width=s)
        tx, ty = x0 + PILL_PAD_X * s, py + PILL_PAD_Y * s
        color = TEXT
    else:
        tx, ty = x0, (H * s - content_h) / 2
        color = TEXT_PLAIN
    d.text((tx, ty), LABEL, font=font, fill=color + (255,))

    # Dots go on their own layer and are composited: drawing a translucent
    # fill straight onto the pill would replace its pixels and punch holes.
    dots = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dd = ImageDraw.Draw(dots)
    baseline = ty + asc
    for i in range(3):
        k = hop(t, i)
        cx = tx + label_w + (DOT_LEAD + DOT_D / 2 + i * DOT_GAP) * s
        cy = baseline - (DOT_D / 2 + BOUNCE * k) * s
        r = DOT_D / 2 * s
        dd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=DOT + (round(255 * (0.5 + 0.5 * k)),))
    img.alpha_composite(dots)
    return img.resize((W, H), Image.LANCZOS)


def build(out: str = OUT, style: str = "pill") -> str:
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found")
    n = round(FPS * PERIOD_S)
    with tempfile.TemporaryDirectory() as tmp:
        for f in range(n):
            frame(f / FPS, style).save(os.path.join(tmp, f"f_{f:03d}.png"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", os.path.join(tmp, "f_%03d.png"),
             "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
             "-b:v", "0", "-crf", "28", "-an", out],
            check=True,
        )
    size = os.path.getsize(out)
    assert size <= 256 * 1024, f"{size} bytes > Telegram's 256 KB sticker limit"
    print(f"wrote {out} ({size / 1024:.1f} KB, {n} frames, {W}x{H}, {PERIOD_S}s @ {FPS} fps, {style})")
    return out


def preview(path: str, style: str = "pill") -> None:
    """GIF of the loop over a light, a dark and a green chat background,
    at roughly the size Telegram shows it (512 px → about 200 pt)."""
    bgs = [(255, 255, 255), (23, 33, 43), (168, 196, 140)]
    frames = []
    for f in range(round(FPS * PERIOD_S)):
        fr = frame(f / FPS, style)
        canvas = Image.new("RGB", (W, H * len(bgs)))
        for j, bg in enumerate(bgs):
            tile = Image.new("RGBA", (W, H), bg + (255,))
            tile.alpha_composite(fr)
            canvas.paste(tile.convert("RGB"), (0, j * H))
        frames.append(canvas)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / FPS), loop=0)
    print(f"wrote preview {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--style", choices=["pill", "plain"], default="pill")
    ap.add_argument("--preview", help="also write a GIF preview here")
    ap.add_argument("--preview-only", action="store_true", help="skip the WEBM")
    a = ap.parse_args()
    if not a.preview_only:
        build(a.out, a.style)
    if a.preview:
        preview(a.preview, a.style)
