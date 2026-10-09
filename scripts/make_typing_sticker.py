"""
Build the Telegram "working…" sticker: three dots that fade and swell in
turn, on a transparent background, looping seamlessly.

Telegram video-sticker rules (Bot API): WEBM / VP9, one side exactly 512 px
and the other at most 512, at most 3 s, at most 30 fps, at most 256 KB, no
audio. Sent as a sticker it plays on a loop with no message bubble, which
editing a text message every couple of seconds can never imitate.

    .venv/bin/python scripts/make_typing_sticker.py          # needs ffmpeg with libvpx-vp9
    .venv/bin/python scripts/make_typing_sticker.py --preview /tmp/dots.gif

Output: static/telegram/typing-dots.webm
"""
from __future__ import annotations

import argparse, math, os, shutil, subprocess, sys, tempfile

from PIL import Image, ImageDraw

W, H = 512, 176            # one side exactly 512; a short strip, not a square
FPS, PERIOD_S = 30, 1.2    # one full cycle; the file is exactly one period, so it loops cleanly
SS = 4                     # supersampling for smooth edges
DOT_D, GAP = 64, 104       # dot diameter, centre-to-centre distance
COLOR = (142, 142, 147)    # mid grey: readable on light and dark chat backgrounds
STAGGER, PULSE = 0.18, 0.62  # each dot starts 18 % of a cycle after the previous; pulse lasts 62 %

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "static", "telegram", "typing-dots.webm")


def intensity(t: float, i: int) -> float:
    p = (t / PERIOD_S - i * STAGGER) % 1.0
    return math.sin(math.pi * p / PULSE) ** 2 if p < PULSE else 0.0


def frame(t: float) -> Image.Image:
    img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i in range(3):
        k = intensity(t, i)
        r = DOT_D / 2 * (0.82 + 0.18 * k) * SS
        cx, cy = (W / 2 + (i - 1) * GAP) * SS, H / 2 * SS
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=COLOR + (round(255 * (0.38 + 0.62 * k)),))
    return img.resize((W, H), Image.LANCZOS)


def build(out: str = OUT) -> str:
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found")
    n = round(FPS * PERIOD_S)
    with tempfile.TemporaryDirectory() as tmp:
        for f in range(n):
            frame(f / FPS).save(os.path.join(tmp, f"f_{f:03d}.png"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", os.path.join(tmp, "f_%03d.png"),
             "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
             "-b:v", "0", "-crf", "30", "-an", out],
            check=True,
        )
    size = os.path.getsize(out)
    assert size <= 256 * 1024, f"{size} bytes > Telegram's 256 KB sticker limit"
    print(f"wrote {out} ({size / 1024:.1f} KB, {n} frames, {W}x{H}, {PERIOD_S}s @ {FPS} fps)")
    return out


def preview(path: str) -> None:
    """GIF of the loop over a light, a dark and a green chat background."""
    bgs = [(255, 255, 255), (23, 33, 43), (168, 196, 140)]
    frames = []
    for f in range(round(FPS * PERIOD_S)):
        fr = frame(f / FPS)
        canvas = Image.new("RGB", (W, H * len(bgs)))
        for j, bg in enumerate(bgs):
            tile = Image.new("RGBA", (W, H), bg + (255,))
            tile.alpha_composite(fr)
            canvas.paste(tile.convert("RGB"), (0, j * H))
        frames.append(canvas.resize((W // 2, H * len(bgs) // 2), Image.LANCZOS))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / FPS), loop=0)
    print(f"wrote preview {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--preview", help="also write a GIF preview here")
    a = ap.parse_args()
    build(a.out)
    if a.preview:
        preview(a.preview)
