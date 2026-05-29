"""Record a demo of the Next.js chat UI as MP4 + GIF for the README.

Run AFTER ``docker compose up -d`` is healthy:

    pip install playwright pillow
    playwright install chromium
    python scripts/capture_demo.py
    # → docs/demo.mp4, docs/demo.gif, docs/screenshot.png

The script drives a scripted walkthrough of all four modes — Router (SQL
adapter + decision graph), Agent (calculator trace), Auto (heuristic
dispatch), and LangGraph (state-graph viz) — typing each query with a
human-like delay so the recording reads naturally. Playwright records the
session to webm; ffmpeg then produces a high-quality MP4 and a compressed,
looping GIF for the README. gifsicle further shrinks the GIF if present.

Kept in ``scripts/`` (not the package) so it doesn't ship in the wheel.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

UI_URL = os.environ.get("AM_UI_URL", "http://localhost:7007")
DOCS = Path(__file__).resolve().parents[1] / "docs"
DOCS.mkdir(parents=True, exist_ok=True)

# (mode label, query, whether to expand the trace afterwards)
SCRIPT: list[tuple[str, str, bool]] = [
    ("Router", "What is the molecular formula of caffeine?", False),
    ("Router", "Top 5 customers by total revenue — write the SQL.", False),
    ("Agent", "Compute 2**16 + 17, then explain that figure as a finance metric.", True),
    ("Auto", "What is the capital of France?", False),
    ("LangGraph", "Find the SMILES for caffeine, then summarise its pharmacology.", True),
]


def _wait_for_idle(page, timeout_ms: int = 120_000) -> None:
    """Wait until the run completes (the 'running (…)' pulse disappears)."""
    page.wait_for_function(
        "() => !document.body.innerText.includes('running (')",
        timeout=timeout_ms,
    )


def _send(page, mode: str, query: str, expand_trace: bool) -> None:
    # Switch mode (accessible name includes the emoji, so match by substring).
    page.get_by_role("button", name=mode, exact=False).first.click()
    time.sleep(0.6)
    box = page.locator("textarea")
    box.click()
    box.fill("")
    box.type(query, delay=28)          # human-like keystrokes for the video
    time.sleep(0.4)
    box.press("Enter")                  # show off Enter-to-send
    _wait_for_idle(page)
    time.sleep(0.8)
    if expand_trace:
        try:
            page.get_by_text("show trace").first.click(timeout=2_000)
            time.sleep(0.9)
        except Exception:
            pass
    # Let the viewer read the answer.
    time.sleep(1.4)


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: install playwright first:\n"
              "  pip install playwright pillow\n"
              "  playwright install chromium", file=sys.stderr)
        return 1

    record_dir = DOCS / ".rec"
    if record_dir.exists():
        shutil.rmtree(record_dir)
    record_dir.mkdir(parents=True)

    size = {"width": 1440, "height": 900}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=size,
            device_scale_factor=2,        # crisp text in the recording
            record_video_dir=str(record_dir),
            record_video_size=size,
        )
        page = context.new_page()
        print(f"[capture_demo] opening {UI_URL}", flush=True)
        page.goto(UI_URL, wait_until="networkidle", timeout=30_000)
        page.wait_for_selector('body[data-testid="app-ready"]', timeout=10_000)
        time.sleep(1.2)                   # hold on the empty state

        hero_taken = False
        for mode, query, expand in SCRIPT:
            print(f"[capture_demo] {mode}: {query[:48]}…", flush=True)
            _send(page, mode, query, expand)
            # Grab the README hero off the first Router answer (graph visible).
            if not hero_taken and mode == "Router":
                page.screenshot(path=str(DOCS / "screenshot.png"), full_page=False)
                print(f"  → {DOCS / 'screenshot.png'}", flush=True)
                hero_taken = True

        time.sleep(1.0)
        context.close()                   # finalises the .webm
        browser.close()

    webm = next(iter(record_dir.glob("*.webm")), None)
    if not webm:
        print("ERROR: no .webm recorded", file=sys.stderr)
        return 1

    if not shutil.which("ffmpeg"):
        print("[capture_demo] ffmpeg not on PATH — keeping raw webm only at "
              f"{webm}", flush=True)
        return 0

    # --- High-quality MP4 (linkable in the README) ---
    mp4 = DOCS / "demo.mp4"
    print(f"[capture_demo] ffmpeg → {mp4}", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(webm),
        "-movflags", "+faststart",
        "-pix_fmt", "yuv420p",
        "-vf", "scale=1280:-2:flags=lanczos",
        "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
        str(mp4),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  → {mp4} ({mp4.stat().st_size / 1e6:.1f} MB)", flush=True)

    # --- Looping GIF (README hero), two-pass palette for clean colour ---
    gif = DOCS / "demo.gif"
    palette = record_dir / "palette.png"
    vf = "fps=12,scale=960:-1:flags=lanczos"
    print(f"[capture_demo] ffmpeg → {gif}", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(webm),
        "-vf", f"{vf},palettegen=max_colors=128",
        str(palette),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(webm), "-i", str(palette),
        "-lavfi", f"{vf}[x];[x][1:v]paletteuse=dither=bayer",
        "-loop", "0", str(gif),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    size_mb = gif.stat().st_size / 1e6
    print(f"  → {gif} ({size_mb:.1f} MB)", flush=True)

    if shutil.which("gifsicle"):
        subprocess.run(["gifsicle", "-O3", "--lossy=80",
                        "-o", str(gif), str(gif)], check=True)
        print(f"  → gifsicle compressed to "
              f"{gif.stat().st_size / 1e6:.1f} MB", flush=True)

    shutil.rmtree(record_dir, ignore_errors=True)
    print("[capture_demo] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
