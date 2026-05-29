"""Capture 1080p footage of the running Next.js chat UI for the demo video.

Run AFTER ``docker compose up -d`` is healthy:

    pip install playwright pillow
    playwright install chromium
    python scripts/capture_demo.py
    # → video/public/app.mp4  (raw footage, fed to Remotion)
    # → docs/screenshot.png   (static README fallback)

Then composite the branded intro/outro and render the final video+GIF:

    scripts/build_demo.sh        # capture → Remotion render → GIF (one shot)

The script drives a scripted walkthrough of all four modes — Router (SQL
adapter + decision graph), Agent (calculator trace), Auto (heuristic
dispatch), and LangGraph (state-graph viz) — typing each query with a
human-like delay so the recording reads naturally.

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
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
APP_MP4 = ROOT / "video" / "public" / "app.mp4"
DOCS.mkdir(parents=True, exist_ok=True)
APP_MP4.parent.mkdir(parents=True, exist_ok=True)

W, H = 1920, 1080

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
    page.get_by_role("button", name=mode, exact=False).first.click()
    time.sleep(0.6)
    box = page.locator("textarea")
    box.click()
    box.fill("")
    box.type(query, delay=30)           # human-like keystrokes for the video
    time.sleep(0.4)
    box.press("Enter")                  # show off Enter-to-send
    _wait_for_idle(page)
    time.sleep(0.8)
    if expand_trace:
        try:
            page.get_by_text("show trace").first.click(timeout=2_000)
            time.sleep(1.0)
        except Exception:
            pass
    time.sleep(1.5)                     # let the viewer read the answer


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

    size = {"width": W, "height": H}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=size,
            record_video_dir=str(record_dir),
            record_video_size=size,
        )
        page = context.new_page()
        print(f"[capture_demo] opening {UI_URL}", flush=True)
        page.goto(UI_URL, wait_until="networkidle", timeout=30_000)
        page.wait_for_selector('body[data-testid="app-ready"]', timeout=10_000)
        time.sleep(1.4)                  # hold on the empty state

        hero_taken = False
        for mode, query, expand in SCRIPT:
            print(f"[capture_demo] {mode}: {query[:48]}…", flush=True)
            _send(page, mode, query, expand)
            if not hero_taken and mode == "Router":
                page.screenshot(path=str(DOCS / "screenshot.png"), full_page=False)
                print(f"  → {DOCS / 'screenshot.png'}", flush=True)
                hero_taken = True

        time.sleep(1.0)
        context.close()                  # finalises the .webm
        browser.close()

    webm = next(iter(record_dir.glob("*.webm")), None)
    if not webm:
        print("ERROR: no .webm recorded", file=sys.stderr)
        return 1

    if not shutil.which("ffmpeg"):
        print(f"[capture_demo] ffmpeg not on PATH; raw footage at {webm}", file=sys.stderr)
        return 1

    # Transcode to a clean H.264 mp4 that Remotion consumes as static input.
    print(f"[capture_demo] ffmpeg → {APP_MP4}", flush=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(webm),
        "-vf", f"scale={W}:{H}:flags=lanczos,fps=30",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(APP_MP4),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  → {APP_MP4} ({APP_MP4.stat().st_size / 1e6:.1f} MB)", flush=True)

    shutil.rmtree(record_dir, ignore_errors=True)
    print("[capture_demo] done. Next: scripts/build_demo.sh "
          "(or `npm --prefix video run render`).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
