"""Record a 25-second demo GIF of the Next.js chat UI.

Run AFTER ``docker compose up -d`` is healthy:

    pip install playwright pillow
    playwright install chromium
    python scripts/capture_demo.py
    # → docs/router.png, docs/agent_trace.png, docs/demo.gif (if ffmpeg)

The script drives the chat in two modes (Router → SQL adapter, Agent →
calculator + adapter), waits for streaming responses, and screenshots the
trace expander. If ffmpeg is on PATH, the recorded webm is converted to a
≤ 5 MB GIF for the README.

This is the script the prompt's "Phase 2c" asks for — kept in
``scripts/`` (not the package) so it doesn't ship in the wheel.
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

ROUTER_QUERY = "What is the molecular formula of caffeine?"
AGENT_QUERY = "Compute 2**16+17, then explain that figure as a finance metric."


def _wait_for_response(page, timeout_ms: int = 60_000) -> None:
    """Wait until an assistant bubble appears AND the streaming pulse dot
    is gone (i.e. the run is complete)."""
    page.wait_for_selector(".panel", state="visible", timeout=timeout_ms)
    page.wait_for_function(
        "() => !document.body.innerText.includes('running (')",
        timeout=timeout_ms,
    )


def main() -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("ERROR: install playwright first:\n"
              "  pip install playwright pillow\n"
              "  playwright install chromium", file=sys.stderr)
        return 1

    record_dir = DOCS / ".rec"
    record_dir.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(record_dir),
            record_video_size={"width": 1440, "height": 900},
        )
        page = context.new_page()
        print(f"[capture_demo] opening {UI_URL}", flush=True)
        page.goto(UI_URL, wait_until="networkidle", timeout=30_000)
        # The Next.js root layout sets data-testid="app-ready" on <body>.
        page.wait_for_selector('body[data-testid="app-ready"]', timeout=10_000)
        time.sleep(0.5)

        # --- Router demo ---
        print("[capture_demo] router demo …", flush=True)
        # Mode toggle defaults to Router on first paint; click anyway to be sure.
        page.get_by_role("button", name="Router").click()
        page.locator("textarea").fill(ROUTER_QUERY)
        page.get_by_role("button", name="Send").click()
        _wait_for_response(page)
        time.sleep(0.5)
        page.screenshot(path=str(DOCS / "router.png"), full_page=False)
        print(f"  → {DOCS / 'router.png'}", flush=True)

        # --- Agent demo ---
        print("[capture_demo] agent demo …", flush=True)
        page.get_by_role("button", name="Agent").click()
        page.locator("textarea").fill(AGENT_QUERY)
        page.get_by_role("button", name="Send").click()
        _wait_for_response(page, timeout_ms=120_000)
        time.sleep(0.5)
        # Expand the trace so the screenshot shows the agent's CALL chain.
        try:
            page.get_by_text("show trace").first.click(timeout=2_000)
            time.sleep(0.3)
        except Exception:
            pass
        page.screenshot(path=str(DOCS / "agent_trace.png"), full_page=False)
        print(f"  → {DOCS / 'agent_trace.png'}", flush=True)

        # Pick the agent screenshot as the README hero — it shows the trace.
        shutil.copyfile(DOCS / "agent_trace.png", DOCS / "screenshot.png")
        print(f"  → {DOCS / 'screenshot.png'}", flush=True)

        # Close so the .webm is finalised.
        context.close()
        browser.close()

    # Convert the recorded .webm to a sub-5MB GIF if ffmpeg is available.
    webm = next(iter(record_dir.glob("*.webm")), None)
    gif = DOCS / "demo.gif"
    if webm and shutil.which("ffmpeg"):
        print(f"[capture_demo] ffmpeg → {gif}", flush=True)
        subprocess.run([
            "ffmpeg", "-y", "-i", str(webm),
            "-vf", "fps=10,scale=960:-1:flags=lanczos",
            "-loop", "0", str(gif),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        size_mb = gif.stat().st_size / 1e6
        print(f"  → {gif} ({size_mb:.1f} MB)", flush=True)
        if size_mb > 5 and shutil.which("gifsicle"):
            subprocess.run(["gifsicle", "-O3", "--lossy=80",
                            "-o", str(gif), str(gif)], check=True)
            print(f"  → gifsicle compressed to "
                  f"{gif.stat().st_size / 1e6:.1f} MB", flush=True)
    else:
        print("[capture_demo] ffmpeg not on PATH — skipping GIF; "
              "screenshots are enough.", flush=True)

    # Leave .rec/ behind for inspection; user can rm -rf if they want.
    print("[capture_demo] done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
