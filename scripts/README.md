# Scripts

One-offs that don't ship in the wheel.

| script | what it does |
|---|---|
| `capture_demo.py` | Drive the running Next.js UI (port 7007) with Playwright + Chromium, record a scripted walkthrough of all four modes (Router → Agent → Auto → LangGraph), and write `docs/demo.mp4` (high-quality), `docs/demo.gif` (README hero), and `docs/screenshot.png` (static fallback) via `ffmpeg` + `gifsicle`. |

```bash
# After `docker compose up -d` is healthy:
pip install playwright pillow
playwright install chromium

python scripts/capture_demo.py
# → docs/demo.mp4, docs/demo.gif, docs/screenshot.png
```

The GIF is the inline README hero (autoplays + loops on GitHub); the MP4
is linked beside it for full quality. Set `AM_UI_URL` if the UI isn't on
the default `http://localhost:7007`.
