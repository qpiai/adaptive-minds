# Scripts

One-offs that don't ship in the wheel.

| script | what it does |
|---|---|
| `capture_demo.py` | Drive the running Next.js UI (port 7007) with Playwright + Chromium, record a 25-s demo of Router + Agent flows, and write `docs/{router,agent_trace,screenshot}.png` + `docs/demo.gif` (if `ffmpeg` is on PATH). |

```bash
# After `docker compose up -d` is healthy:
pip install playwright pillow
playwright install chromium

python scripts/capture_demo.py
# → docs/router.png, docs/agent_trace.png, docs/screenshot.png, docs/demo.gif
```

The README's hero image (`docs/screenshot.png`) is just a copy of
`docs/agent_trace.png` — the agent screenshot is the more interesting one
because it shows the multi-step trace expander.
