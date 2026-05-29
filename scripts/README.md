# Scripts

One-offs that don't ship in the wheel.

| script | what it does |
|---|---|
| `capture_demo.py` | Drive the running Next.js UI (port 7007) with Playwright + Chromium through all four modes (Router → Agent → Auto → LangGraph), and write 1080p footage to `video/public/app.mp4` + a static `docs/screenshot.png`. |
| `build_demo.sh` | One-shot pipeline: capture → Remotion render (branded intro/outro) → high-quality `docs/demo.mp4` + `docs/demo.gif`. |

```bash
# After `docker compose up -d` is healthy:
pip install playwright pillow
playwright install chromium

scripts/build_demo.sh
# → docs/demo.mp4, docs/demo.gif, docs/screenshot.png
```

Tunables (env): `AM_UI_URL` (default `http://localhost:7007`), `GIF_WIDTH`,
`GIF_FPS`. The Remotion project lives in [`video/`](../video/) — see its
README to restyle the intro/outro.
