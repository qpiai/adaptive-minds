# Demo video (Remotion)

Composites a branded **intro + outro** around the Playwright-captured app
footage and renders the README's `docs/demo.mp4` / `docs/demo.gif`.

```
intro (4s)  →  real app walkthrough (all 4 modes)  →  outro (6s)
🧠 brand        Router · Agent · Auto · LangGraph       stats + CTA + links
```

## Build it

From the repo root, with the docker stack healthy on `:7007`:

```bash
scripts/build_demo.sh
# 1. capture 1080p footage  → video/public/app.mp4
# 2. remotion render        → docs/demo.mp4
# 3. ffmpeg + gifsicle      → docs/demo.gif
```

Or step by step:

```bash
python scripts/capture_demo.py        # → video/public/app.mp4
npm --prefix video install            # first time only
npm --prefix video run render         # → docs/demo.mp4
npm --prefix video run studio         # live-edit the intro/outro
```

## Layout

| file | role |
|---|---|
| `src/index.ts` | Remotion entry (`registerRoot`). |
| `src/Root.tsx` | `<Composition>`; sizes the timeline to intro + footage + outro via `calculateMetadata`. |
| `src/Demo.tsx` | The intro, the `OffthreadVideo` app clip (sped 1.35×), and the outro. |
| `public/app.mp4` | Captured footage (gitignored — regenerate with the capture script). |

Edit `Demo.tsx` to restyle the brand cards, stats, or CTA.
