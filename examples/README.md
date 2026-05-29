# Examples

Three runnable examples assuming a vLLM server is already up:

```bash
$(adaptive-minds serve --catalog catalogs/qwen25_30.yaml) &
export VLLM_BASE=http://localhost:8000/v1
```

| File                 | What it does                                            |
|----------------------|---------------------------------------------------------|
| `basic_router.py`    | Three single-step routing examples (SQL, Cypher, PII).  |
| `basic_agent.py`     | Multi-step agent: calculator → finance expert → FINAL.  |
| `curl_examples.sh`   | Direct vLLM curl calls (no Python).                     |

For a minimal smoke test before running the full 30-adapter catalog,
disable all but two adapters in `catalogs/qwen25_30.yaml`
(`enabled: false`) so vLLM only downloads what you need.
