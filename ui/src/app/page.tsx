"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import {
  Background,
  Controls,
  MarkerType,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { api, type Adapter, type ChatMode, type ChatResult, type Step } from "@/lib/api";

// ---------- types -----------------------------------------------------------

type Turn =
  | { role: "user"; content: string; ts: number }
  | {
      role: "assistant";
      content: string;
      ts: number;
      mode: ChatMode;
      result: ChatResult;
    };

const SAMPLES: Record<ChatMode, string[]> = {
  router: [
    "What is the molecular formula of caffeine?",
    "Top 5 customers by total revenue — write the SQL.",
    "Cypher: count User nodes connected to Order via :PLACED.",
    "Explain a SPARQL query that returns the spouses of every U.S. president.",
  ],
  agent: [
    "Compute 2**16+17, then explain that figure as a finance metric.",
    "Find the SMILES for caffeine, then describe its mechanism of action.",
    "Run 'wc -l README.md' and tell me whether it's more than 200 lines.",
    "Plan an LP: maximise 3x + 5y subject to x + y ≤ 4, x, y ≥ 0.",
  ],
};

// ---------- helpers ---------------------------------------------------------

function formatElapsed(s?: number): string {
  if (s == null) return "";
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  return `${s.toFixed(1)} s`;
}

function stepTone(step: Step): "brain" | "expert" | "tool" | "other" {
  const k = (step.kind ?? step.label ?? "").toLowerCase();
  if (k.includes("router") || k.includes("brain")) return "brain";
  if (k === "expert" || k.startsWith("call [expert]")) return "expert";
  if (k === "tool" || k.startsWith("call [tool]")) return "tool";
  return "other";
}

// ---------- xyflow viz of the router decision ------------------------------

function RouterFlow({ result }: { result: ChatResult }) {
  const { nodes, edges } = useMemo(() => {
    const adapter = result.adapter_id ?? "—";
    const nodes: Node[] = [
      {
        id: "user",
        data: { label: "🧑 query" },
        position: { x: 0, y: 0 },
        style: nodeStyle("user"),
      },
      {
        id: "router",
        data: { label: "🧠 base · router" },
        position: { x: 220, y: 0 },
        style: nodeStyle("brain"),
      },
      {
        id: "adapter",
        data: { label: `🧬 ${adapter}` },
        position: { x: 460, y: 0 },
        style: nodeStyle("expert"),
      },
    ];
    const edges: Edge[] = [
      {
        id: "e1",
        source: "user",
        target: "router",
        animated: true,
        style: { stroke: "#a78bfa", strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#a78bfa" },
      },
      {
        id: "e2",
        source: "router",
        target: "adapter",
        animated: true,
        label: "selects",
        labelStyle: { fill: "#a3e635", fontSize: 11 },
        style: { stroke: "#34d399", strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#34d399" },
      },
    ];
    return { nodes, edges };
  }, [result.adapter_id]);

  return (
    <div className="h-32 w-full overflow-hidden rounded-xl border border-neutral-800/80 bg-neutral-950/60">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        panOnDrag={false}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={16} color="#1f2937" />
        <Controls position="bottom-right" showInteractive={false} />
      </ReactFlow>
    </div>
  );
}

function nodeStyle(kind: "user" | "brain" | "expert"): React.CSSProperties {
  const base: React.CSSProperties = {
    padding: 10,
    fontSize: 12,
    border: "1px solid",
    borderRadius: 12,
    width: 170,
    color: "#e5e7eb",
    fontWeight: 500,
    background: "rgba(20,22,32,0.85)",
    backdropFilter: "blur(4px)",
  };
  if (kind === "user")
    return { ...base, background: "rgba(99,102,241,0.16)", borderColor: "#6366f1", color: "#c7d2fe" };
  if (kind === "brain")
    return { ...base, background: "rgba(139,92,246,0.16)", borderColor: "#7c3aed", color: "#ddd6fe" };
  return { ...base, background: "rgba(16,185,129,0.16)", borderColor: "#047857", color: "#a7f3d0" };
}

// ---------- trace step chip ------------------------------------------------

function StepChip({ step }: { step: Step }) {
  const tone = stepTone(step);
  const colour =
    tone === "brain"
      ? "bg-violet-500/15 border-violet-400/30 text-violet-200"
      : tone === "expert"
      ? "bg-emerald-500/15 border-emerald-400/30 text-emerald-200"
      : tone === "tool"
      ? "bg-sky-500/15 border-sky-400/30 text-sky-200"
      : "bg-neutral-700/20 border-neutral-500/30 text-neutral-300";
  const name =
    step.name ?? step.adapter ?? step.decision ?? step.label ?? "step";
  const icon =
    tone === "brain" ? "🧠" : tone === "expert" ? "🧬" : tone === "tool" ? "🛠" : "•";
  return (
    <span
      className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${colour}`}
      title={step.observation || step.sub_query || step.raw_response}
    >
      {icon} {String(name).slice(0, 28)}
      {step.elapsed != null && (
        <span className="ml-1 opacity-60">{formatElapsed(step.elapsed)}</span>
      )}
    </span>
  );
}

// ---------- assistant turn -------------------------------------------------

function AssistantBubble({ turn }: { turn: Extract<Turn, { role: "assistant" }> }) {
  const [showTrace, setShowTrace] = useState(false);
  const r = turn.result;
  const ok = r.ok && r.response;
  return (
    <div className="panel space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <span className="rounded-full border border-neutral-700 bg-neutral-900/70 px-2 py-0.5 text-neutral-300">
          mode: <b className="text-neutral-100">{turn.mode}</b>
        </span>
        {r.adapter_id && (
          <span className="rounded-full border border-emerald-700/70 bg-emerald-900/30 px-2 py-0.5 text-emerald-200">
            adapter: <span className="font-mono">{r.adapter_id}</span>
          </span>
        )}
        {r.elapsed != null && (
          <span className="rounded-full border border-neutral-700 bg-neutral-900/70 px-2 py-0.5 text-neutral-400">
            {formatElapsed(r.elapsed)}
          </span>
        )}
      </div>

      {turn.mode === "router" && r.adapter_id && <RouterFlow result={r} />}

      <div className="text-sm">
        {ok ? (
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
              {r.response ?? ""}
            </ReactMarkdown>
          </div>
        ) : (
          <div className="whitespace-pre-wrap text-rose-300">
            {r.error ?? "(no response)"}
          </div>
        )}
      </div>

      {r.steps && r.steps.length > 0 && (
        <div>
          <button
            onClick={() => setShowTrace((s) => !s)}
            className="text-[11px] text-neutral-400 underline-offset-4 hover:underline"
          >
            {showTrace ? "▼ hide trace" : "▶ show trace"} ({r.steps.length} steps)
          </button>
          {showTrace && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {r.steps.map((s, i) => (
                <StepChip key={i} step={s} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------- main -----------------------------------------------------------

export default function Page() {
  const [mode, setMode] = useState<ChatMode>(
    (process.env.NEXT_PUBLIC_DEFAULT_MODE as ChatMode) === "agent" ? "agent" : "router",
  );
  const [query, setQuery] = useState("");
  const [chat, setChat] = useState<Turn[]>([]);
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);

  // load /health + /adapters once
  useEffect(() => {
    Promise.all([api.health(), api.adapters()])
      .then(([h, a]) => {
        setHealthy(h.ok);
        setAdapters(a);
        setError(null);
      })
      .catch((e: unknown) => {
        setHealthy(false);
        setError(e instanceof Error ? e.message : String(e));
      });
  }, []);

  // auto-scroll on new turn
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chat]);

  const onSubmit = useCallback(async () => {
    const q = query.trim();
    if (!q || running) return;
    const userTurn: Turn = { role: "user", content: q, ts: Date.now() };
    setChat((c) => [...c, userTurn]);
    setQuery("");
    setRunning(true);
    try {
      const r = await api.chat(q, mode);
      const asst: Turn = {
        role: "assistant",
        content: r.response ?? "",
        ts: Date.now(),
        mode,
        result: r,
      };
      setChat((c) => [...c, asst]);
    } catch (e: unknown) {
      const asst: Turn = {
        role: "assistant",
        content: "",
        ts: Date.now(),
        mode,
        result: {
          ok: false,
          mode,
          error: e instanceof Error ? e.message : String(e),
        },
      };
      setChat((c) => [...c, asst]);
    } finally {
      setRunning(false);
    }
  }, [query, mode, running]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      void onSubmit();
    }
  };

  return (
    <main className="flex h-screen flex-col">
      {/* HEADER */}
      <header className="relative z-50 flex shrink-0 items-center justify-between border-b border-neutral-800/80 bg-neutral-950/40 px-4 py-2.5 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-emerald-500 via-sky-500 to-violet-500 text-base">
            🧠
          </div>
          <div>
            <h1 className="text-sm font-semibold leading-none">Adaptive Minds</h1>
            <p className="text-[11px] text-neutral-400">
              {adapters.length ? `${adapters.length} adapters` : "loading adapters…"} · base = Qwen2.5-7B-Instruct ·
              vLLM via FastAPI
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs">
          {/* Mode toggle */}
          <div className="flex items-center rounded-lg border border-neutral-700 bg-neutral-900/70 p-0.5">
            {(["router", "agent"] as ChatMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={
                  "rounded-md px-2.5 py-1 text-sm transition " +
                  (mode === m
                    ? "bg-emerald-700/40 text-emerald-100"
                    : "text-neutral-300 hover:bg-neutral-800 hover:text-neutral-100")
                }
              >
                {m === "router" ? "🎯 Router" : "🤖 Agent"}
              </button>
            ))}
          </div>
          {/* Health pill */}
          <span
            className={
              "rounded-full px-2 py-1 text-[11px] " +
              (healthy === true
                ? "bg-emerald-900/40 text-emerald-300"
                : healthy === false
                ? "bg-rose-900/40 text-rose-300"
                : "bg-neutral-800 text-neutral-400")
            }
          >
            {healthy === true ? "● online" : healthy === false ? "● offline" : "● …"}
          </span>
        </div>
      </header>

      {/* BODY */}
      <div className="grid min-h-0 flex-1 grid-cols-[260px_1fr] gap-3 overflow-hidden p-3">
        {/* SIDEBAR — adapters */}
        <aside className="panel flex min-h-0 flex-col overflow-hidden">
          <div className="shrink-0 border-b border-neutral-800/80 px-3 py-2 text-xs font-semibold text-neutral-300">
            Adapters{" "}
            <span className="text-neutral-500">({adapters.length})</span>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {error && (
              <div className="p-3 text-xs text-rose-300">{error}</div>
            )}
            {adapters.map((a) => (
              <div
                key={a.id}
                className="border-b border-neutral-900 px-3 py-1.5 text-xs"
                title={a.description}
              >
                <div className="font-mono text-[11px] text-neutral-100">
                  {a.id}
                </div>
                <div className="truncate text-[10px] text-neutral-500">
                  {a.description}
                </div>
              </div>
            ))}
          </div>
          <div className="shrink-0 border-t border-neutral-800/80 p-3 text-[10px] text-neutral-500">
            <p>
              <b>Router</b> — one LLM call picks an adapter; that adapter
              answers.
            </p>
            <p className="mt-1">
              <b>Agent</b> — ReAct loop over adapters + tools (calculator,
              code, shell, websearch, pulp).
            </p>
          </div>
        </aside>

        {/* CHAT */}
        <section className="panel flex min-h-0 flex-col overflow-hidden">
          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
            {chat.length === 0 && (
              <div className="space-y-2 text-xs text-neutral-400">
                <p>Try one of these in {mode} mode:</p>
                <div className="flex flex-wrap gap-1.5">
                  {SAMPLES[mode].map((s) => (
                    <button
                      key={s}
                      onClick={() => setQuery(s)}
                      className="rounded-full border border-neutral-700 bg-neutral-900/70 px-2 py-0.5 text-[11px] hover:border-neutral-500"
                    >
                      {s.slice(0, 60)}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <AnimatePresence initial={false}>
              {chat.map((t, i) =>
                t.role === "user" ? (
                  <motion.div
                    key={`u-${i}`}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="ml-auto max-w-3xl rounded-2xl border border-sky-700/60 bg-sky-950/30 px-4 py-2 text-sm text-sky-100"
                  >
                    {t.content}
                  </motion.div>
                ) : (
                  <motion.div
                    key={`a-${i}`}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    <AssistantBubble turn={t} />
                  </motion.div>
                ),
              )}
            </AnimatePresence>
            {running && (
              <div className="text-xs text-neutral-500">running ({mode})…</div>
            )}
            <div ref={endRef} />
          </div>
          <div className="shrink-0 border-t border-neutral-800/80 p-3">
            <div className="flex items-end gap-2">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={
                  mode === "router"
                    ? "Ask a domain question. Cmd/Ctrl + ↵ to send."
                    : "Ask something that needs tools or multiple experts. Cmd/Ctrl + ↵ to send."
                }
                className="h-20 w-full resize-y rounded-lg border border-neutral-700 bg-neutral-900 p-2 text-sm focus:border-emerald-500 focus:outline-none"
              />
              <button
                onClick={() => void onSubmit()}
                disabled={!query.trim() || running}
                className="h-20 shrink-0 rounded-lg border border-emerald-700/70 bg-emerald-950/40 px-4 text-sm font-semibold text-emerald-100 hover:border-emerald-500 hover:bg-emerald-900/40 disabled:opacity-50"
              >
                {running ? "…" : "Send"}
              </button>
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
