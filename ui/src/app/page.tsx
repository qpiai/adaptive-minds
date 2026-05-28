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

import {
  api,
  MODES,
  type Adapter,
  type ChatMode,
  type ChatResult,
  type Step,
} from "@/lib/api";

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
  ],
  agent: [
    "Compute 2**16+17, then explain that figure as a finance metric.",
    "Find the SMILES for caffeine, then describe its mechanism of action.",
    "Run 'wc -l README.md' and tell me whether it's more than 200 lines.",
  ],
  auto: [
    "What is the capital of France?",
    "Compute 2**10, then explain the result.",
    "Write SQL for top 5 customers by revenue.",
  ],
  langgraph: [
    "Plan an LP: maximise 3x + 5y subject to x + y ≤ 4, x, y ≥ 0.",
    "Find the SMILES for caffeine, then summarise its pharmacology.",
    "Compute fibonacci(30), then write SQL that filters orders ≥ that value.",
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
  if (k.includes("router") || k.includes("brain") || k.includes("plan") || k.includes("synthesise")) return "brain";
  if (k === "expert" || k.startsWith("call [expert]")) return "expert";
  if (k === "tool" || k.startsWith("call [tool]")) return "tool";
  return "other";
}

// ---------- xyflow viz of the routing decision -----------------------------

function RouterFlow({ result }: { result: ChatResult }) {
  const { nodes, edges } = useMemo(() => {
    const adapter = result.adapter_id ?? "—";
    const nodes: Node[] = [
      { id: "user",    data: { label: "🧑 query" },         position: { x: 0,   y: 0 }, style: nodeStyle("user") },
      { id: "router",  data: { label: "🧠 base · router" }, position: { x: 220, y: 0 }, style: nodeStyle("brain") },
      { id: "adapter", data: { label: `🧬 ${adapter}` },    position: { x: 460, y: 0 }, style: nodeStyle("expert") },
    ];
    const edges: Edge[] = [
      { id: "e1", source: "user", target: "router",
        animated: true,
        style: { stroke: "#a78bfa", strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#a78bfa" } },
      { id: "e2", source: "router", target: "adapter",
        animated: true, label: "selects",
        labelStyle: { fill: "#a3e635", fontSize: 11 },
        style: { stroke: "#34d399", strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: "#34d399" } },
    ];
    return { nodes, edges };
  }, [result.adapter_id]);

  return <FlowCanvas nodes={nodes} edges={edges} height={130} />;
}

function LangGraphFlow({ result }: { result: ChatResult }) {
  // Render plan → dispatch → synthesise as a small horizontal pipeline; we
  // colour each node by whether the trace actually visited it.
  const { nodes, edges } = useMemo(() => {
    const labels = (result.steps ?? []).map((s) => (s.label ?? "") + " " + (s.kind ?? ""));
    const visited = (key: string) =>
      labels.some((l) => l.toLowerCase().includes(key));
    const planVisited = visited("plan") || visited("brain");
    const dispatchVisited = visited("expert") || visited("tool") || visited("call");
    const synthVisited = visited("synthesise") || visited("commit");
    const nodes: Node[] = [
      { id: "start",      data: { label: "⏱ start" },         position: { x: 0,   y: 0 }, style: nodeStyle("user") },
      { id: "plan",       data: { label: "📋 plan" },          position: { x: 180, y: 0 }, style: nodeStyle(planVisited ? "brain" : "muted") },
      { id: "dispatch",   data: { label: "🛠 dispatch" },      position: { x: 360, y: 0 }, style: nodeStyle(dispatchVisited ? "expert" : "muted") },
      { id: "synthesise", data: { label: "✨ synthesise" },     position: { x: 540, y: 0 }, style: nodeStyle(synthVisited ? "brain" : "muted") },
      { id: "end",        data: { label: "✅ FINAL" },          position: { x: 720, y: 0 }, style: nodeStyle("done") },
    ];
    const edge = (id: string, s: string, t: string, on: boolean): Edge => ({
      id, source: s, target: t, animated: on,
      style: { stroke: on ? "#34d399" : "#3f3f4a", strokeWidth: on ? 2 : 1, strokeDasharray: on ? undefined : "4 4" },
      markerEnd: { type: MarkerType.ArrowClosed, color: on ? "#34d399" : "#3f3f4a" },
    });
    const edges: Edge[] = [
      edge("e1", "start", "plan", true),
      edge("e2", "plan", "dispatch", dispatchVisited),
      edge("e3", "dispatch", "plan", dispatchVisited && planVisited),
      edge("e4", "dispatch", "synthesise", synthVisited),
      edge("e5", "plan", "end", !dispatchVisited && !synthVisited),
      edge("e6", "synthesise", "end", synthVisited),
    ];
    return { nodes, edges };
  }, [result.steps]);

  return <FlowCanvas nodes={nodes} edges={edges} height={150} />;
}

function FlowCanvas({ nodes, edges, height }: { nodes: Node[]; edges: Edge[]; height: number }) {
  return (
    <div className="overflow-hidden rounded-xl border border-neutral-800/80 bg-neutral-950/60"
         style={{ height }}>
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

function nodeStyle(kind: "user" | "brain" | "expert" | "muted" | "done"): React.CSSProperties {
  const base: React.CSSProperties = {
    padding: 8, fontSize: 12, border: "1px solid", borderRadius: 12,
    width: 150, color: "#e5e7eb", fontWeight: 500,
    background: "rgba(20,22,32,0.85)",
  };
  if (kind === "user")
    return { ...base, background: "rgba(99,102,241,0.16)",  borderColor: "#6366f1", color: "#c7d2fe" };
  if (kind === "brain")
    return { ...base, background: "rgba(139,92,246,0.16)",  borderColor: "#7c3aed", color: "#ddd6fe" };
  if (kind === "expert")
    return { ...base, background: "rgba(16,185,129,0.16)",  borderColor: "#047857", color: "#a7f3d0" };
  if (kind === "done")
    return { ...base, background: "rgba(34,197,94,0.16)",   borderColor: "#16a34a", color: "#bbf7d0" };
  // muted
  return { ...base, background: "rgba(38,38,51,0.4)", borderColor: "#3f3f4a", color: "#737280" };
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
  const name = step.name ?? step.adapter ?? step.decision ?? step.label ?? "step";
  const icon = tone === "brain" ? "🧠" : tone === "expert" ? "🧬" : tone === "tool" ? "🛠" : "•";
  return (
    <span
      className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${colour}`}
      title={step.observation || step.sub_query || step.raw_response}
    >
      {icon} {String(name).slice(0, 28)}
      {step.elapsed != null && <span className="ml-1 opacity-60">{formatElapsed(step.elapsed)}</span>}
    </span>
  );
}

// ---------- assistant turn -------------------------------------------------

function AssistantBubble({ turn }: { turn: Extract<Turn, { role: "assistant" }> }) {
  const [showTrace, setShowTrace] = useState(false);
  const r = turn.result;
  const ok = r.ok && r.response;
  const showRouterFlow = (turn.mode === "router" || (turn.mode === "auto" && r.auto_decision?.picked === "router")) && !!r.adapter_id;
  const showLangGraphFlow = turn.mode === "langgraph" || (turn.mode === "auto" && r.auto_decision?.picked === "agent");
  return (
    <div className="panel space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <span className="rounded-full border border-neutral-700 bg-neutral-900/70 px-2 py-0.5 text-neutral-300">
          mode: <b className="text-neutral-100">{turn.mode}</b>
        </span>
        {r.auto_decision && (
          <span className="rounded-full border border-amber-700/70 bg-amber-900/30 px-2 py-0.5 text-amber-200">
            auto → <b>{r.auto_decision.picked}</b>{" "}
            <span className="opacity-60">({r.auto_decision.reason})</span>
          </span>
        )}
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

      {showRouterFlow && <RouterFlow result={r} />}
      {showLangGraphFlow && <LangGraphFlow result={r} />}

      <div className="text-sm">
        {ok ? (
          <div className="prose prose-invert prose-sm max-w-none">
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
              {r.response ?? ""}
            </ReactMarkdown>
          </div>
        ) : (
          <div className="whitespace-pre-wrap text-rose-300">{r.error ?? "(no response)"}</div>
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
              {r.steps.map((s, i) => <StepChip key={i} step={s} />)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------- main -----------------------------------------------------------

export default function Page() {
  const defaultMode = (process.env.NEXT_PUBLIC_DEFAULT_MODE as ChatMode) ?? "router";
  const [mode, setMode] = useState<ChatMode>(
    MODES.some((m) => m.id === defaultMode) ? defaultMode : "router",
  );
  const [query, setQuery] = useState("");
  const [chat, setChat] = useState<Turn[]>([]);
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    Promise.all([api.health(), api.adapters()])
      .then(([h, a]) => { setHealthy(h.ok); setAdapters(a); setError(null); })
      .catch((e: unknown) => { setHealthy(false); setError(e instanceof Error ? e.message : String(e)); });
  }, []);

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
      setChat((c) => [...c, { role: "assistant", content: r.response ?? "", ts: Date.now(), mode, result: r }]);
    } catch (e: unknown) {
      setChat((c) => [...c, {
        role: "assistant", content: "", ts: Date.now(), mode,
        result: { ok: false, mode, error: e instanceof Error ? e.message : String(e) },
      }]);
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

  const activeMode = MODES.find((m) => m.id === mode)!;

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
              {adapters.length ? `${adapters.length} adapters` : "loading…"} · base = Qwen2.5-7B-Instruct ·
              FastAPI + vLLM
            </p>
          </div>
        </div>
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
      </header>

      {/* MODE TABS */}
      <nav className="relative z-40 flex shrink-0 items-center gap-1 border-b border-neutral-800/80 bg-neutral-950/40 px-3 py-2 backdrop-blur">
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={
              "group relative rounded-lg px-3 py-1.5 text-sm transition " +
              (mode === m.id
                ? "bg-emerald-700/30 text-emerald-100 ring-1 ring-emerald-600/60"
                : "text-neutral-300 hover:bg-neutral-800/60 hover:text-neutral-100")
            }
          >
            <span className="mr-1">{m.icon}</span>
            {m.label}
          </button>
        ))}
        <div className="ml-3 text-[11px] text-neutral-500">{activeMode.blurb}</div>
      </nav>

      {/* BODY */}
      <div className="grid min-h-0 flex-1 grid-cols-[260px_1fr] gap-3 overflow-hidden p-3">
        {/* SIDEBAR */}
        <aside className="panel flex min-h-0 flex-col overflow-hidden">
          <div className="shrink-0 border-b border-neutral-800/80 px-3 py-2 text-xs font-semibold text-neutral-300">
            Adapters <span className="text-neutral-500">({adapters.length})</span>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {error && <div className="p-3 text-xs text-rose-300">{error}</div>}
            {adapters.map((a) => (
              <div
                key={a.id}
                className="border-b border-neutral-900 px-3 py-1.5 text-xs"
                title={a.description}
              >
                <div className="font-mono text-[11px] text-neutral-100">{a.id}</div>
                <div className="truncate text-[10px] text-neutral-500">{a.description}</div>
              </div>
            ))}
          </div>
          <div className="shrink-0 border-t border-neutral-800/80 p-3 text-[10px] text-neutral-500">
            Pick a mode above. Cmd/Ctrl + ↵ to send.
          </div>
        </aside>

        {/* CHAT */}
        <section className="panel flex min-h-0 flex-col overflow-hidden">
          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
            {chat.length === 0 && (
              <div className="space-y-2 text-xs text-neutral-400">
                <p>Try one of these in <b className="text-neutral-200">{activeMode.label}</b> mode:</p>
                <div className="flex flex-wrap gap-1.5">
                  {SAMPLES[mode].map((s) => (
                    <button
                      key={s}
                      onClick={() => setQuery(s)}
                      className="rounded-full border border-neutral-700 bg-neutral-900/70 px-2 py-0.5 text-[11px] hover:border-neutral-500"
                    >
                      {s.length > 60 ? s.slice(0, 60) + "…" : s}
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
              <div className="flex items-center gap-2 text-xs text-neutral-500">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                running ({mode})…
              </div>
            )}
            <div ref={endRef} />
          </div>

          <div className="shrink-0 border-t border-neutral-800/80 p-3">
            <div className="flex items-end gap-2">
              <textarea
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={`Ask in ${activeMode.label} mode. Cmd/Ctrl + ↵ to send.`}
                className="h-20 w-full resize-y rounded-lg border border-neutral-700 bg-neutral-900 p-2 text-sm focus:border-emerald-500 focus:outline-none"
              />
              <button
                onClick={() => void onSubmit()}
                disabled={!query.trim() || running}
                className="h-20 shrink-0 rounded-lg border border-emerald-700/70 bg-gradient-to-br from-emerald-700/40 to-sky-700/40 px-4 text-sm font-semibold text-emerald-100 hover:from-emerald-600/50 hover:to-sky-600/50 disabled:opacity-50"
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
