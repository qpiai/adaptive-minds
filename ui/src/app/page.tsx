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

type Theme = "light" | "dark";

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

const STORAGE_KEY = "am-chat-v1";
const THEME_KEY = "am-theme";

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

function edgePalette(theme: Theme) {
  return {
    violet: "#8b5cf6",
    emerald: "#10b981",
    label: theme === "dark" ? "#a3e635" : "#4d7c0f",
    idle: theme === "dark" ? "#3f3f4a" : "#cbd5e1",
    grid: theme === "dark" ? "#1f2937" : "#e2e8f0",
  };
}

function RouterFlow({ result, theme }: { result: ChatResult; theme: Theme }) {
  const { nodes, edges } = useMemo(() => {
    const p = edgePalette(theme);
    const adapter = result.adapter_id ?? "—";
    const nodes: Node[] = [
      { id: "user",    data: { label: "🧑 query" },         position: { x: 0,   y: 0 }, style: nodeStyle("user", theme) },
      { id: "router",  data: { label: "🧠 base · router" }, position: { x: 220, y: 0 }, style: nodeStyle("brain", theme) },
      { id: "adapter", data: { label: `🧬 ${adapter}` },    position: { x: 460, y: 0 }, style: nodeStyle("expert", theme) },
    ];
    const edges: Edge[] = [
      { id: "e1", source: "user", target: "router",
        animated: true,
        style: { stroke: p.violet, strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: p.violet } },
      { id: "e2", source: "router", target: "adapter",
        animated: true, label: "selects",
        labelStyle: { fill: p.label, fontSize: 11 },
        style: { stroke: p.emerald, strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: p.emerald } },
    ];
    return { nodes, edges };
  }, [result.adapter_id, theme]);

  return <FlowCanvas nodes={nodes} edges={edges} height={130} theme={theme} />;
}

function LangGraphFlow({ result, theme }: { result: ChatResult; theme: Theme }) {
  // Render plan → dispatch → synthesise as a small horizontal pipeline; we
  // colour each node by whether the trace actually visited it.
  const { nodes, edges } = useMemo(() => {
    const p = edgePalette(theme);
    const labels = (result.steps ?? []).map((s) => (s.label ?? "") + " " + (s.kind ?? ""));
    const visited = (key: string) =>
      labels.some((l) => l.toLowerCase().includes(key));
    const planVisited = visited("plan") || visited("brain");
    const dispatchVisited = visited("expert") || visited("tool") || visited("call");
    const synthVisited = visited("synthesise") || visited("commit");
    const nodes: Node[] = [
      { id: "start",      data: { label: "⏱ start" },         position: { x: 0,   y: 0 }, style: nodeStyle("user", theme) },
      { id: "plan",       data: { label: "📋 plan" },          position: { x: 180, y: 0 }, style: nodeStyle(planVisited ? "brain" : "muted", theme) },
      { id: "dispatch",   data: { label: "🛠 dispatch" },      position: { x: 360, y: 0 }, style: nodeStyle(dispatchVisited ? "expert" : "muted", theme) },
      { id: "synthesise", data: { label: "✨ synthesise" },     position: { x: 540, y: 0 }, style: nodeStyle(synthVisited ? "brain" : "muted", theme) },
      { id: "end",        data: { label: "✅ FINAL" },          position: { x: 720, y: 0 }, style: nodeStyle("done", theme) },
    ];
    const edge = (id: string, s: string, t: string, on: boolean): Edge => ({
      id, source: s, target: t, animated: on,
      style: { stroke: on ? p.emerald : p.idle, strokeWidth: on ? 2 : 1, strokeDasharray: on ? undefined : "4 4" },
      markerEnd: { type: MarkerType.ArrowClosed, color: on ? p.emerald : p.idle },
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
  }, [result.steps, theme]);

  return <FlowCanvas nodes={nodes} edges={edges} height={150} theme={theme} />;
}

function FlowCanvas({ nodes, edges, height, theme }: { nodes: Node[]; edges: Edge[]; height: number; theme: Theme }) {
  return (
    <div className="overflow-hidden rounded-xl border border-neutral-200 bg-neutral-50/70 dark:border-neutral-800/80 dark:bg-neutral-950/60"
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
        <Background gap={16} color={edgePalette(theme).grid} />
        <Controls position="bottom-right" showInteractive={false} />
      </ReactFlow>
    </div>
  );
}

function nodeStyle(
  kind: "user" | "brain" | "expert" | "muted" | "done",
  theme: Theme,
): React.CSSProperties {
  const base: React.CSSProperties = {
    padding: 8, fontSize: 12, border: "1px solid", borderRadius: 12,
    width: 150, fontWeight: 500,
  };
  if (theme === "light") {
    const L = { ...base, color: "#0f172a", background: "rgba(255,255,255,0.92)", borderColor: "#cbd5e1" };
    if (kind === "user")
      return { ...L, background: "rgba(99,102,241,0.12)",  borderColor: "#6366f1", color: "#3730a3" };
    if (kind === "brain")
      return { ...L, background: "rgba(139,92,246,0.12)",  borderColor: "#7c3aed", color: "#5b21b6" };
    if (kind === "expert")
      return { ...L, background: "rgba(16,185,129,0.14)",  borderColor: "#059669", color: "#065f46" };
    if (kind === "done")
      return { ...L, background: "rgba(34,197,94,0.14)",   borderColor: "#16a34a", color: "#15803d" };
    return { ...L, background: "rgba(226,232,240,0.6)", borderColor: "#cbd5e1", color: "#94a3b8" };
  }
  const D = { ...base, color: "#e5e7eb", background: "rgba(20,22,32,0.85)", borderColor: "#3f3f4a" };
  if (kind === "user")
    return { ...D, background: "rgba(99,102,241,0.16)",  borderColor: "#6366f1", color: "#c7d2fe" };
  if (kind === "brain")
    return { ...D, background: "rgba(139,92,246,0.16)",  borderColor: "#7c3aed", color: "#ddd6fe" };
  if (kind === "expert")
    return { ...D, background: "rgba(16,185,129,0.16)",  borderColor: "#047857", color: "#a7f3d0" };
  if (kind === "done")
    return { ...D, background: "rgba(34,197,94,0.16)",   borderColor: "#16a34a", color: "#bbf7d0" };
  return { ...D, background: "rgba(38,38,51,0.4)", borderColor: "#3f3f4a", color: "#737280" };
}

// ---------- trace step chip ------------------------------------------------

function StepChip({ step }: { step: Step }) {
  const tone = stepTone(step);
  const colour =
    tone === "brain"
      ? "bg-violet-100 border-violet-300 text-violet-700 dark:bg-violet-500/15 dark:border-violet-400/30 dark:text-violet-200"
      : tone === "expert"
      ? "bg-emerald-100 border-emerald-300 text-emerald-700 dark:bg-emerald-500/15 dark:border-emerald-400/30 dark:text-emerald-200"
      : tone === "tool"
      ? "bg-sky-100 border-sky-300 text-sky-700 dark:bg-sky-500/15 dark:border-sky-400/30 dark:text-sky-200"
      : "bg-neutral-100 border-neutral-300 text-neutral-600 dark:bg-neutral-700/20 dark:border-neutral-500/30 dark:text-neutral-300";
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

// ---------- copy-to-clipboard ----------------------------------------------

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = useCallback(() => {
    if (!text) return;
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    });
  }, [text]);
  return (
    <button
      onClick={onCopy}
      className="rounded-md border border-neutral-300 bg-white/70 px-2 py-0.5 text-[11px] text-neutral-600 hover:border-neutral-400 hover:text-neutral-900 dark:border-neutral-700/70 dark:bg-neutral-900/60 dark:text-neutral-300 dark:hover:border-neutral-500 dark:hover:text-neutral-100"
      title="Copy response"
    >
      {copied ? "✓ copied" : "⧉ copy"}
    </button>
  );
}

// ---------- assistant turn -------------------------------------------------

function AssistantBubble({ turn, theme }: { turn: Extract<Turn, { role: "assistant" }>; theme: Theme }) {
  const [showTrace, setShowTrace] = useState(false);
  const r = turn.result;
  const ok = r.ok && r.response;
  const showRouterFlow = (turn.mode === "router" || (turn.mode === "auto" && r.auto_decision?.picked === "router")) && !!r.adapter_id;
  const showLangGraphFlow = turn.mode === "langgraph" || (turn.mode === "auto" && r.auto_decision?.picked === "agent");
  return (
    <div className="panel space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <span className="rounded-full border border-neutral-300 bg-neutral-100 px-2 py-0.5 text-neutral-600 dark:border-neutral-700 dark:bg-neutral-900/70 dark:text-neutral-300">
          mode: <b className="text-neutral-900 dark:text-neutral-100">{turn.mode}</b>
        </span>
        {r.auto_decision && (
          <span className="rounded-full border border-amber-300 bg-amber-100 px-2 py-0.5 text-amber-800 dark:border-amber-700/70 dark:bg-amber-900/30 dark:text-amber-200">
            auto → <b>{r.auto_decision.picked}</b>{" "}
            <span className="opacity-60">({r.auto_decision.reason})</span>
          </span>
        )}
        {r.adapter_id && (
          <span className="rounded-full border border-emerald-300 bg-emerald-100 px-2 py-0.5 text-emerald-800 dark:border-emerald-700/70 dark:bg-emerald-900/30 dark:text-emerald-200">
            adapter: <span className="font-mono">{r.adapter_id}</span>
          </span>
        )}
        {r.elapsed != null && (
          <span className="rounded-full border border-neutral-300 bg-neutral-100 px-2 py-0.5 text-neutral-500 dark:border-neutral-700 dark:bg-neutral-900/70 dark:text-neutral-400">
            {formatElapsed(r.elapsed)}
          </span>
        )}
        {ok && (
          <span className="ml-auto">
            <CopyButton text={r.response ?? ""} />
          </span>
        )}
      </div>

      {showRouterFlow && <RouterFlow result={r} theme={theme} />}
      {showLangGraphFlow && <LangGraphFlow result={r} theme={theme} />}

      <div className="text-sm">
        {ok ? (
          <div className="prose prose-sm max-w-none dark:prose-invert">
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeHighlight]}>
              {r.response ?? ""}
            </ReactMarkdown>
          </div>
        ) : (
          <div className="whitespace-pre-wrap text-rose-600 dark:text-rose-300">{r.error ?? "(no response)"}</div>
        )}
      </div>

      {r.steps && r.steps.length > 0 && (
        <div>
          <button
            onClick={() => setShowTrace((s) => !s)}
            className="text-[11px] text-neutral-500 underline-offset-4 hover:underline dark:text-neutral-400"
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
  const [hydrated, setHydrated] = useState(false);
  const [theme, setTheme] = useState<Theme>("dark");
  const [adapters, setAdapters] = useState<Adapter[]>([]);
  const [healthy, setHealthy] = useState<boolean | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Sync theme state with the class the no-flash script set on <html>.
  useEffect(() => {
    setTheme(document.documentElement.classList.contains("dark") ? "dark" : "light");
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((prev) => {
      const next: Theme = prev === "dark" ? "light" : "dark";
      const root = document.documentElement;
      if (next === "dark") root.classList.add("dark");
      else root.classList.remove("dark");
      try { window.localStorage.setItem(THEME_KEY, next); } catch {/* ignore */}
      return next;
    });
  }, []);

  // Hydrate persisted chat once on mount; guarded so we don't write back
  // an empty array before we've read.
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw) as Turn[];
        if (Array.isArray(parsed)) setChat(parsed);
      }
    } catch {/* ignore corrupt storage */}
    setHydrated(true);
  }, []);

  // Persist after every chat change (post-hydration to avoid wipe-on-load).
  useEffect(() => {
    if (!hydrated) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(chat));
    } catch {/* quota / private-mode — ignore */}
  }, [chat, hydrated]);

  useEffect(() => {
    Promise.all([api.health(), api.adapters()])
      .then(([h, a]) => { setHealthy(h.ok); setAdapters(a); setError(null); })
      .catch((e: unknown) => { setHealthy(false); setError(e instanceof Error ? e.message : String(e)); });
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [chat, running]);

  // Auto-grow the textarea up to ~10 lines, then scroll.
  const resizeTextarea = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 240) + "px";
  }, []);
  useEffect(resizeTextarea, [query, resizeTextarea]);

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
      textareaRef.current?.focus();
    }
  }, [query, mode, running]);

  // ChatGPT-style: Enter sends, Shift+Enter newline. `isComposing` guards
  // against submitting mid-IME composition (Japanese / Chinese / Korean).
  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void onSubmit();
    }
  };

  const onClear = useCallback(() => {
    if (chat.length === 0) return;
    if (!window.confirm("Clear all chat history?")) return;
    setChat([]);
  }, [chat.length]);

  const activeMode = MODES.find((m) => m.id === mode)!;

  const ghostBtn =
    "rounded-lg border border-neutral-300 bg-white/70 px-2.5 py-1 text-[11px] text-neutral-600 transition hover:border-neutral-400 hover:text-neutral-900 dark:border-neutral-700 dark:bg-neutral-900/60 dark:text-neutral-300 dark:hover:text-neutral-100";

  return (
    <main className="flex h-screen flex-col">
      {/* HEADER */}
      <header className="relative z-50 flex shrink-0 items-center justify-between border-b border-neutral-200 bg-white/70 px-4 py-2.5 backdrop-blur dark:border-neutral-800/80 dark:bg-neutral-950/40">
        <div className="flex items-center gap-3">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-gradient-to-br from-emerald-500 via-sky-500 to-violet-500 text-base">
            🧠
          </div>
          <div>
            <h1 className="text-sm font-semibold leading-none text-neutral-900 dark:text-neutral-100">Adaptive Minds</h1>
            <p className="text-[11px] text-neutral-500 dark:text-neutral-400">
              {adapters.length ? `${adapters.length} adapters` : "loading…"} · base = Qwen2.5-7B-Instruct ·
              FastAPI + vLLM
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={toggleTheme}
            className={ghostBtn}
            title="Toggle light / dark"
          >
            {theme === "dark" ? "☀️ Light" : "🌙 Dark"}
          </button>
          <button
            onClick={onClear}
            disabled={chat.length === 0}
            className="rounded-lg border border-neutral-300 bg-white/70 px-2.5 py-1 text-[11px] text-neutral-600 transition hover:border-rose-400 hover:bg-rose-50 hover:text-rose-700 disabled:cursor-not-allowed disabled:opacity-40 dark:border-neutral-700 dark:bg-neutral-900/60 dark:text-neutral-300 dark:hover:border-rose-700/60 dark:hover:bg-rose-950/30 dark:hover:text-rose-200"
            title="Clear chat history"
          >
            🗑 Clear
          </button>
          <span
            className={
              "rounded-full px-2 py-1 text-[11px] " +
              (healthy === true
                ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300"
                : healthy === false
                ? "bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300"
                : "bg-neutral-200 text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400")
            }
          >
            {healthy === true ? "● online" : healthy === false ? "● offline" : "● …"}
          </span>
        </div>
      </header>

      {/* MODE TABS */}
      <nav className="relative z-40 flex shrink-0 items-center gap-1 border-b border-neutral-200 bg-white/70 px-3 py-2 backdrop-blur dark:border-neutral-800/80 dark:bg-neutral-950/40">
        {MODES.map((m) => (
          <button
            key={m.id}
            onClick={() => setMode(m.id)}
            className={
              "group relative rounded-lg px-3 py-1.5 text-sm transition " +
              (mode === m.id
                ? "bg-emerald-100 text-emerald-800 ring-1 ring-emerald-400/60 dark:bg-emerald-700/30 dark:text-emerald-100 dark:ring-emerald-600/60"
                : "text-neutral-600 hover:bg-neutral-100 hover:text-neutral-900 dark:text-neutral-300 dark:hover:bg-neutral-800/60 dark:hover:text-neutral-100")
            }
          >
            <span className="mr-1">{m.icon}</span>
            {m.label}
          </button>
        ))}
        <div className="ml-3 hidden text-[11px] text-neutral-500 md:block">{activeMode.blurb}</div>
      </nav>

      {/* BODY */}
      <div className="grid min-h-0 flex-1 grid-cols-[260px_1fr] gap-3 overflow-hidden p-3">
        {/* SIDEBAR */}
        <aside className="panel flex min-h-0 flex-col overflow-hidden">
          <div className="shrink-0 border-b border-neutral-200 px-3 py-2 text-xs font-semibold text-neutral-700 dark:border-neutral-800/80 dark:text-neutral-300">
            Adapters <span className="text-neutral-400 dark:text-neutral-500">({adapters.length})</span>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto">
            {error && <div className="p-3 text-xs text-rose-600 dark:text-rose-300">{error}</div>}
            {adapters.map((a) => (
              <div
                key={a.id}
                className="border-b border-neutral-100 px-3 py-1.5 text-xs dark:border-neutral-900"
                title={a.description}
              >
                <div className="font-mono text-[11px] text-neutral-900 dark:text-neutral-100">{a.id}</div>
                <div className="truncate text-[10px] text-neutral-500">{a.description}</div>
              </div>
            ))}
          </div>
          <div className="shrink-0 border-t border-neutral-200 p-3 text-[10px] leading-relaxed text-neutral-500 dark:border-neutral-800/80">
            <div>↵ to send · Shift + ↵ for newline</div>
            <div className="mt-1">History persists locally.</div>
          </div>
        </aside>

        {/* CHAT */}
        <section className="panel flex min-h-0 flex-col overflow-hidden">
          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
            {chat.length === 0 && (
              <div className="mx-auto flex h-full max-w-xl flex-col items-center justify-center gap-4 text-center">
                <div className="grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-emerald-500 via-sky-500 to-violet-500 text-2xl shadow-lg shadow-emerald-500/10">
                  🧠
                </div>
                <div>
                  <h2 className="text-base font-semibold text-neutral-900 dark:text-neutral-100">
                    Adaptive Minds — {activeMode.label} mode
                  </h2>
                  <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">{activeMode.blurb}</p>
                </div>
                <div className="flex w-full flex-col gap-1.5">
                  {SAMPLES[mode].map((s) => (
                    <button
                      key={s}
                      onClick={() => {
                        setQuery(s);
                        textareaRef.current?.focus();
                      }}
                      className="rounded-lg border border-neutral-200 bg-white/60 px-3 py-2 text-left text-xs text-neutral-700 transition hover:border-emerald-400 hover:bg-emerald-50 dark:border-neutral-800 dark:bg-neutral-900/50 dark:text-neutral-200 dark:hover:border-emerald-700/60 dark:hover:bg-neutral-900"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <AnimatePresence initial={false}>
              {chat.map((t, i) =>
                t.role === "user" ? (
                  <motion.div
                    key={`u-${t.ts}-${i}`}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="ml-auto max-w-3xl whitespace-pre-wrap rounded-2xl border border-sky-300 bg-sky-100 px-4 py-2 text-sm text-sky-900 dark:border-sky-700/60 dark:bg-sky-950/30 dark:text-sky-100"
                  >
                    {t.content}
                  </motion.div>
                ) : (
                  <motion.div
                    key={`a-${t.ts}-${i}`}
                    initial={{ opacity: 0, y: 6 }}
                    animate={{ opacity: 1, y: 0 }}
                  >
                    <AssistantBubble turn={t} theme={theme} />
                  </motion.div>
                ),
              )}
            </AnimatePresence>
            {running && (
              <div className="flex items-center gap-2 text-xs text-neutral-500">
                <span className="inline-block h-2 w-2 animate-pulse rounded-full bg-emerald-500 dark:bg-emerald-400" />
                running ({mode})…
              </div>
            )}
            <div ref={endRef} />
          </div>

          <div className="shrink-0 border-t border-neutral-200 p-3 dark:border-neutral-800/80">
            <div className="flex items-end gap-2 rounded-2xl border border-neutral-300 bg-white p-2 focus-within:border-emerald-500/70 focus-within:ring-1 focus-within:ring-emerald-500/30 dark:border-neutral-700 dark:bg-neutral-900 dark:focus-within:border-emerald-600/70 dark:focus-within:ring-emerald-600/30">
              <textarea
                ref={textareaRef}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={onKeyDown}
                placeholder={`Message Adaptive Minds (${activeMode.label} mode)…`}
                rows={1}
                className="min-h-[28px] w-full resize-none bg-transparent px-1 text-sm leading-relaxed text-neutral-900 placeholder:text-neutral-400 focus:outline-none dark:text-neutral-100 dark:placeholder:text-neutral-500"
              />
              <button
                onClick={() => void onSubmit()}
                disabled={!query.trim() || running}
                className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-gradient-to-br from-emerald-600 to-sky-600 text-base text-white shadow-sm transition hover:from-emerald-500 hover:to-sky-500 disabled:cursor-not-allowed disabled:opacity-40"
                title="Send (Enter)"
              >
                {running ? (
                  <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                ) : (
                  <span>↑</span>
                )}
              </button>
            </div>
            <div className="mt-1.5 px-1 text-[10px] text-neutral-500">
              ↵ send · Shift + ↵ newline · responses can be inaccurate, verify important info.
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}
