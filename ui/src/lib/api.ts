// Typed fetchers for the Adaptive Minds FastAPI server.
// Default: relative `/api/am/*` proxied by Next.js to `server:8765` over
// the docker network (see next.config.mjs rewrites). Override
// NEXT_PUBLIC_AM_API_BASE to point at FastAPI directly.

export const API_BASE =
  process.env.NEXT_PUBLIC_AM_API_BASE ?? "/api/am";

export type Adapter = {
  id: string;
  name: string;
  description: string;
  keywords: string[];
  hf_id: string;
};

export type Health = {
  ok: boolean;
  vllm_base: string;
  n_adapters: number;
  catalog: string | null;
  tools: string[];
};

export type Step = {
  label?: string;
  kind?: string;
  name?: string;
  adapter?: string;
  decision?: string;
  observation?: string;
  argument?: string;
  sub_query?: string;
  raw_response?: string;
  elapsed?: number;
  [k: string]: unknown;
};

export type AutoDecision = { picked: "agent" | "router"; reason: string };

export type ChatResult = {
  ok: boolean;
  mode: string;
  response?: string;
  adapter_id?: string;
  elapsed?: number;
  steps?: Step[];
  error?: string;
  request_mode?: ChatMode;
  auto_decision?: AutoDecision;
};

export type ChatMode = "router" | "agent" | "auto" | "langgraph";

export const MODES: { id: ChatMode; label: string; icon: string; blurb: string }[] = [
  { id: "router",    label: "Router",    icon: "🎯", blurb: "One LLM call picks an adapter; that adapter answers. Best for single-domain queries." },
  { id: "agent",     label: "Agent",     icon: "🤖", blurb: "ReAct loop: brain emits CALL / FINAL; runtime executes tools + LoRA experts." },
  { id: "auto",      label: "Auto",      icon: "🪄", blurb: "Heuristic dispatcher: short single-domain queries → Router; compound or multi-step → Agent." },
  { id: "langgraph", label: "LangGraph", icon: "🕸️", blurb: "Same ReAct loop as Agent, expressed as a LangGraph StateGraph (plan → dispatch → synthesise)." },
];

async function jget<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  if (!r.ok) {
    let body = "";
    try { body = await r.text(); } catch {/**/}
    throw new Error(`${r.status} ${r.statusText}: ${body.slice(0, 240)}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  health: () => jget<Health>("/health"),
  adapters: () => jget<Adapter[]>("/adapters"),
  chat: (
    query: string,
    mode: ChatMode,
    opts?: Partial<{ temperature: number; max_tokens: number }>,
  ) =>
    jget<ChatResult>("/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        query,
        mode,
        temperature: opts?.temperature ?? 0.3,
        max_tokens:
          opts?.max_tokens ??
          (mode === "router" ? 512 : 1024),
      }),
    }),
};
