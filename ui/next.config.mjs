/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone output keeps the runtime image small — only .next/standalone/
  // is copied into the runner stage in docker/Dockerfile.ui.
  output: "standalone",
  reactStrictMode: true,
  // Default to relative URLs (`/api/am/*`) so the bundle is portable across
  // hosts. The rewrite below proxies to the FastAPI server over the docker
  // network. Override NEXT_PUBLIC_AM_API_BASE only to bypass the proxy.
  // Use `||` not `??` so an empty-string env (which Docker ARG defaults
  // to) also falls back to the default — `??` only catches nullish.
  env: {
    NEXT_PUBLIC_AM_API_BASE: process.env.NEXT_PUBLIC_AM_API_BASE || "/api/am",
  },
  async rewrites() {
    const upstream =
      process.env.AM_UPSTREAM_BASE ?? "http://server:8765";
    return [
      { source: "/api/am/:path*", destination: `${upstream}/:path*` },
    ];
  },
};

export default nextConfig;
