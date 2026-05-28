/** @type {import('next').NextConfig} */
const nextConfig = {
  // Standalone output keeps the runtime image small — only .next/standalone/
  // is copied into the runner stage in docker/Dockerfile.ui.
  output: "standalone",
  reactStrictMode: true,
  // Browser → FastAPI is direct (CORS open on the server). The Next.js
  // server-side environment doesn't need the API base.
  env: {
    NEXT_PUBLIC_AM_API_BASE:
      process.env.NEXT_PUBLIC_AM_API_BASE ?? "http://localhost:8765",
  },
};

export default nextConfig;
