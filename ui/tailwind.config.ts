import type { Config } from "tailwindcss";

const config: Config = {
  // Class strategy: <html> carries `dark` by default (set in layout.tsx), so
  // the app is dark out of the box. The header toggle removes the class → light.
  darkMode: "class",
  content: [
    "./src/app/**/*.{ts,tsx}",
    "./src/components/**/*.{ts,tsx}",
    "./src/lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        bg: { 0: "#07070b", 1: "#0d0d13", 2: "#14141c" },
        edge: { 0: "#262633", 1: "#33334a" },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
