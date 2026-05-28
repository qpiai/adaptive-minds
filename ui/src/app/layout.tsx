import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Adaptive Minds",
  description:
    "LoRA adapters as callable tools for one base model — router + ReAct agent over vLLM.",
};

export const viewport: Viewport = {
  themeColor: "#07070b",
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body data-testid="app-ready" className="antialiased">
        {children}
      </body>
    </html>
  );
}
