import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Adaptive Minds",
  description:
    "LoRA adapters as callable tools for one base model — router + ReAct agent over vLLM.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f7f8fa" },
    { media: "(prefers-color-scheme: dark)", color: "#07070b" },
  ],
  width: "device-width",
  initialScale: 1,
};

// Set the theme class before first paint to avoid a flash. Default is dark;
// only an explicit `am-theme=light` in localStorage switches to light.
const THEME_INIT = `
(function(){try{
  if(localStorage.getItem('am-theme')==='light'){
    document.documentElement.classList.remove('dark');
  }else{
    document.documentElement.classList.add('dark');
  }
}catch(e){document.documentElement.classList.add('dark');}})();
`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_INIT }} />
      </head>
      <body data-testid="app-ready" className="antialiased">
        {children}
      </body>
    </html>
  );
}
