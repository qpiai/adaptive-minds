import React from "react";
import {
  AbsoluteFill,
  Easing,
  interpolate,
  OffthreadVideo,
  Sequence,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";
import { loadFont } from "@remotion/google-fonts/Inter";

const { fontFamily } = loadFont();

export const FPS = 30;
export const INTRO_FRAMES = 120; // 4.0s
export const OUTRO_FRAMES = 180; // 6.0s
export const PLAYBACK = 1.35; // speed-up applied to the captured footage

const FONT = `${fontFamily}, -apple-system, "Segoe UI", Roboto, sans-serif`;
const ACCENT = "linear-gradient(110deg,#34d399 0%,#38bdf8 50%,#a78bfa 100%)";

// ---------- shared bits -----------------------------------------------------

const Stage: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AbsoluteFill
    style={{
      fontFamily: FONT,
      background:
        "radial-gradient(1200px 700px at 50% 18%, #131a2b 0%, #0a0e16 55%, #06080d 100%)",
      color: "#e7ecf3",
      overflow: "hidden",
    }}
  >
    {/* ambient colour glows */}
    <AbsoluteFill
      style={{
        background:
          "radial-gradient(520px 360px at 16% 88%, rgba(52,211,153,0.16), transparent 70%), radial-gradient(560px 380px at 86% 12%, rgba(167,139,250,0.16), transparent 70%)",
      }}
    />
    {children}
  </AbsoluteFill>
);

const GradientText: React.FC<{
  children: React.ReactNode;
  size: number;
  weight?: number;
  ls?: number;
}> = ({ children, size, weight = 800, ls = -1 }) => (
  <span
    style={{
      fontSize: size,
      fontWeight: weight,
      letterSpacing: ls,
      lineHeight: 1.05,
      backgroundImage: ACCENT,
      WebkitBackgroundClip: "text",
      backgroundClip: "text",
      color: "transparent",
    }}
  >
    {children}
  </span>
);

// ---------- intro -----------------------------------------------------------

const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const pop = spring({ frame, fps, config: { damping: 11, mass: 0.7 } });
  const logoScale = interpolate(pop, [0, 1], [0.4, 1]);
  const logoRot = interpolate(pop, [0, 1], [-18, 0]);

  const rise = (start: number) => ({
    opacity: interpolate(frame, [start, start + 18], [0, 1], { extrapolateRight: "clamp" }),
    transform: `translateY(${interpolate(frame, [start, start + 18], [26, 0], {
      extrapolateRight: "clamp",
      easing: Easing.out(Easing.cubic),
    })}px)`,
  });

  const fadeOut = interpolate(frame, [INTRO_FRAMES - 16, INTRO_FRAMES], [1, 0], {
    extrapolateLeft: "clamp",
  });

  const modes = ["🎯 Router", "🤖 Agent", "🪄 Auto", "🕸️ LangGraph"];

  return (
    <Stage>
      <AbsoluteFill
        style={{
          opacity: fadeOut,
          justifyContent: "center",
          alignItems: "center",
          gap: 34,
          padding: 80,
          textAlign: "center",
        }}
      >
        <div
          style={{
            width: 168,
            height: 168,
            borderRadius: 40,
            display: "grid",
            placeItems: "center",
            fontSize: 92,
            background: ACCENT,
            boxShadow: "0 30px 90px rgba(56,189,248,0.35)",
            transform: `scale(${logoScale}) rotate(${logoRot}deg)`,
          }}
        >
          🧠
        </div>

        <div style={rise(12)}>
          <GradientText size={120}>Adaptive Minds</GradientText>
        </div>

        <div style={{ ...rise(26), fontSize: 40, color: "#aeb8c8", fontWeight: 500 }}>
          LoRA adapters as <b style={{ color: "#e7ecf3" }}>callable tools</b> for agent orchestration
        </div>

        <div style={{ ...rise(40), display: "flex", gap: 16, marginTop: 8 }}>
          {modes.map((m, i) => {
            const s = 46 + i * 7;
            const o = interpolate(frame, [s, s + 14], [0, 1], { extrapolateRight: "clamp" });
            const y = interpolate(frame, [s, s + 14], [16, 0], {
              extrapolateRight: "clamp",
              easing: Easing.out(Easing.cubic),
            });
            return (
              <span
                key={m}
                style={{
                  opacity: o,
                  transform: `translateY(${y}px)`,
                  fontSize: 30,
                  fontWeight: 600,
                  padding: "12px 24px",
                  borderRadius: 999,
                  color: "#dbe4f0",
                  border: "1px solid rgba(120,140,170,0.35)",
                  background: "rgba(20,26,40,0.6)",
                }}
              >
                {m}
              </span>
            );
          })}
        </div>
      </AbsoluteFill>
    </Stage>
  );
};

// ---------- app footage -----------------------------------------------------

const AppClip: React.FC<{ durationInFrames: number }> = ({ durationInFrames }) => {
  const frame = useCurrentFrame();
  const fadeIn = interpolate(frame, [0, 14], [0, 1], { extrapolateRight: "clamp" });
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 14, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp" },
  );
  return (
    <AbsoluteFill style={{ background: "#06080d", opacity: Math.min(fadeIn, fadeOut) }}>
      <OffthreadVideo
        src={staticFile("app.mp4")}
        playbackRate={PLAYBACK}
        style={{ width: "100%", height: "100%", objectFit: "cover" }}
      />
    </AbsoluteFill>
  );
};

// ---------- outro -----------------------------------------------------------

const Stat: React.FC<{ value: string; label: string; delay: number }> = ({
  value,
  label,
  delay,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame: frame - delay, fps, config: { damping: 13 } });
  return (
    <div
      style={{
        opacity: s,
        transform: `translateY(${interpolate(s, [0, 1], [24, 0])}px)`,
        minWidth: 340,
        padding: "30px 36px",
        borderRadius: 24,
        background: "rgba(18,24,38,0.7)",
        border: "1px solid rgba(120,140,170,0.25)",
        textAlign: "center",
      }}
    >
      <div style={{ marginBottom: 8 }}>
        <GradientText size={68}>{value}</GradientText>
      </div>
      <div style={{ fontSize: 26, color: "#aeb8c8", fontWeight: 500 }}>{label}</div>
    </div>
  );
};

const Outro: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const headO = interpolate(frame, [4, 24], [0, 1], { extrapolateRight: "clamp" });
  const headY = interpolate(frame, [4, 24], [24, 0], {
    extrapolateRight: "clamp",
    easing: Easing.out(Easing.cubic),
  });
  const cta = spring({ frame: frame - 80, fps, config: { damping: 12 } });
  const linksO = interpolate(frame, [96, 116], [0, 1], { extrapolateRight: "clamp" });
  const fadeIn = interpolate(frame, [0, 12], [0, 1], { extrapolateRight: "clamp" });

  return (
    <Stage>
      <AbsoluteFill
        style={{
          opacity: fadeIn,
          justifyContent: "center",
          alignItems: "center",
          gap: 46,
          padding: 90,
          textAlign: "center",
        }}
      >
        <div style={{ opacity: headO, transform: `translateY(${headY}px)` }}>
          <div style={{ fontSize: 84, fontWeight: 800, letterSpacing: -1 }}>
            One base model. <GradientText size={84}>Many experts.</GradientText>
          </div>
        </div>

        <div style={{ display: "flex", gap: 26 }}>
          <Stat value="98.3%" label="routing accuracy" delay={26} />
          <Stat value="30" label="LoRA specialists" delay={38} />
          <Stat value="4" label="orchestration modes" delay={50} />
        </div>

        <div
          style={{
            transform: `scale(${interpolate(cta, [0, 1], [0.8, 1])})`,
            opacity: cta,
            fontSize: 36,
            fontWeight: 700,
            color: "#06281d",
            padding: "20px 44px",
            borderRadius: 999,
            background: ACCENT,
            boxShadow: "0 18px 60px rgba(52,211,153,0.4)",
          }}
        >
          ★ Star us on GitHub
        </div>

        <div style={{ opacity: linksO, fontSize: 30, color: "#9fb0c6", fontWeight: 500 }}>
          github.com/qpiai/adaptive-minds
          <span style={{ margin: "0 18px", color: "#3a4760" }}>·</span>
          arXiv 2510.15416
        </div>
      </AbsoluteFill>
    </Stage>
  );
};

// ---------- composition root ------------------------------------------------

export const Demo: React.FC<{ appFrames: number }> = ({ appFrames }) => {
  return (
    <AbsoluteFill style={{ background: "#06080d" }}>
      <Sequence durationInFrames={INTRO_FRAMES}>
        <Intro />
      </Sequence>
      <Sequence from={INTRO_FRAMES} durationInFrames={appFrames}>
        <AppClip durationInFrames={appFrames} />
      </Sequence>
      <Sequence from={INTRO_FRAMES + appFrames} durationInFrames={OUTRO_FRAMES}>
        <Outro />
      </Sequence>
    </AbsoluteFill>
  );
};
