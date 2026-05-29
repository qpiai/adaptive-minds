import React from "react";
import { Composition, staticFile } from "remotion";
import { getVideoMetadata } from "@remotion/media-utils";
import { Demo, FPS, INTRO_FRAMES, OUTRO_FRAMES, PLAYBACK } from "./Demo";

// The app footage lives in public/app.mp4 (written by scripts/capture_demo.py).
// We size the whole composition dynamically: intro + (sped-up footage) + outro.
export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="Demo"
      component={Demo}
      fps={FPS}
      width={1920}
      height={1080}
      durationInFrames={1800}
      calculateMetadata={async () => {
        const meta = await getVideoMetadata(staticFile("app.mp4"));
        const appFrames = Math.ceil((meta.durationInSeconds / PLAYBACK) * FPS);
        return {
          durationInFrames: INTRO_FRAMES + appFrames + OUTRO_FRAMES,
          props: { appFrames },
        };
      }}
      defaultProps={{ appFrames: 1300 }}
    />
  );
};
