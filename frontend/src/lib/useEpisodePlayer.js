import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * Playback state (current frame, play/pause, speed) for a replay object
 * shaped like `EpisodeRecorder.to_dict()` - shared by the Dashboard's
 * recorded-episode viewer and the Editor's live sandbox runs so both scrub,
 * play and total reward the same way.
 */
export function useEpisodePlayer(episode) {
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(8);
  const timer = useRef(null);

  useEffect(() => {
    setFrameIndex(0);
    setPlaying(false);
  }, [episode]);

  const totalFrames = episode?.frames?.length ?? 0;

  useEffect(() => {
    if (!playing) return undefined;
    if (totalFrames === 0 || frameIndex >= totalFrames - 1) {
      setPlaying(false);
      return undefined;
    }
    timer.current = window.setTimeout(() => setFrameIndex((current) => current + 1), 1000 / speed);
    return () => window.clearTimeout(timer.current);
  }, [playing, speed, frameIndex, totalFrames]);

  const frame = episode?.frames?.[frameIndex] ?? null;

  const trail = useMemo(() => {
    if (!episode?.frames) return [];
    return episode.frames.slice(0, frameIndex + 1).map((item) => item.robot.position);
  }, [episode, frameIndex]);

  /** Sum of every frame's reward from the start of the episode up to (and including) the current frame. */
  const cumulativeReward = useMemo(() => {
    if (!episode?.frames) return 0;
    let total = 0;
    for (let i = 1; i <= frameIndex && i < episode.frames.length; i += 1) {
      total += episode.frames[i].reward ?? 0;
    }
    return total;
  }, [episode, frameIndex]);

  function togglePlay() {
    setPlaying((current) => {
      if (!current && frameIndex >= totalFrames - 1) setFrameIndex(0);
      return !current;
    });
  }

  function seek(value) {
    setPlaying(false);
    setFrameIndex(value);
  }

  return {
    frameIndex,
    playing,
    speed,
    setSpeed,
    totalFrames,
    frame,
    trail,
    cumulativeReward,
    togglePlay,
    seek,
  };
}
