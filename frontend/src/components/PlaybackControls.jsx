/** Transport controls for stepping through a recorded episode. */
export default function PlaybackControls({
  playing,
  onTogglePlay,
  frameIndex,
  totalFrames,
  onSeek,
  speed,
  onSpeedChange,
}) {
  const lastFrame = Math.max(totalFrames - 1, 0);
  const disabled = totalFrames === 0;

  return (
    <div className="playback">
      <button type="button" onClick={() => onSeek(0)} disabled={disabled} title="Back to start">
        ⏮
      </button>
      <button
        type="button"
        onClick={() => onSeek(Math.max(0, frameIndex - 1))}
        disabled={disabled}
        title="Previous step"
      >
        ◀
      </button>
      <button type="button" className="primary" onClick={onTogglePlay} disabled={disabled}>
        {playing ? '⏸ Pause' : '▶ Play'}
      </button>
      <button
        type="button"
        onClick={() => onSeek(Math.min(lastFrame, frameIndex + 1))}
        disabled={disabled}
        title="Next step"
      >
        ▶
      </button>

      <input
        type="range"
        min="0"
        max={lastFrame}
        value={frameIndex}
        onChange={(event) => onSeek(Number(event.target.value))}
        disabled={disabled}
        aria-label="Timeline"
      />
      <span className="frame-counter">
        {frameIndex} / {lastFrame}
      </span>

      <label className="speed">
        Speed
        <select value={speed} onChange={(event) => onSpeedChange(Number(event.target.value))}>
          <option value={4}>0.5x</option>
          <option value={8}>1x</option>
          <option value={16}>2x</option>
          <option value={32}>4x</option>
        </select>
      </label>
    </div>
  );
}
