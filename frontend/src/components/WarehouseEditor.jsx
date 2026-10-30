import { useEffect, useMemo, useRef, useState } from 'react';
import { CellType, MAX_GRID_SIZE, MIN_GRID_SIZE, validateGrid } from '../lib/grid';
import { CELL_LEGEND, drawFloor } from '../lib/cellArt';
import CellSwatch from './CellSwatch';

// These were a third copy of simulation/warehouse.py's MIN_/MAX_CUSTOM_SIZE.
// They now come from lib/grid, which is the module that has to agree with the
// Python limits anyway because it is what validates against them.
const MIN_SIZE = MIN_GRID_SIZE;
const MAX_SIZE = MAX_GRID_SIZE;

function blankGrid(height, width) {
  const grid = Array.from({ length: height }, (_, r) =>
    Array.from({ length: width }, (_, c) =>
      r === 0 || r === height - 1 || c === 0 || c === width - 1 ? CellType.WALL : CellType.EMPTY
    )
  );
  return grid;
}

function resizeGrid(grid, height, width) {
  const next = blankGrid(height, width);
  for (let r = 0; r < Math.min(grid.length, height); r += 1) {
    for (let c = 0; c < Math.min(grid[0].length, width); c += 1) {
      next[r][c] = grid[r][c];
    }
  }
  return next;
}

const CELL_PX = 26;

/**
 * Click-and-drag grid painter. Fully controlled from the outside: it owns no
 * simulation state, only the grid array, and reports every change through
 * `onChange`.
 */
export default function WarehouseEditor({ grid, onChange }) {
  const [tool, setTool] = useState(CellType.WALL);
  const [dims, setDims] = useState({ height: grid.length, width: grid[0].length });
  const painting = useRef(false);
  const canvasRef = useRef(null);
  const wrapperRef = useRef(null);

  const validation = useMemo(() => validateGrid(grid), [grid]);

  /** What has actually been painted. A layout can be invalid simply because
   *  one of the three required kinds of cell is missing, and the count says so
   *  before the reader has to read an error message. */
  const counts = useMemo(() => {
    const tally = { storage: 0, packing: 0, charging: 0 };
    grid.forEach((row) =>
      row.forEach((value) => {
        if (value === CellType.STORAGE) tally.storage += 1;
        else if (value === CellType.PACKING) tally.packing += 1;
        else if (value === CellType.CHARGING) tally.charging += 1;
      })
    );
    return tally;
  }, [grid]);

  useEffect(() => {
    setDims({ height: grid.length, width: grid[0].length });
  }, [grid]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const height = grid.length;
    const width = grid[0].length;
    const ratio = window.devicePixelRatio || 1;
    const pxWidth = width * CELL_PX;
    const pxHeight = height * CELL_PX;
    canvas.width = pxWidth * ratio;
    canvas.height = pxHeight * ratio;
    canvas.style.width = `${pxWidth}px`;
    canvas.style.height = `${pxHeight}px`;

    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, pxWidth, pxHeight);
    // The same renderer the replay canvas uses, so a warehouse looks the same
    // while you are drawing it as it does when a robot runs on it.
    drawFloor(ctx, { grid, width, height }, CELL_PX);
  }, [grid]);

  function cellAt(event) {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const col = Math.floor(x / CELL_PX);
    const row = Math.floor(y / CELL_PX);
    if (row < 0 || row >= grid.length || col < 0 || col >= grid[0].length) return null;
    return [row, col];
  }

  function paint(event) {
    const cell = cellAt(event);
    if (!cell) return;
    const [row, col] = cell;
    if (grid[row][col] === tool) return;
    const next = grid.map((r) => [...r]);
    next[row][col] = tool;
    onChange(next);
  }

  function handlePointerDown(event) {
    painting.current = true;
    paint(event);
  }
  function handlePointerMove(event) {
    if (painting.current) paint(event);
  }
  function stopPainting() {
    painting.current = false;
  }

  function applyResize() {
    const height = Math.min(MAX_SIZE, Math.max(MIN_SIZE, Math.round(dims.height)));
    const width = Math.min(MAX_SIZE, Math.max(MIN_SIZE, Math.round(dims.width)));
    onChange(resizeGrid(grid, height, width));
  }

  function clear() {
    onChange(blankGrid(grid.length, grid[0].length));
  }

  return (
    <div className="warehouse-editor">
      <div className="editor-toolbar">
        <div className="palette">
          {CELL_LEGEND.map((item) => (
            <button
              key={item.type}
              type="button"
              className={`palette-button ${tool === item.type ? 'selected' : ''}`}
              onClick={() => setTool(item.type)}
              title={item.hint}
            >
              <CellSwatch type={item.type} />
              {item.label}
            </button>
          ))}
        </div>

        <div className="editor-dims">
          <label>
            Height
            <input
              type="number"
              min={MIN_SIZE}
              max={MAX_SIZE}
              value={dims.height}
              onChange={(e) => setDims({ ...dims, height: Number(e.target.value) })}
            />
          </label>
          <label>
            Width
            <input
              type="number"
              min={MIN_SIZE}
              max={MAX_SIZE}
              value={dims.width}
              onChange={(e) => setDims({ ...dims, width: Number(e.target.value) })}
            />
          </label>
          <button type="button" onClick={applyResize}>Resize</button>
          <button type="button" onClick={clear}>Clear</button>
        </div>
      </div>

      <div className="canvas-wrapper" ref={wrapperRef}>
        <canvas
          ref={canvasRef}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={stopPainting}
          onPointerLeave={stopPainting}
        />
      </div>

      {/* The verdict sits directly under the grid, not under the run controls:
          it is feedback on the thing you are currently touching, and it is the
          only reason the Run button is ever disabled. */}
      <p className={`validation-status ${validation.valid ? 'ok' : 'fail'}`}>
        <b>{validation.valid ? 'Valid layout' : 'Not runnable yet'}</b>
        {validation.valid
          ? ' — every pick point, packing station and charger can reach each other.'
          : ` — ${validation.error}`}
      </p>
      <p className="muted small">
        Pick a brush, then click or drag on the grid. {counts.storage} pick{' '}
        {counts.storage === 1 ? 'point' : 'points'}, {counts.packing} packing, {counts.charging}{' '}
        charging. The robot always starts on the charger nearest the top-left corner.
      </p>
    </div>
  );
}

export { blankGrid, MIN_SIZE, MAX_SIZE };
