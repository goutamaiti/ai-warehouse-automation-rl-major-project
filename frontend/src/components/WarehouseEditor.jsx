import { useEffect, useMemo, useRef, useState } from 'react';
import { CELL_COLORS, CELL_NAMES, CellType, validateGrid } from '../lib/grid';

const MIN_SIZE = 5;
const MAX_SIZE = 40; // mirrors simulation/warehouse.py::MAX_CUSTOM_SIZE

const PALETTE = [
  { type: CellType.EMPTY, label: 'Empty', hint: 'walkable aisle' },
  { type: CellType.WALL, label: 'Wall', hint: 'blocks movement' },
  { type: CellType.SHELF, label: 'Shelf', hint: 'blocks movement' },
  { type: CellType.STORAGE, label: 'Storage', hint: 'package pickup point' },
  { type: CellType.PACKING, label: 'Packing', hint: 'delivery station' },
  { type: CellType.CHARGING, label: 'Charging', hint: 'robot starts here' },
];

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
    for (let r = 0; r < height; r += 1) {
      for (let c = 0; c < width; c += 1) {
        ctx.fillStyle = CELL_COLORS[grid[r][c]] ?? CELL_COLORS[CellType.EMPTY];
        ctx.fillRect(c * CELL_PX, r * CELL_PX, CELL_PX, CELL_PX);
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.10)';
        ctx.strokeRect(c * CELL_PX + 0.5, r * CELL_PX + 0.5, CELL_PX - 1, CELL_PX - 1);
      }
    }
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
          {PALETTE.map((item) => (
            <button
              key={item.type}
              type="button"
              className={`palette-button ${tool === item.type ? 'selected' : ''}`}
              style={{ '--swatch': CELL_COLORS[item.type] }}
              onClick={() => setTool(item.type)}
              title={item.hint}
            >
              <span className="swatch" />
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

      <p className={`validation-status ${validation.valid ? 'ok' : 'fail'}`}>
        {validation.valid
          ? 'Layout is valid: every storage point, packing station and charging station can reach each other.'
          : validation.error}
      </p>
      <p className="muted small">
        {Object.entries(CELL_NAMES).map(([value, name]) => `${name}`).join(' · ')} — click a
        palette tool, then click or drag on the grid to paint. The robot always starts on the
        charging station nearest the top-left corner.
      </p>
    </div>
  );
}

export { blankGrid, MIN_SIZE, MAX_SIZE };
