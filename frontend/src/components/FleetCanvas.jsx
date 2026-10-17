import { useEffect, useRef } from 'react';

import { CellType as CELL } from '../lib/grid';

/**
 * Canvas view of one live fleet frame.
 *
 * Separate from WarehouseCanvas rather than an extension of it: that component
 * draws a single recorded robot and is what every committed replay renders
 * through. Widening it for N robots, zones and queued requests would put the
 * measured view and the live view on one code path, where a change for the
 * demo could quietly alter how a recorded result is drawn.
 *
 * Purely presentational - it simulates nothing and decides nothing.
 */

//: Light-ground palette. The cell meanings are the same ones the simulation
//: uses; only the values are retuned for a paper background.
const INK = {
  paper: '#ffffff',
  grid: 'rgba(28, 32, 36, 0.07)',
  rule: 'rgba(28, 32, 36, 0.22)',
  ruler: '#8b9299',
  wall: '#1c2024',
  shelf: '#e8eaec',
  shelfSide: '#c3c8cd',
  arrow: 'rgba(28, 32, 36, 0.26)',
  storage: '#4a5057',
  packing: '#8b9299',
  charging: '#c3c8cd',
  robot: '#1c2024',
  robotCarrying: '#4a5057',
  robotDead: '#c3c8cd',
  obstacle: '#8b9299',
  zoneA: 'rgba(28, 32, 36, 0.025)',
  zoneB: 'rgba(28, 32, 36, 0.05)',
  pending: '#4a5057',
};

const MONO = "ui-monospace, 'SF Mono', Menlo, monospace";

const CELL_FILL = {
  [CELL.EMPTY]: null,
  [CELL.WALL]: INK.wall,
  [CELL.SHELF]: INK.shelf,
  [CELL.STORAGE]: INK.storage,
  [CELL.PACKING]: INK.packing,
  [CELL.CHARGING]: INK.charging,
};

const MAX_CELL = 26;
//: Below this the ruler labels and robot ids stop being readable.
const MIN_LABEL_CELL = 15;

//: Width of the ruler gutter. The card asks for a row and a column, so the map
//: has to be readable as a coordinate grid - without this you have to count
//: cells by eye, which is exactly the thing that made the card unusable.
const GUTTER = 20;

export default function FleetCanvas({ layout, frame, highlight }) {
  const canvasRef = useRef(null);
  const wrapperRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!canvas || !wrapper || !layout) return undefined;

    const draw = () => {
      const available = wrapper.clientWidth || 640;
      const cell = Math.max(
        6,
        Math.min(MAX_CELL, Math.floor((available - GUTTER) / layout.width))
      );
      const width = cell * layout.width + GUTTER;
      const height = cell * layout.height + GUTTER;
      const ratio = window.devicePixelRatio || 1;

      canvas.width = width * ratio;
      canvas.height = height * ratio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.fillStyle = INK.paper;
      ctx.fillRect(0, 0, width, height);

      // Rulers first, in page coordinates, then shift the origin so every
      // drawing call below can keep using plain cell indices.
      const step = cell >= 15 ? 1 : cell >= 10 ? 2 : 5;
      ctx.fillStyle = INK.ruler;
      ctx.font = `500 ${Math.min(11, Math.max(8, Math.round(cell * 0.46)))}px ${MONO}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let col = 0; col < layout.width; col += step) {
        ctx.fillText(String(col), GUTTER + col * cell + cell / 2, GUTTER / 2);
      }
      ctx.textAlign = 'right';
      for (let row = 0; row < layout.height; row += step) {
        ctx.fillText(String(row), GUTTER - 5, GUTTER + row * cell + cell / 2);
      }
      ctx.textAlign = 'center';
      ctx.translate(GUTTER, GUTTER);

      const zones = layout.zones ?? [];
      // 1. zone washes, so the sections of the floor are visible at a glance
      zones.forEach((zone, index) => {
        ctx.fillStyle = index % 2 === 0 ? INK.zoneA : INK.zoneB;
        (zone.cells ?? []).forEach(([row, col]) => {
          ctx.fillRect(col * cell, row * cell, cell, cell);
        });
      });

      // 2. the static map. Shelving is drawn with a lit top face and a darker
      //    side so a dense pod field reads as physical racking rather than a
      //    block of colour - without it the realistic floor looks flat.
      const lift = Math.max(2, Math.round(cell * 0.22));
      for (let row = 0; row < layout.height; row += 1) {
        for (let col = 0; col < layout.width; col += 1) {
          const type = layout.grid[row][col];
          const fill = CELL_FILL[type];
          if (!fill) continue;
          const x = col * cell;
          const y = row * cell;
          if (type === CELL.SHELF) {
            // Racking: a lit top face and a darker exposed side, so a dense pod
            // field reads as physical structure.
            const exposed =
              row + 1 >= layout.height || layout.grid[row + 1][col] !== CELL.SHELF;
            if (exposed) {
              ctx.fillStyle = INK.shelfSide;
              ctx.fillRect(x, y + cell - lift, cell, lift);
            }
            ctx.fillStyle = fill;
            ctx.fillRect(x, y, cell, cell - (exposed ? lift : 0));
          } else if (type === CELL.WALL) {
            ctx.fillStyle = fill;
            ctx.fillRect(x, y, cell, cell);
          } else {
            // Storage, packing and charging are places ON the floor, not
            // blockages: a robot stands on them. Filling the whole cell made
            // the realistic floor's 306 pick points read as a dark
            // checkerboard sealing the aisles, with the actual racking looking
            // like empty space. They are drawn as markers instead - filled,
            // outlined and dashed - which also keeps them apart without a
            // second colour.
            const inset = cell * 0.22;
            const size = cell - inset * 2;
            if (type === CELL.STORAGE) {
              ctx.fillStyle = fill;
              ctx.fillRect(x + inset, y + inset, size, size);
            } else {
              ctx.strokeStyle = fill;
              ctx.lineWidth = Math.max(1, cell * 0.08);
              if (type === CELL.CHARGING) ctx.setLineDash([cell * 0.14, cell * 0.1]);
              ctx.strokeRect(x + inset, y + inset, size, size);
              ctx.setLineDash([]);
            }
          }
        }
      }

      // 2b. one-way lane arrows, faint. This is the floor's actual traffic
      //     rule, and seeing it is how the congestion difference is explained.
      if (layout.directions) {
        ctx.strokeStyle = INK.arrow;
        ctx.lineWidth = 1;
        const every = cell >= 14 ? 2 : 3;
        for (let row = 0; row < layout.height; row += every) {
          for (let col = 0; col < layout.width; col += every) {
            const code = layout.directions[row][col];
            if (!code) continue;
            const x = col * cell + cell / 2;
            const y = row * cell + cell / 2;
            const a = Math.max(2, cell * 0.16);
            ctx.beginPath();
            if (code === 1) { ctx.moveTo(x - a, y - a); ctx.lineTo(x + a, y); ctx.lineTo(x - a, y + a); }
            if (code === 2) { ctx.moveTo(x + a, y - a); ctx.lineTo(x - a, y); ctx.lineTo(x + a, y + a); }
            if (code === 3) { ctx.moveTo(x - a, y - a); ctx.lineTo(x, y + a); ctx.lineTo(x + a, y - a); }
            if (code === 4) { ctx.moveTo(x - a, y + a); ctx.lineTo(x, y - a); ctx.lineTo(x + a, y + a); }
            ctx.stroke();
          }
        }
      }

      // 3. a faint grid, because this is a grid world and hiding that helps nobody
      ctx.strokeStyle = INK.grid;
      ctx.lineWidth = 1;
      for (let col = 0; col <= layout.width; col += 1) {
        ctx.beginPath();
        ctx.moveTo(col * cell + 0.5, 0);
        ctx.lineTo(col * cell + 0.5, height);
        ctx.stroke();
      }
      for (let row = 0; row <= layout.height; row += 1) {
        ctx.beginPath();
        ctx.moveTo(0, row * cell + 0.5);
        ctx.lineTo(width, row * cell + 0.5);
        ctx.stroke();
      }

      // 4. waiting cells, marked faintly so an empty-looking floor is explicable
      ctx.strokeStyle = INK.rule;
      zones.forEach((zone) => {
        (zone.waiting_points ?? []).forEach(([row, col]) => {
          ctx.strokeRect(col * cell + cell * 0.3, row * cell + cell * 0.3, cell * 0.4, cell * 0.4);
        });
      });

      if (!frame) return;

      // 5. queued requests still waiting for a robot
      (frame.requests ?? [])
        .filter((request) => request.status === 'waiting')
        .forEach((request) => {
          const [row, col] = request.pickup;
          ctx.fillStyle = INK.pending;
          ctx.beginPath();
          ctx.arc(col * cell + cell / 2, row * cell + cell / 2, cell * 0.16, 0, Math.PI * 2);
          ctx.fill();
        });

      // 6. the cell the product card is pointing at
      if (highlight) {
        const [row, col] = highlight;
        ctx.strokeStyle = INK.pending;
        ctx.lineWidth = 2;
        ctx.strokeRect(col * cell + 1, row * cell + 1, cell - 2, cell - 2);
        ctx.lineWidth = 1;
      }

      // 7. moving traffic
      (frame.obstacles ?? []).forEach((obstacle) => {
        const [row, col] = obstacle.position;
        ctx.fillStyle = INK.obstacle;
        ctx.beginPath();
        ctx.arc(col * cell + cell / 2, row * cell + cell / 2, cell * 0.26, 0, Math.PI * 2);
        ctx.fill();
      });

      // 8. the robots, each labelled, with a battery arc
      (frame.robots ?? []).forEach((robot) => {
        const [row, col] = robot.position;
        const x = col * cell + cell / 2;
        const y = row * cell + cell / 2;
        const radius = cell * 0.34;

        ctx.fillStyle = !robot.alive
          ? INK.robotDead
          : robot.carrying
            ? INK.robotCarrying
            : INK.robot;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();

        if (robot.alive) {
          const level = Math.max(0, Math.min(1, (robot.battery ?? 0) / 100));
          ctx.strokeStyle = INK.robot;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, radius + 2.5, -Math.PI / 2, -Math.PI / 2 + level * Math.PI * 2);
          ctx.stroke();
          ctx.lineWidth = 1;
        }

        if (cell >= MIN_LABEL_CELL) {
          ctx.fillStyle = INK.paper;
          ctx.font = `600 ${Math.round(cell * 0.42)}px ${MONO}`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(String(robot.id), x, y + 0.5);
        }
      });
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, [layout, frame, highlight]);

  return (
    <div className="floor-canvas" ref={wrapperRef}>
      <canvas ref={canvasRef} />
    </div>
  );
}
