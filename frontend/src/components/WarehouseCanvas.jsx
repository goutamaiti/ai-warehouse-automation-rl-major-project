import { useEffect, useRef } from 'react';
import { CellType as CELL, CELL_COLORS } from '../lib/grid';

/**
 * Canvas view of one recorded frame.
 *
 * The component is purely presentational: it draws the layout that came with
 * the recording plus the robot, obstacles and task markers of the current
 * frame. It never simulates anything itself.
 */

const COLORS = {
  grid: 'rgba(148, 163, 184, 0.07)',
  trail: 'rgba(125, 211, 252, 0.28)',
  robot: '#60a5fa',
  robotCarrying: '#fbbf24',
  obstacle: '#f43f5e',
  pickup: '#22d3ee',
  dropoff: '#34d399',
  batteryOk: '#34d399',
  batteryLow: '#fbbf24',
  batteryCritical: '#f43f5e',
};

const MAX_CELL_SIZE = 30;

function batteryColor(state) {
  if (state === 'critical') return COLORS.batteryCritical;
  if (state === 'low') return COLORS.batteryLow;
  return COLORS.batteryOk;
}

export default function WarehouseCanvas({ layout, frame, trail = [] }) {
  const canvasRef = useRef(null);
  const wrapperRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!canvas || !wrapper || !layout) return undefined;

    const draw = () => {
      const available = wrapper.clientWidth || 640;
      const cellSize = Math.max(
        6,
        Math.min(MAX_CELL_SIZE, Math.floor(available / layout.width))
      );
      const width = cellSize * layout.width;
      const height = cellSize * layout.height;
      const ratio = window.devicePixelRatio || 1;

      canvas.width = width * ratio;
      canvas.height = height * ratio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const ctx = canvas.getContext('2d');
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      ctx.clearRect(0, 0, width, height);

      // 1. static map
      for (let row = 0; row < layout.height; row += 1) {
        for (let col = 0; col < layout.width; col += 1) {
          const value = layout.grid[row][col];
          ctx.fillStyle = CELL_COLORS[value] ?? CELL_COLORS[CELL.EMPTY];
          ctx.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
          ctx.strokeStyle = COLORS.grid;
          ctx.lineWidth = 1;
          ctx.strokeRect(col * cellSize + 0.5, row * cellSize + 0.5, cellSize - 1, cellSize - 1);
        }
      }

      // 2. where the robot has already been
      ctx.fillStyle = COLORS.trail;
      trail.forEach(([row, col]) => {
        ctx.fillRect(col * cellSize + 2, row * cellSize + 2, cellSize - 4, cellSize - 4);
      });

      if (!frame) return;

      const marker = (position, color) => {
        const [row, col] = position;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(col * cellSize + 2, row * cellSize + 2, cellSize - 4, cellSize - 4);
      };

      // 3. task markers
      if (frame.task) {
        marker(frame.task.pickup, COLORS.pickup);
        marker(frame.task.dropoff, COLORS.dropoff);
      }

      // 4. dynamic obstacles
      ctx.fillStyle = COLORS.obstacle;
      (frame.obstacles ?? []).forEach(({ position }) => {
        const [row, col] = position;
        ctx.beginPath();
        ctx.roundRect(
          col * cellSize + 3,
          row * cellSize + 3,
          cellSize - 6,
          cellSize - 6,
          3
        );
        ctx.fill();
      });

      // 5. robot, with a battery arc around it
      const [robotRow, robotCol] = frame.robot.position;
      const centerX = robotCol * cellSize + cellSize / 2;
      const centerY = robotRow * cellSize + cellSize / 2;
      const radius = Math.max(3, cellSize / 2 - 4);

      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.fillStyle = frame.robot.carrying ? COLORS.robotCarrying : COLORS.robot;
      ctx.fill();
      ctx.strokeStyle = '#04060b';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      const level = Math.max(0, Math.min(1, (frame.robot.battery ?? 0) / 100));
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius + 2.5, -Math.PI / 2, -Math.PI / 2 + level * Math.PI * 2);
      ctx.strokeStyle = batteryColor(frame.robot.battery_state);
      ctx.lineWidth = 2;
      ctx.stroke();
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(wrapper);
    return () => observer.disconnect();
  }, [layout, frame, trail]);

  return (
    <div className="canvas-wrapper" ref={wrapperRef}>
      <canvas ref={canvasRef} />
      <ul className="legend">
        <li><span className="swatch" style={{ background: COLORS.robot }} /> robot</li>
        <li><span className="swatch" style={{ background: COLORS.robotCarrying }} /> carrying</li>
        <li><span className="swatch" style={{ background: COLORS.obstacle }} /> obstacle</li>
        <li><span className="swatch outline" style={{ borderColor: COLORS.pickup }} /> pickup</li>
        <li><span className="swatch outline" style={{ borderColor: COLORS.dropoff }} /> drop-off</li>
        <li><span className="swatch" style={{ background: CELL_COLORS[CELL.SHELF] }} /> shelf</li>
        <li><span className="swatch" style={{ background: CELL_COLORS[CELL.CHARGING] }} /> charger</li>
      </ul>
    </div>
  );
}
