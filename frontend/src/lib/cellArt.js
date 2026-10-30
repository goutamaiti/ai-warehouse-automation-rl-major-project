/**
 * How a warehouse cell is drawn - one vocabulary, used everywhere.
 *
 * The problem this solves
 * -----------------------
 * Flattening the site to a single hue left the six cell types as six greys,
 * and six greys is four too many: #4a5057 (storage) against #8b9299 (packing)
 * and #e8eaec (shelf) against #c3c8cd (charging) are not tellable apart at the
 * size of a palette swatch, let alone on a projector. The editor's palette was
 * the worst of it - you could not see which tool you had selected, which makes
 * a painting tool useless.
 *
 * So the types are told apart by MARK, not by tint:
 *
 *   empty      nothing, just the hairline grid
 *   wall       the whole cell, solid ink
 *   shelf      a block with a lit top face and a darker exposed side
 *   storage    a filled square, inset
 *   packing    an outlined square, inset
 *   charging   a dashed outlined square, inset
 *
 * Three of those are filled shapes, three are not; two are inset outlines told
 * apart by dash. That survives greyscale printing, a washed-out projector and
 * a 6px cell, none of which a four-grey ramp does.
 *
 * Storage, packing and charging are inset marks rather than filled cells for a
 * second reason: a robot STANDS on them, they are not blockages. Filled, the
 * realistic floor's 306 pick points read as a dark checkerboard sealing every
 * aisle while the actual racking looked like empty space - the floor rendered
 * inside out.
 *
 * Why one module
 * --------------
 * This vocabulary was invented for the live fleet floor and lived inside
 * FleetCanvas, so the editor and the replay canvas kept their flat fills and
 * the three views of the same warehouse disagreed about what a warehouse looks
 * like. Painting a packing station and watching it play back as a grey square
 * is the kind of thing a reader notices and cannot unsee. One module, one
 * answer.
 */

import { CellType as CELL } from './grid';
import { MAP_THEMES } from './mapTheme';

/**
 * Ink for each type, from the map palette (lib/mapTheme). Null means "draw
 * nothing" - an aisle is the floor. Each station type has its own colour as
 * well as its own mark, so a delivery station and a charger can be told apart
 * by either.
 */
function markInk(type, ink) {
  if (type === CELL.WALL) return ink.wall;
  if (type === CELL.SHELF) return ink.shelf;
  if (type === CELL.STORAGE) return ink.storage;
  if (type === CELL.PACKING) return ink.packing;
  if (type === CELL.CHARGING) return ink.charging;
  return null;
}

/**
 * A pick point whose product is waiting to be collected: the same inset box
 * as every pick point, filled green instead. No ring or extra mark - the box
 * itself changing colour is the highlight. An urgent one gets a darker edge.
 */
export function drawActivePickup(ctx, x, y, cell, palette = MAP_THEMES.light, urgent = false) {
  const inset = cell * 0.22;
  const size = cell - inset * 2;
  ctx.fillStyle = palette.activePickup;
  ctx.fillRect(x + inset, y + inset, size, size);
  if (urgent) {
    ctx.strokeStyle = palette.activePickupUrgent;
    ctx.lineWidth = Math.max(1.5, cell * 0.1);
    ctx.strokeRect(x + inset, y + inset, size, size);
    ctx.lineWidth = 1;
  }
}

/**
 * The palette, in the order a person builds a warehouse: the structure first,
 * then the three things that give it a purpose. `mark` names the shape so the
 * legend can describe it in words as well as draw it.
 */
export const CELL_LEGEND = [
  { type: CELL.EMPTY, label: 'Aisle', mark: 'blank', hint: 'walkable floor' },
  { type: CELL.WALL, label: 'Wall', mark: 'solid', hint: 'blocks movement' },
  { type: CELL.SHELF, label: 'Racking', mark: 'block', hint: 'blocks movement' },
  { type: CELL.STORAGE, label: 'Pick point', mark: 'filled', hint: 'a package is collected here' },
  { type: CELL.PACKING, label: 'Packing', mark: 'outline', hint: 'a package is delivered here' },
  { type: CELL.CHARGING, label: 'Charger', mark: 'dashed', hint: 'the robot starts and recharges here' },
];

/**
 * One cell's mark, drawn at (x, y) in canvas pixels.
 *
 * `exposed` says whether the cell below this one is NOT racking, i.e. whether
 * this block's side face can be seen. Passing it in rather than looking it up
 * keeps this function ignorant of the grid, so a palette swatch can call it
 * with no layout at all.
 */
export function drawCellMark(ctx, type, x, y, cell, exposed = true, palette = MAP_THEMES.light) {
  const ink = markInk(type, palette);
  if (!ink) return;

  if (type === CELL.WALL) {
    ctx.fillStyle = ink;
    ctx.fillRect(x, y, cell, cell);
    return;
  }

  if (type === CELL.SHELF) {
    const lift = Math.max(2, Math.round(cell * 0.22));
    if (exposed) {
      ctx.fillStyle = palette.shelfSide;
      ctx.fillRect(x, y + cell - lift, cell, lift);
    }
    ctx.fillStyle = ink;
    ctx.fillRect(x, y, cell, cell - (exposed ? lift : 0));
    return;
  }

  const inset = cell * 0.22;
  const size = cell - inset * 2;
  if (type === CELL.STORAGE) {
    ctx.fillStyle = ink;
    ctx.fillRect(x + inset, y + inset, size, size);
    return;
  }
  ctx.strokeStyle = ink;
  ctx.lineWidth = Math.max(1, cell * 0.08);
  if (type === CELL.CHARGING) ctx.setLineDash([cell * 0.14, cell * 0.1]);
  ctx.strokeRect(x + inset, y + inset, size, size);
  ctx.setLineDash([]);
}

/**
 * The whole static map: paper, hairline grid, then every cell's mark.
 *
 * `offsetX`/`offsetY` exist for the views that reserve a ruler gutter.
 */
export function drawFloor(
  ctx,
  layout,
  cell,
  { offsetX = 0, offsetY = 0, grid = true, palette = MAP_THEMES.light } = {}
) {
  const width = cell * layout.width;
  const height = cell * layout.height;

  ctx.fillStyle = palette.floor;
  ctx.fillRect(offsetX, offsetY, width, height);

  if (grid) {
    ctx.strokeStyle = palette.grid;
    ctx.lineWidth = 1;
    for (let col = 0; col <= layout.width; col += 1) {
      const x = offsetX + col * cell + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, offsetY);
      ctx.lineTo(x, offsetY + height);
      ctx.stroke();
    }
    for (let row = 0; row <= layout.height; row += 1) {
      const y = offsetY + row * cell + 0.5;
      ctx.beginPath();
      ctx.moveTo(offsetX, y);
      ctx.lineTo(offsetX + width, y);
      ctx.stroke();
    }
  }

  for (let row = 0; row < layout.height; row += 1) {
    for (let col = 0; col < layout.width; col += 1) {
      const type = layout.grid[row][col];
      const exposed =
        row + 1 >= layout.height || layout.grid[row + 1][col] !== CELL.SHELF;
      drawCellMark(ctx, type, offsetX + col * cell, offsetY + row * cell, cell, exposed, palette);
    }
  }
}

/**
 * A palette swatch: the same mark, at swatch size, on its own little floor.
 * Drawn rather than approximated in CSS, because the whole point is that the
 * button shows exactly what the brush paints.
 */
export function drawSwatch(canvas, type, size = 18, palette = MAP_THEMES.light) {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = size * ratio;
  canvas.height = size * ratio;
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;

  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = palette.floor;
  ctx.fillRect(0, 0, size, size);
  ctx.strokeStyle = palette.grid;
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, size - 1, size - 1);
  drawCellMark(ctx, type, 0, 0, size, true, palette);
}
