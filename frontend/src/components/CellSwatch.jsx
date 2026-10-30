import { useEffect, useRef } from 'react';

import { drawSwatch } from '../lib/cellArt';
import { useMapTheme } from '../lib/mapTheme';

/**
 * One cell type, drawn at swatch size by the renderer that paints the floor.
 *
 * A CSS colour chip can only show the tint, and tint is exactly what stopped
 * telling these apart when the site went to a single hue: a pick point and a
 * packing station were two greys you had to hold side by side to distinguish.
 * They are now a filled square and an outlined one, and that difference only
 * shows if the swatch is the real mark rather than a block of its colour.
 *
 * Used by the editor palette and by the replay canvas legend, so the key and
 * the map cannot drift apart.
 */
export default function CellSwatch({ type, size = 18 }) {
  const ref = useRef(null);
  const palette = useMapTheme();
  useEffect(() => {
    if (ref.current) drawSwatch(ref.current, type, size, palette);
  }, [type, size, palette]);
  return <canvas className="swatch-canvas" ref={ref} aria-hidden="true" />;
}
