/**
 * Map colours, one palette per site theme - the single place a map's colours
 * are decided, used by every canvas (replay, editor, live floor, swatches) and
 * by the legends that explain them.
 *
 * The map is the one place on the site with hue, on purpose, so every entity
 * can be told apart from the back of a room: a blue floor and grid, a robot
 * that is black when empty and turns YELLOW when it is carrying, a pick point
 * that turns GREEN while its product is waiting to be collected, red delivery
 * stations, violet chargers and magenta moving obstacles. Yellow is used for
 * nothing else, so "carrying" can never be mistaken for anything.
 *
 * Canvas needs plain colour strings, not CSS variables, hence a JS module.
 */

import { useEffect, useState } from 'react';

export const MAP_THEMES = {
  light: {
    page: '#ffffff',
    floor: '#e8f1ff',
    grid: '#9dbceb',
    wall: '#1e3a8a',
    shelf: '#a9b6c8',
    shelfSide: '#7a889e',
    storage: '#1e3a8a',
    packing: '#dc2626',
    charging: '#7c3aed',
    //: The pick point whose product is waiting to be collected.
    activePickup: '#16a34a',
    activePickupUrgent: '#14532d',
    obstacle: '#db2777',
    arrow: '#2563eb',
    ruler: '#64748b',
    zoneA: 'rgba(37, 99, 235, 0.03)',
    zoneB: 'rgba(37, 99, 235, 0.07)',
    waiting: 'rgba(30, 58, 138, 0.35)',
    highlight: '#1d4ed8',
    trail: 'rgba(15, 23, 42, 0.14)',
    dropoff: '#64748b',
    robot: '#000000',
    robotLabel: '#ffffff',
    robotCarrying: '#facc15',
    robotCarryingLabel: '#000000',
    robotOutline: '#000000',
    robotDead: '#b6bec9',
    battery: '#000000',
    batteryLow: '#64748b',
    batteryCritical: '#b6bec9',
  },
  // Dark: the same hues lifted for a navy floor. The empty robot is WHITE
  // because black would vanish into the floor; carrying stays yellow and a
  // waiting product stays green in both themes, so the colours always mean
  // the same thing.
  dark: {
    page: '#121417',
    floor: '#0b1b36',
    grid: '#24508f',
    wall: '#4b6fb0',
    shelf: '#33496b',
    shelfSide: '#4f6890',
    storage: '#93c5fd',
    packing: '#f87171',
    charging: '#a78bfa',
    activePickup: '#4ade80',
    activePickupUrgent: '#bbf7d0',
    obstacle: '#f472b6',
    arrow: '#60a5fa',
    ruler: '#a3aab1',
    zoneA: 'rgba(147, 197, 253, 0.03)',
    zoneB: 'rgba(147, 197, 253, 0.07)',
    waiting: 'rgba(147, 197, 253, 0.40)',
    highlight: '#60a5fa',
    trail: 'rgba(241, 245, 249, 0.16)',
    dropoff: '#a3aab1',
    robot: '#f8fafc',
    robotLabel: '#0b1b36',
    robotCarrying: '#facc15',
    robotCarryingLabel: '#000000',
    robotOutline: '#0b1b36',
    robotDead: '#4b5a70',
    battery: '#f8fafc',
    batteryLow: '#a3aab1',
    batteryCritical: '#4b5a70',
  },
};

function currentTheme() {
  if (typeof document === 'undefined') return 'light';
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
}

/** The palette for the site's current theme, re-rendering when it changes. */
export function useMapTheme() {
  const [theme, setTheme] = useState(currentTheme);
  useEffect(() => {
    const observer = new MutationObserver(() => setTheme(currentTheme()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    setTheme(currentTheme());
    return () => observer.disconnect();
  }, []);
  return MAP_THEMES[theme];
}
