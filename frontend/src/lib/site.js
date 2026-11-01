/** The product's working name. Change it here and in index.html. */
export const BRAND = 'FloorCheck';

export const PAGES = [
  { id: 'home', label: 'Home', href: '#/' },
  { id: 'test', label: 'Test a plan', href: '#/test' },
  { id: 'watch', label: 'Watch it run', href: '#/watch' },
  { id: 'research', label: 'The research', href: '#/research' },
];

/** `#/research/method` -> ['research', 'method'], and anything unknown -> home. */
export function parseHash(hash) {
  const [page, view] = hash.replace(/^#\/?/, '').split('/');
  return [PAGES.some((p) => p.id === page) ? page : 'home', view ?? null];
}
