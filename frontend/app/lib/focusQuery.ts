/**
 * Jump to the query composer and put the caret in it.
 *
 * Deliberately DOM-based rather than a ref passed down through props or a
 * context: the three things that trigger it — the nav item, the hero cue and
 * the `/` shortcut — live in three different components, and two of them
 * (Nav, and the key handler) have no business knowing how the query page is
 * assembled. The composer publishes a stable id and they all aim at that.
 *
 * Scrolling is left to the element's own `scroll-margin-top`, which tracks the
 * sticky header's real height, so this lands clear of it at any width. It is
 * also silent under prefers-reduced-motion, because `scroll-behavior: smooth`
 * is already switched off there.
 */
export const QUERY_FIELD_ID = 'question';

export function focusQuery(): void {
  const field = document.getElementById(QUERY_FIELD_ID) as HTMLTextAreaElement | null;
  if (!field) return;

  const composer = field.closest('.composer') ?? field;
  composer.scrollIntoView({ block: 'start' });

  // `preventScroll` so focus does not fight the smooth scroll above and snap
  // the page there instantly.
  field.focus({ preventScroll: true });
  const end = field.value.length;
  field.setSelectionRange(end, end);
}
