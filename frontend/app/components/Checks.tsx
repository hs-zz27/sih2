'use client';

/**
 * Input checks, as a list.
 *
 * Status is carried by a mark, a colour and a word, never by colour alone —
 * the same rule the telemetry chart follows for its two series.
 */

import type { Check } from '../lib/events';

const MARK: Record<Check['status'], string> = {
  PASS: '✓',
  WARN: '!',
  FAIL: '×',
};

export default function Checks({ checks }: { checks: Check[] }) {
  if (checks.length === 0) return null;
  return (
    <ul className="checks">
      {checks.map((check, i) => (
        <li key={`${check.name}-${i}`} className={check.status.toLowerCase()}>
          <span className="mk" aria-hidden="true">
            {MARK[check.status]}
          </span>
          <span>{check.message}</span>
          <span className="st">{check.status}</span>
        </li>
      ))}
    </ul>
  );
}
