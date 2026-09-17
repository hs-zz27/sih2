'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Bi-temporal swipe and optical–SAR blend comparators (plan task 2.12).
 *
 * Two interactions over the same two-image primitive:
 *
 * - **swipe** for a bi-temporal pair: a hard vertical wipe. Change detection is
 *   about spotting what moved, and a hard edge makes small differences pop in a
 *   way a cross-fade actively hides.
 * - **blend** for an optical–SAR pair: a continuous opacity mix, because the
 *   point there is seeing how the two modalities *overlay*, not which pixel
 *   belongs to which date.
 *
 * Both are pointer-driven and keyboard-accessible, and both label which image
 * is on which side — an unlabelled comparator invites reading the change
 * backwards.
 */

type Props = {
  api: string;
  runId: string;
  roles: string[];
};

const SWIPE_PAIRS: [string, string][] = [['t1', 't2']];
const BLEND_PAIRS: [string, string][] = [['optical', 'sar']];

export default function Comparator({ api, runId, roles }: Props) {
  const [position, setPosition] = useState(50);
  const [failed, setFailed] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const swipe = SWIPE_PAIRS.find(([a, b]) => roles.includes(a) && roles.includes(b));
  const blend = BLEND_PAIRS.find(([a, b]) => roles.includes(a) && roles.includes(b));
  const pair = swipe ?? blend;
  const mode: 'swipe' | 'blend' = swipe ? 'swipe' : 'blend';

  const onMove = useCallback((clientX: number) => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    setPosition(Math.max(0, Math.min(100, pct)));
  }, []);

  useEffect(() => {
    if (mode !== 'swipe') return;
    const up = () => { dragging.current = false; };
    const move = (e: PointerEvent) => { if (dragging.current) onMove(e.clientX); };
    window.addEventListener('pointerup', up);
    window.addEventListener('pointermove', move);
    return () => {
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointermove', move);
    };
  }, [mode, onMove]);

  if (!pair) return null;
  const [left, right] = pair;
  const url = (role: string) => `${api}/runs/${runId}/preview/${role}?max_edge=768`;

  if (failed) {
    return (
      <section className="panel">
        <div className="panel-head">
          <span className="label">Comparator</span>
        </div>
        <p className="answer empty">
          Previews are unavailable for this run — the source images may no longer
          be on disk.
        </p>
      </section>
    );
  }

  return (
    <section className="panel">
      <div className="panel-head">
        <span className="label">
          <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 4h16v16H4z" />
            <path d="M12 4v16" />
          </svg>
          {mode === 'swipe' ? 'Bi-temporal swipe' : 'Optical–SAR blend'}
        </span>
        <span className="spacer" />
        <span className="meta">preview/{'{role}'} · max_edge 768</span>
      </div>

      <div className="cmp-labels">
        <span>{mode === 'swipe' ? `${left} (earlier)` : left}</span>
        <span>{mode === 'swipe' ? `${right} (later)` : right}</span>
      </div>

      <div
        className="cmp"
        ref={containerRef}
        onPointerDown={(e) => {
          if (mode !== 'swipe') return;
          dragging.current = true;
          onMove(e.clientX);
        }}
      >
        <img src={url(left)} alt={left} onError={() => setFailed(true)} />
        <img
          src={url(right)}
          alt={right}
          className="cmp-overlay"
          onError={() => setFailed(true)}
          style={
            mode === 'swipe'
              ? { clipPath: `inset(0 0 0 ${position}%)` }
              : { opacity: position / 100 }
          }
        />
        {mode === 'swipe' && (
          <div className="cmp-handle" style={{ left: `${position}%` }} />
        )}
      </div>

      <input
        type="range"
        min={0}
        max={100}
        value={position}
        aria-label={mode === 'swipe' ? 'Swipe position' : 'Blend amount'}
        onChange={(e) => setPosition(Number(e.target.value))}
        style={{ width: '100%', marginTop: 10 }}
      />
      <p className="cap">
        {mode === 'swipe'
          ? 'Drag the divider, or use the slider / arrow keys.'
          : `Showing ${Math.round(position)}% ${right} over ${left}.`}
      </p>
    </section>
  );
}
