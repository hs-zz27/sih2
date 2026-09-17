'use client';

/**
 * The pipeline board.
 *
 * The published design animated a canned seven-node run on a timer. On the
 * real page that would be a lie, so the board is driven by the SSE stream
 * instead: a node lights when its event arrives, and the clock is the actual
 * elapsed wall time since `run_started`. With no run yet it renders its own
 * skeleton with em-dashes rather than going blank — but it never invents a
 * number.
 *
 * Three corrections against the published board, all checked in the executor:
 *
 * 1. The kind is `STEP`, not `SPECIALIST` — `emit("step", ...)` at
 *    satquery/controller/executor.py:284.
 * 2. There is a `VERIFICATION` node between the steps and confidence —
 *    `emit("verification", ...)` at satquery/controller/executor.py:398. The
 *    published board had no such node, and it is the one that says whether
 *    the sentences in the answer are supported by the payload.
 * 3. The number of step nodes comes from the run. The executor loops over a
 *    plan, so a run may have one step or five, and they are *sequential* —
 *    the board chains them rather than fanning them out in parallel.
 */

import { motion, useReducedMotion } from 'framer-motion';
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';

import type { TraceEvent } from '../lib/events';
import { isCalibrated } from '../lib/events';

const NODE_W = 196;
const NODE_H = 96;
const COL_GAP = 56;
const PAD = 16;
const LANE_Y = 152;
const TOP_Y = 36;
const BOTTOM_Y = 272;
const DESIGN_H = 400;

/**
 * Minimum time a node holds the baton before the next one takes it.
 *
 * The board is driven by the SSE stream, and a cached or trivial run can emit
 * every event inside about a tenth of a second. Bound straight to the events,
 * the whole graph flicked from empty to finished in one frame — technically
 * accurate and completely unreadable. The reveal is rate-limited instead, so
 * the board always plays through the pipeline in order and you can see which
 * stage produced what. It only ever lags a fast run; a slow one is still
 * governed by the events themselves, and the wall clock beside it is the
 * measured number either way.
 */
const REVEAL_MS = 280;

type NodeState = 'idle' | 'active' | 'done';

type BoardNode = {
  id: string;
  kind: string;
  title: string;
  value: string;
  sub: string;
  badge?: { text: string; tone: 'ok' | 'warn' | 'fail' };
  x: number;
  y: number;
  state: NodeState;
};

function ms(value: unknown): string {
  const n = Number(value);
  if (!Number.isFinite(n)) return '—';
  return n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${Math.round(n)} ms`;
}

/**
 * Build the board from whatever events have arrived so far.
 *
 * Every value on a node is read out of an event payload. A node with no event
 * yet shows an em-dash: the board's shape is known in advance, its readings
 * are not.
 */
function buildBoard(
  events: TraceEvent[],
  running: boolean,
): { nodes: BoardNode[]; edges: [string, string][]; phase: string } {
  const first = (name: string) => events.find((e) => e.name === name)?.data;
  const ingest = first('ingest');
  const routing = first('routing');
  const steps = events.filter((e) => e.name === 'step').map((e) => e.data);
  const verification = first('verification');
  const confidence = first('confidence');
  const complete = first('complete');
  const failed = first('error');

  const nodes: BoardNode[] = [];
  const edges: [string, string][] = [];
  let col = 0;
  const at = (c: number) => PAD + c * (NODE_W + COL_GAP);

  const settle = (has: unknown, isNext: boolean): NodeState =>
    has ? 'done' : isNext && running ? 'active' : 'idle';

  // --- ingest -------------------------------------------------------------
  const checks: any[] = ingest?.checks ?? [];
  const counts = {
    pass: checks.filter((c) => c.status === 'PASS').length,
    warn: checks.filter((c) => c.status === 'WARN').length,
    fail: checks.filter((c) => c.status === 'FAIL').length,
  };
  const images: any[] = ingest?.images ?? [];
  nodes.push({
    id: 'ingest',
    kind: 'INGEST',
    title: 'Ingest',
    value: ingest ? `${images.length} scene${images.length === 1 ? '' : 's'} · ${checks.length} checks` : '—',
    sub: ingest
      ? [
          images.map((i) => i.role).join(' · ') || 'no roles',
          // Said on the node, not only in the trace: the whole board describes
          // a run on a crop when one was selected.
          images.some((i) => i.aoi_applied) ? 'cropped to area' : '',
        ]
          .filter(Boolean)
          .join(' · ')
      : '',
    badge: !ingest
      ? undefined
      : counts.fail
        ? { text: 'FAIL', tone: 'fail' }
        : counts.warn
          ? { text: 'WARN', tone: 'warn' }
          : { text: 'PASS', tone: 'ok' },
    x: at(col++),
    y: LANE_Y,
    state: settle(ingest, true),
  });

  // --- routing ------------------------------------------------------------
  nodes.push({
    id: 'route',
    kind: 'ROUTING',
    title: 'Route',
    value: routing?.selected_task ?? '—',
    sub: routing ? `confidence ${Number(routing.confidence ?? 0).toFixed(2)}` : '',
    badge: routing ? { text: 'DONE', tone: 'ok' } : undefined,
    x: at(col++),
    y: LANE_Y,
    state: settle(routing, Boolean(ingest)),
  });
  edges.push(['ingest', 'route']);

  // --- steps, in the order the executor ran them --------------------------
  // A run with no steps still gets one placeholder so the board keeps its
  // shape while the first step is in flight.
  const stepCount = Math.max(steps.length, 1);
  let previous = 'route';
  for (let i = 0; i < stepCount; i++) {
    const step = steps[i];
    const id = `step-${i}`;
    nodes.push({
      id,
      kind: 'STEP',
      title: step?.tool ?? `Step ${i + 1}`,
      value: step ? ms(step.runtime_ms) : '—',
      sub: step ? [step.rationale_tag, step.version].filter(Boolean).join(' · ') : '',
      badge: step ? { text: 'DONE', tone: 'ok' } : undefined,
      x: at(col++),
      y: LANE_Y,
      state: settle(step, i === steps.length && Boolean(routing)),
    });
    edges.push([previous, id]);
    previous = id;
  }

  // --- verification -------------------------------------------------------
  const gate = verification?.entailment_gate;
  const conflicts: string[] = verification?.conflicts ?? [];
  nodes.push({
    id: 'verify',
    kind: 'VERIFICATION',
    title: 'Verify',
    // `retained` alone reads as "verified", and the trace is explicit that it
    // is not: a sentence nothing in the payload speaks to is unverifiable,
    // neither supported nor contradicted. The board shows the split.
    value: gate ? `${gate.retained}/${gate.sentences} retained` : '—',
    sub: gate
      ? `${gate.flagged} flagged · ${gate.unverifiable ?? 0} unverifiable · ${gate.backend}`
      : '',
    badge: !verification
      ? undefined
      : conflicts.length || (gate?.flagged ?? 0) > 0
        ? { text: 'FLAGGED', tone: 'warn' }
        : { text: 'CLEAN', tone: 'ok' },
    x: at(col++),
    y: LANE_Y,
    state: settle(verification, steps.length > 0),
  });
  edges.push([previous, 'verify']);

  // --- confidence and the answer, side by side ----------------------------
  const lastCol = at(col);
  nodes.push({
    id: 'confidence',
    kind: 'CONFIDENCE',
    title: 'Confidence',
    value: confidence ? Number(confidence.final).toFixed(2) : '—',
    sub: confidence
      ? `${confidence.band} · ${isCalibrated(confidence) ? 'calibrated' : 'uncalibrated'}`
      : '',
    badge: confidence ? { text: confidence.band, tone: 'ok' } : undefined,
    x: lastCol,
    y: TOP_Y,
    state: settle(confidence, Boolean(verification)),
  });
  nodes.push({
    id: 'answer',
    kind: failed ? 'ERROR' : 'COMPLETE',
    title: failed ? 'Failed' : complete?.abstained ? 'Abstained' : 'Answer',
    value: failed
      ? 'run stopped'
      : complete
        ? complete.abstained
          ? 'no answer returned'
          : 'persisted'
        : '—',
    sub: complete?.abstain_trigger ?? (complete ? 'overlays ready' : ''),
    badge: failed
      ? { text: 'ERROR', tone: 'fail' }
      : complete
        ? complete.abstained
          ? { text: 'ABSTAIN', tone: 'warn' }
          : { text: 'DONE', tone: 'ok' }
        : undefined,
    x: lastCol,
    y: BOTTOM_Y,
    state: settle(complete ?? failed, Boolean(verification)),
  });
  edges.push(['verify', 'confidence'], ['verify', 'answer']);

  const phase = failed
    ? 'error'
    : complete
      ? complete.abstained
        ? 'abstained'
        : 'complete'
      : verification
        ? 'confidence'
        : steps.length
          ? 'step'
          : routing
            ? 'step'
            : ingest
              ? 'routing'
              : running
                ? 'ingest'
                : 'idle';

  return { nodes, edges, phase };
}

function edgePath(a: BoardNode, b: BoardNode): string {
  const x1 = a.x + NODE_W;
  const y1 = a.y + NODE_H / 2;
  const x2 = b.x;
  const y2 = b.y + NODE_H / 2;
  const dx = Math.max(46, (x2 - x1) * 0.55);
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

export default function Pipeline({
  events,
  running,
  runId,
  startedAt,
}: {
  events: TraceEvent[];
  running: boolean;
  runId: string;
  startedAt: number | null;
}) {
  const reduce = useReducedMotion();
  const fitRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const [stacked, setStacked] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  const [revealed, setRevealed] = useState(0);

  const { nodes, edges, phase } = buildBoard(events, running);
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const designWidth = Math.max(...nodes.map((n) => n.x + NODE_W)) + PAD;

  // How far the events have actually got, against how far the board has been
  // allowed to show. The second chases the first, one node at a time.
  const reached = nodes.filter((n) => n.state !== 'idle').length;

  const displayState = (index: number): NodeState => {
    if (index < revealed) return 'done';
    if (index === revealed && (revealed < reached || running)) return 'active';
    return 'idle';
  };

  const progress = nodes.length ? Math.min(revealed, nodes.length) / nodes.length : 0;

  /* Advance one node per tick until the board has caught up with the stream.
     Under reduced motion there is no reveal to watch, so it jumps straight to
     wherever the events are. */
  useEffect(() => {
    if (revealed >= reached) return;
    if (reduce) {
      setRevealed(reached);
      return;
    }
    const id = window.setTimeout(() => setRevealed((r) => r + 1), REVEAL_MS);
    return () => window.clearTimeout(id);
  }, [revealed, reached, reduce]);

  /* A new run rewinds the board. */
  useEffect(() => {
    setRevealed(0);
  }, [startedAt]);

  /* The wall clock is measured, not scripted: it ticks while the run is open
     and freezes on the last reading when it closes. */
  useEffect(() => {
    if (!startedAt) return;
    if (!running) {
      setElapsed((e) => e);
      return;
    }
    const id = window.setInterval(() => setElapsed((Date.now() - startedAt) / 1000), 100);
    return () => window.clearInterval(id);
  }, [startedAt, running]);

  useEffect(() => {
    if (startedAt === null) setElapsed(0);
  }, [startedAt]);

  const fit = useCallback(() => {
    const box = fitRef.current;
    const stage = stageRef.current;
    if (!box || !stage) return;
    const narrow = box.clientWidth < 900;
    setStacked(narrow);
    if (narrow) {
      stage.style.transform = '';
      box.style.height = '';
      return;
    }
    const scale = Math.min(box.clientWidth / designWidth, 1);
    stage.style.width = `${designWidth}px`;
    stage.style.height = `${DESIGN_H}px`;
    stage.style.transform = `scale(${scale})`;
    box.style.height = `${DESIGN_H * scale}px`;
  }, [designWidth]);

  useLayoutEffect(() => {
    fit();
    window.addEventListener('resize', fit);
    return () => window.removeEventListener('resize', fit);
  }, [fit]);

  return (
    <section className="panel" aria-label="Pipeline">
      <div className="rig-head">
        <span className="live">
          <motion.span
            className={`pulse${running ? '' : ' idle'}`}
            animate={running && !reduce ? { opacity: [1, 0.25, 1], scale: [1, 1.35, 1] } : { opacity: 1, scale: 1 }}
            transition={running && !reduce ? { duration: 2.2, repeat: Infinity, ease: 'easeInOut' } : { duration: 0 }}
          />
          <span className="label">{running ? 'Live pipeline' : 'Pipeline'}</span>
        </span>
        <span className="clock">
          {runId ? runId : 'no run yet'} · <b>{elapsed.toFixed(2)} s</b>
        </span>
        <span className="rail">
          <motion.i
            animate={{ width: `${progress * 100}%` }}
            transition={{ duration: reduce ? 0 : 0.35, ease: [0.16, 0.9, 0.28, 1] }}
          />
        </span>
        <span className="label">{phase}</span>
      </div>

      <div className="stage-fit" ref={fitRef}>
        <div className={`stage${stacked ? ' stacked' : ''}`} ref={stageRef}>
          <svg
            className="wires"
            viewBox={`0 0 ${designWidth} ${DESIGN_H}`}
            aria-hidden="true"
          >
            <defs>
              <linearGradient id="flowGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stopColor="#A16207" stopOpacity="0.22" />
                <stop offset="55%" stopColor="#E8C39E" stopOpacity="0.95" />
                <stop offset="100%" stopColor="#E8C39E" stopOpacity="0.4" />
              </linearGradient>
            </defs>
            {edges.map(([from, to]) => {
              const a = byId[from];
              const b = byId[to];
              if (!a || !b) return null;
              const d = edgePath(a, b);
              // An edge lights when the node it feeds does, so the flow runs
              // ahead of each node rather than all at once.
              const lit = displayState(nodes.indexOf(b)) !== 'idle';
              return (
                <g key={`${from}-${to}`}>
                  <path className="edge-base" d={d} />
                  <motion.path
                    className="edge-flow"
                    d={d}
                    initial={false}
                    animate={{ pathLength: lit ? 1 : 0 }}
                    transition={{ duration: reduce ? 0 : 0.45, ease: [0.16, 0.9, 0.28, 1] }}
                  />
                </g>
              );
            })}
          </svg>

          {nodes.map((node, index) => {
            const state = displayState(index);
            return (
            <motion.div
              key={node.id}
              className={`node${state === 'idle' ? '' : ` ${state}`}`}
              style={
                stacked
                  ? undefined
                  : { left: node.x, top: node.y, width: NODE_W, height: NODE_H }
              }
              initial={false}
              // The one place a spring belongs: a node taking the baton. Two
              // states, not a keyframe array — a spring can only interpolate
              // between two values, and handing it three throws.
              animate={{ scale: state === 'active' && !reduce ? 1.025 : 1 }}
              transition={{ type: 'spring', stiffness: 260, damping: 14 }}
            >
              <span className="kind">{node.kind}</span>
              <span className="title">{node.title}</span>
              <span className="val">{state === 'idle' ? '—' : node.value}</span>
              <span className="sub">{node.sub}</span>
              {node.badge && state === 'done' && (
                <span
                  className={`badge${node.badge.tone === 'ok' ? '' : ` ${node.badge.tone}`}`}
                >
                  {node.badge.text}
                </span>
              )}
              <motion.span
                className="node-fill"
                initial={false}
                animate={{ width: state === 'idle' ? '0%' : '100%' }}
                transition={{
                  duration: reduce ? 0 : state === 'active' ? REVEAL_MS / 1000 : 0.4,
                  ease: state === 'active' ? 'linear' : [0.16, 0.9, 0.28, 1],
                }}
              />
            </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
