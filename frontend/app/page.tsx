'use client';

import dynamic from 'next/dynamic';
import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import Comparator from './Comparator';
import MapView from './MapView';
import Checks from './components/Checks';
import ConfidenceCard from './components/ConfidenceCard';
import Enter from './components/Enter';
import Pipeline from './components/Pipeline';
import QueryComposer from './components/QueryComposer';
import Telemetry from './components/Telemetry';
import type { Check, Confidence, TraceEvent, Verification } from './lib/events';
import { isCalibrated, parseSSE } from './lib/events';
import { hasGeoreference, sceneFootprint, type Bounds } from './lib/footprint';
import { focusQuery } from './lib/focusQuery';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// WebGL has no business running on the server, and the hero must not block
// the query console behind a 150 kB download either.
const Satellite = dynamic(() => import('./components/Satellite'), { ssr: false });

function summarise(event: TraceEvent): string {
  const d = event.data;
  if (d == null) return '';
  switch (event.name) {
    case 'run_started':
      return `run_id ${d.run_id} · ${d.images ?? ''} image(s)`;
    case 'ingest':
      return (d.checks ?? [])
        .map((c: Check) => `[${c.status}] ${c.message}`)
        .join('\n');
    case 'routing':
      return `selected_task ${d.selected_task}`;
    case 'step':
      return `${d.tool} · ${d.runtime_ms} ms · ${d.rationale_tag ?? ''}`;
    case 'verification': {
      const g = d.entailment_gate ?? {};
      return `${g.retained}/${g.sentences} retained · ${g.flagged} flagged · ${g.unverifiable ?? 0} unverifiable · ${g.backend}`;
    }
    case 'confidence':
      return `${Number(d.final).toFixed(4)} ${d.band}`;
    case 'complete':
      return d.abstained ? `abstained · ${d.abstain_trigger ?? ''}` : 'answer persisted';
    case 'error':
      return String(d.message ?? d);
    default:
      return JSON.stringify(d).slice(0, 400);
  }
}

export default function Page() {
  const [query, setQuery] = useState('Describe this image.');
  // File[] rather than FileList: the composer removes attachments one at a
  // time and adds to the set across several picks, and a FileList is
  // read-only.
  const [files, setFiles] = useState<File[]>([]);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [answer, setAnswer] = useState<string>('');
  const [task, setTask] = useState<string>('');
  const [confidence, setConfidence] = useState<Confidence | null>(null);
  const [verification, setVerification] = useState<Verification | null>(null);
  const [checks, setChecks] = useState<Check[]>([]);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string>('');
  const [runId, setRunId] = useState<string>('');
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  // The run id arrives on `run_started`, but the trace is only persisted when
  // the run finishes: `/runs/{id}/overlays` 404s until then, by design. This
  // flag is what separates "the run exists" from "the run can be queried".
  const [runComplete, setRunComplete] = useState(false);
  const [abstained, setAbstained] = useState(false);
  const [roles, setRoles] = useState<string[]>([]);
  const [footprint, setFootprint] = useState<Bounds | null>(null);
  const [geolocatable, setGeolocatable] = useState(false);
  // The area drawn on the map, [west, south, east, north] in EPSG:4326. The
  // composer owns picking it; the run sends it.
  const [aoi, setAoi] = useState<Bounds | null>(null);
  const traceRef = useRef<HTMLDivElement>(null);
  const pipelineRef = useRef<HTMLDivElement>(null);

  // The composer owns its own <form> and calls this after preventing the
  // default, so this takes no event.
  const submit = useCallback(
    async () => {
      if (files.length === 0) {
        setError('Attach one or two scenes first.');
        return;
      }

      const began = Date.now();
      setRunning(true);
      setError('');
      setAbstained(false);
      setEvents([]);
      setAnswer('');
      setTask('');
      setConfidence(null);
      setVerification(null);
      setChecks([]);
      setRunId('');
      setRunComplete(false);
      setRoles([]);
      setFootprint(null);
      setGeolocatable(false);
      setStartedAt(began);
      setElapsed(null);

      const form = new FormData();
      form.append('query', query);
      files.forEach((f) => form.append('images', f));
      // Only when an area was actually drawn: an absent field means the whole
      // scene, which is what the API defaults to.
      if (aoi) form.append('aoi', JSON.stringify(aoi));

      try {
        const res = await fetch(`${API}/runs/stream`, { method: 'POST', body: form });
        if (!res.ok || !res.body) {
          throw new Error(`server returned ${res.status}`);
        }

        for await (const event of parseSSE(res.body)) {
          setEvents((prev) => [...prev, event]);

          if (event.name === 'run_started') {
            setRunId(event.data.run_id ?? '');
          } else if (event.name === 'ingest') {
            setChecks(event.data.checks ?? []);
            setRoles((event.data.images ?? []).map((i: any) => i.role));
            // The scene's own position, so the map opens on the imagery even
            // when the run writes no raster overlays.
            setFootprint(sceneFootprint(event.data.images));
            setGeolocatable(hasGeoreference(event.data.images));
          } else if (event.name === 'routing') {
            setTask(event.data.selected_task ?? '');
          } else if (event.name === 'verification') {
            // Previously dropped on the floor. It is the event that says
            // whether the sentences in the answer are supported by the
            // payload, which is not a detail worth hiding.
            setVerification(event.data);
          } else if (event.name === 'confidence') {
            setConfidence(event.data);
          } else if (event.name === 'complete') {
            // The API calls store.complete() BEFORE emitting this event, so
            // by the time it arrives the trace is persisted and the overlay
            // endpoints will answer.
            setRunComplete(true);
            setAnswer(event.data.answer ?? '');
            setAbstained(Boolean(event.data.abstained));
            if (event.data.abstained && event.data.abstain_reason) {
              setError(event.data.abstain_reason);
            }
          } else if (event.name === 'error') {
            setError(event.data.message ?? 'run failed');
          }

          // Keep the newest trace line in view as the run progresses.
          requestAnimationFrame(() => {
            if (traceRef.current) {
              traceRef.current.scrollTop = traceRef.current.scrollHeight;
            }
          });
        }
      } catch (err: any) {
        setError(err?.message ?? String(err));
      } finally {
        setRunning(false);
        setElapsed((Date.now() - began) / 1000);
      }
    },
    [files, query, aoi],
  );

  /**
   * Move to the pipeline when a run opens — it is the thing that plays while
   * the run is live, so it is what you want on screen.
   *
   * Keyed on `startedAt` rather than called inside submit(), and deliberately
   * a frame later. Calling it inline scrolled nothing: the reset that starts a
   * run (`setEvents([])` and the rest) re-renders the board a moment
   * afterwards, the page height changes under the in-flight smooth scroll, and
   * the browser abandons it. Waiting for the commit and one frame of layout
   * means the target is where it will still be when the scroll arrives.
   */
  useEffect(() => {
    if (startedAt === null) return;
    const frame = requestAnimationFrame(() => {
      pipelineRef.current?.scrollIntoView({ block: 'start' });
    });
    return () => cancelAnimationFrame(frame);
  }, [startedAt]);

  /**
   * `/` jumps to the composer, the convention every search-shaped tool uses.
   *
   * Ignored while the caret is already in a field, so typing a path or a
   * fraction into the question does not teleport the page. The nav item and
   * the hero cue do the same thing for anyone not reaching for a shortcut —
   * this is never the only way in.
   */
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== '/' || event.metaKey || event.ctrlKey || event.altKey) return;
      // `event.target` is not always an Element — a key pressed with nothing
      // focused targets the document, and `closest` does not exist there.
      const target = event.target;
      if (
        target instanceof Element &&
        target.closest('input, textarea, select, [contenteditable]')
      ) {
        return;
      }
      event.preventDefault();
      focusQuery();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const gate = verification?.entailment_gate;
  const checkCounts = useMemo(
    () => ({
      pass: checks.filter((c) => c.status === 'PASS').length,
      warn: checks.filter((c) => c.status === 'WARN').length,
      fail: checks.filter((c) => c.status === 'FAIL').length,
    }),
    [checks],
  );

  return (
    <>
      <div className="shell">
        <header className="hero">
          {/* Order matters: sunglow behind, rig in front of it, planet limb
              last. The rig carries its own stacking level because `.limb` is a
              positioned sibling that comes after it in the DOM and would
              otherwise paint straight over the satellite. */}
          <div className="hero-sky" aria-hidden="true">
            <div className="sunglow" />
            <div className="hero-rig">
              <Satellite />
            </div>
            <div className="limb" />
          </div>

          <div className="hero-inner">
            <Enter index={0}>
              <div className="hero-copy">
                <span className="label">Vision-language assistant · remote sensing</span>
                <h1>
                  Ask the imagery.
                  <br />
                  <span className="accent">Watch it reason.</span>
                </h1>
                <p className="hero-lede">
                  Panchromatic, multispectral or SAR — one question in plain language.
                  SatQuery routes it to the right specialist and streams every step, so
                  the answer arrives <b>with its evidence and its doubt attached</b>.
                </p>
                <button type="button" className="hero-cue" onClick={focusQuery}>
                  Ask a question
                  <kbd>/</kbd>
                  <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 5v14M5 12l7 7 7-7" />
                  </svg>
                </button>
              </div>
            </Enter>
          </div>
        </header>

        <Enter index={1}>
          <QueryComposer
            query={query}
            onQueryChange={setQuery}
            files={files}
            onFilesChange={setFiles}
            running={running}
            onSubmit={submit}
            error={error}
            aoi={aoi}
            onAoiChange={setAoi}
          />
        </Enter>

        <main className="deck">
          <Enter index={2} ref={pipelineRef} className="pipeline-anchor">
            <Pipeline
              events={events}
              running={running}
              runId={runId}
              startedAt={startedAt}
            />
          </Enter>

          <Enter index={3}>
            <div className="vitals">
              <div className="vital">
                <div className="k">Run</div>
                <div className="v">{runId || '—'}</div>
                <div className="n">
                  {roles.length ? roles.join(' · ') : 'no scenes ingested yet'}
                </div>
              </div>
              <div className="vital">
                <div className="k">Wall clock</div>
                <div className="v accent">
                  {elapsed == null ? '—' : `${elapsed.toFixed(2)} s`}
                </div>
                <div className="n">measured in the browser</div>
              </div>
              <div className="vital">
                <div className="k">Task</div>
                <div className="v">{task || '—'}</div>
                <div className="n">{task ? 'routed, not asked for' : 'awaiting routing'}</div>
              </div>
              <div className="vital">
                <div className="k">Confidence</div>
                <div className="v accent">
                  {confidence && !abstained ? confidence.final.toFixed(2) : '—'}
                </div>
                <div className="n">
                  {abstained
                    ? 'n/a — the run abstained'
                    : confidence
                      ? `${confidence.band} · ${isCalibrated(confidence) ? 'calibrated' : 'uncalibrated'}`
                      : 'no confidence event yet'}
                </div>
              </div>
              <div className="vital">
                <div className="k">Checks</div>
                <div className="v">
                  {checks.length
                    ? `${checkCounts.pass} / ${checkCounts.warn} / ${checkCounts.fail}`
                    : '—'}
                </div>
                <div className="n">
                  {checks.length ? 'pass · warn · fail' : 'no ingest event yet'}
                </div>
              </div>
            </div>
          </Enter>

          <Enter index={4}>
            <div className="deck-row row-2">
              <section className="panel">
                <div className="panel-head">
                  <span className="label">
                    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M12 3v18M3 12h18" />
                    </svg>
                    {abstained ? 'Abstained' : 'Answer'}
                  </span>
                  <span className="spacer" />
                  <span className="meta">streamed over SSE</span>
                </div>

                {task && (
                  <div className="taskline">
                    <span className="task">{task}</span>
                    {roles.length > 0 && <span className="rid">{roles.join(' · ')}</span>}
                  </div>
                )}

                <p className={`answer${answer ? '' : ' empty'}`}>
                  {answer || (running ? 'Working…' : 'No run yet — choose imagery above.')}
                </p>

                {/* The run is stored the moment it completes, and
                    docs/rehearsal.md recommends presenting the two 56 s
                    Cartosat beats from their stored permalink rather than
                    re-running them live. */}
                {runId && !running && (
                  <Link className="permalink" href={`/runs/${runId}`}>
                    Permalink to this run ({runId})
                    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M5 12h14M13 6l6 6-6 6" />
                    </svg>
                  </Link>
                )}

                {gate && (
                  <p className="caveat">
                    <b>
                      {gate.retained}/{gate.sentences} sentences retained
                    </b>{' '}
                    by the {gate.backend} entailment gate — {gate.flagged} flagged,{' '}
                    {gate.unverifiable ?? 0} unverifiable. Unverifiable is not the same
                    as supported: it means nothing in the payload could speak to the
                    sentence either way.
                    {verification!.conflicts.length > 0 && (
                      <> Conflicts: {verification!.conflicts.join('; ')}.</>
                    )}
                  </p>
                )}

                <Checks checks={checks} />
              </section>

              <ConfidenceCard confidence={confidence} abstained={abstained} />
            </div>
          </Enter>

          <Enter index={5}>
            <Telemetry />
          </Enter>

          {runId && roles.length === 2 && (
            <Enter index={6}>
              <Comparator api={API} runId={runId} roles={roles} />
            </Enter>
          )}

          {/* Mounted for ANY completed run, not just pairs: a single image
              still produces georeferenced index rasters worth putting on a
              map, and the comparator above only applies to two-image inputs.
              The layer names come from the run's own artifact_paths — MapView
              reads /runs/{id}/overlays and is otherwise untouched. */}
          {runId && (
            <Enter index={7}>
              <section className="panel">
                <div className="panel-head">
                  <span className="label">
                    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M3 6l6-3 6 3 6-3v15l-6 3-6-3-6 3z" />
                      <path d="M9 3v15M15 6v15" />
                    </svg>
                    Georeferenced overlays
                  </span>
                  <span className="spacer" />
                  <span className="meta">/runs/{runId}/overlays</span>
                </div>
                <MapView
                  runId={runId}
                  ready={runComplete}
                  footprint={footprint}
                  geolocatable={geolocatable}
                />
              </section>
            </Enter>
          )}

          <Enter index={8}>
            <section className="panel">
              <div className="panel-head">
                <span className="label">Live trace</span>
                <span className="spacer" />
                <span className="meta">{events.length} events</span>
              </div>
              <div className="trace" ref={traceRef}>
                {events.length === 0 && (
                  <div className="event">
                    <span className="name">idle</span>
                    <pre>The stream prints here as the run progresses.</pre>
                  </div>
                )}
                {events.map((event, i) => (
                  <div className="event" key={i}>
                    <span className="name">{event.name}</span>
                    <pre>{summarise(event)}</pre>
                  </div>
                ))}
              </div>
            </section>
          </Enter>
        </main>

        <footer className="foot">
          <span>SatQuery AI · deck</span>
          <span>every reading on this page comes from the run that produced it</span>
        </footer>
      </div>
    </>
  );
}
