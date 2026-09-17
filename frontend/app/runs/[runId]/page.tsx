'use client';

/**
 * Permalink for a stored run (plan task 1.6).
 *
 * The run store already keeps every completed trace, but nothing linked to
 * one - the map, the answer and the artifacts were reachable only in the tab
 * that happened to submit the query. This makes a run addressable, which is
 * also what lets the map viewer be checked against a run someone else made.
 *
 * Re-skinned to the deck, and given the two things the live page gained: the
 * verification panel, and the same confidence card, so a stored run and a live
 * one cannot end up saying different things about the same numbers.
 */

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import MapView from '@/MapView';
import Checks from '@/components/Checks';
import ConfidenceCard from '@/components/ConfidenceCard';
import { hasGeoreference, sceneFootprint } from '@/lib/footprint';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export default function RunPage() {
  const params = useParams<{ runId: string }>();
  const runId = params?.runId;
  const [record, setRecord] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return;
    fetch(`${API}/runs/${runId}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((d) => {
        // The store wraps the trace; a fresh POST returns it bare. Accept both
        // rather than assuming, and parse it if it came back as a string.
        let trace = d.trace ?? d;
        if (typeof trace === 'string') trace = JSON.parse(trace);
        setRecord(trace);
      })
      .catch((e) => setError(String(e)));
  }, [runId]);

  if (error) {
    return (
      <main className="shell">
        <div className="page-head">
          <span className="label">/runs/{runId}</span>
          <h1>Run not available</h1>
        </div>
        <p className="load-error">
          Could not load run {runId}: {error}
        </p>
      </main>
    );
  }

  if (!record) {
    return (
      <main className="shell">
        <div className="page-head">
          <span className="label">/runs/{runId}</span>
          <h1>Stored run</h1>
          <p>Reading the persisted trace…</p>
        </div>
      </main>
    );
  }

  const confidence = record.confidence ?? null;
  const abstained = Boolean(record.abstained);
  const checks = record.ingest?.checks ?? [];
  const steps = record.execution ?? record.steps ?? [];
  const gate = record.verification?.entailment_gate;
  const conflicts: string[] = record.verification?.conflicts ?? [];
  const footprint = sceneFootprint(record.ingest?.images);
  const geolocatable = hasGeoreference(record.ingest?.images);

  return (
    <main className="shell">
      <div className="page-head">
        <span className="label">/runs/{runId}</span>
        <h1>{abstained ? <span className="accent">Abstained</span> : 'Stored run'}</h1>
        <p>{record.query}</p>
      </div>

      <div className="vitals" style={{ marginTop: 'var(--s3)' }}>
        <div className="vital">
          <div className="k">Run id</div>
          <div className="v">{String(runId)}</div>
          <div className="n">replayed, not re-run</div>
        </div>
        <div className="vital">
          <div className="k">Task</div>
          <div className="v">{record.routing?.selected_task ?? '—'}</div>
          <div className="n">routed from the question</div>
        </div>
        <div className="vital">
          <div className="k">Steps</div>
          <div className="v">{Array.isArray(steps) ? steps.length : '—'}</div>
          <div className="n">tools the plan ran</div>
        </div>
        <div className="vital">
          <div className="k">Confidence</div>
          <div className="v accent">
            {confidence && !abstained ? Number(confidence.final).toFixed(2) : '—'}
          </div>
          <div className="n">
            {abstained ? 'n/a — the run abstained' : (confidence?.band ?? 'not recorded')}
          </div>
        </div>
      </div>

      <div className="deck">
        <div className="deck-row row-2">
          <section className="panel">
            <div className="panel-head">
              <span className="label">{abstained ? 'Abstained' : 'Answer'}</span>
              <span className="spacer" />
              <span className="meta">from the stored trace</span>
            </div>

            {abstained ? (
              <>
                <p className="answer">
                  {record.abstain_reason ?? 'The run returned no answer.'}
                </p>
                <p className="caveat">
                  Trigger: {record.abstain_trigger}. {record.abstain_resolving_input}
                </p>
              </>
            ) : (
              <p className="answer">{record.answer}</p>
            )}

            {gate && (
              <p className="caveat">
                <b>
                  {gate.retained}/{gate.sentences} sentences retained
                </b>{' '}
                by the {gate.backend} entailment gate — {gate.flagged} flagged,{' '}
                {gate.unverifiable ?? 0} unverifiable. Unverifiable means nothing in
                the payload could speak to the sentence either way; it is not a pass.
                {conflicts.length > 0 && <> Conflicts: {conflicts.join('; ')}.</>}
              </p>
            )}

            <Checks checks={checks} />

            <p style={{ marginTop: 'var(--s4)' }}>
              <a className="permalink" href={`${API}/runs/${runId}/report.pdf`}>
                Download the PDF report
                <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M12 3v12" />
                  <path d="M8 11l4 4 4-4" />
                  <path d="M4 19h16" />
                </svg>
              </a>
            </p>
          </section>

          {/* Components stay visible even when abstained: they are the
              diagnosis of *why*, and input_quality is usually the low one. */}
          <ConfidenceCard confidence={confidence} abstained={abstained} />
        </div>

        {Array.isArray(steps) && steps.length > 0 && (
          <section className="panel">
            <div className="panel-head">
              <span className="label">Steps</span>
              <span className="spacer" />
              <span className="meta">{steps.length} executed</span>
            </div>
            <div className="tablewrap">
              <table className="reg">
                <thead>
                  <tr>
                    <th>Tool</th>
                    <th>Version</th>
                    <th>Rationale</th>
                    <th style={{ textAlign: 'right' }}>Runtime</th>
                  </tr>
                </thead>
                <tbody>
                  {steps.map((step: any, i: number) => (
                    <tr key={`${step.tool}-${i}`}>
                      <td className="name">{step.tool}</td>
                      <td>{step.version ?? '—'}</td>
                      <td>{step.rationale_tag ?? '—'}</td>
                      <td className="num">
                        {Number.isFinite(Number(step.runtime_ms))
                          ? `${Math.round(Number(step.runtime_ms))} ms`
                          : '—'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        <section className="panel">
          <div className="panel-head">
            <span className="label">Georeferenced overlays</span>
            <span className="spacer" />
            <span className="meta">/runs/{runId}/overlays</span>
          </div>
          <MapView
            runId={String(runId)}
            footprint={footprint}
            geolocatable={geolocatable}
          />
        </section>
      </div>

      <footer className="foot">
        <span>SatQuery AI · stored run</span>
        <span>{record.run_id ?? runId}</span>
      </footer>
    </main>
  );
}
