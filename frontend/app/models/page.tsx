'use client';

/**
 * Model registry page.
 *
 * Nav has linked to /models since it was written, and this file did not exist:
 * the link was a live 404 while `GET /models` sat there ready to answer. That
 * is the whole reason this page exists.
 *
 * It renders `satquery.report.registry.model_registry()` and nothing else.
 * Every number is read from the file the training or evaluation run wrote, and
 * where the registry records a caveat next to a number, the caveat is rendered
 * next to the number here too — a page showing `mAP 0.2854` without "official
 * test shard, 30k patches, 3 epochs, not comparable to the v0 figure" is the
 * exact failure this project has already corrected twice.
 */

import { useEffect, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

type Checkpoint = {
  name: string;
  task: string;
  training: Record<string, any>;
  metrics: Record<string, any>;
  checkpoints: number;
  latest_checkpoint: string | null;
  caveat: string | null;
};

type Downloaded = {
  key: string;
  repo: string | null;
  licence: string | null;
  sha256: string | null;
  path: string | null;
  used_for: string | null;
};

type CalibrationEntry = {
  method: string;
  T: number | null;
  ece_before: number | null;
  ece_after: number | null;
  n_fit: number | null;
  n_eval: number | null;
  dataset: string | null;
  split_note: string | null;
};

type Registry = {
  checkpoints: Checkpoint[];
  downloaded_models: Downloaded[];
  calibration: {
    calibrated: Record<string, CalibrationEntry>;
    rejected: Record<string, CalibrationEntry>;
  };
  note: string;
};

/**
 * The headline metrics, in the order a reader wants them.
 *
 * `metrics.json` is written by the training run and its shape differs per
 * task, so the page cannot hardcode a key. It ranks by the measures that mean
 * something across tasks and shows the top three; a run whose metrics use none
 * of these still gets its first few numbers rather than an empty cell.
 */
const HEADLINE = ['map', 'bleu4', 'bleu_4', 'miou', 'iou', 'f1', 'precision', 'recall', 'acc'];

/** Counts are not scores. They are still shown, just not ranked as headlines. */
function rank(label: string): number {
  const key = label.toLowerCase();
  const hit = HEADLINE.findIndex((h) => key.endsWith(h) || key.includes(`_${h}`));
  return hit < 0 ? 99 : hit;
}

function formatMetric(value: number): string {
  if (Number.isInteger(value) && Math.abs(value) >= 10) return value.toLocaleString();
  return value.toFixed(4);
}

function headlineMetrics(metrics: Record<string, any>): [string, number][] {
  const flat: [string, number][] = [];
  const walk = (obj: any, prefix = '') => {
    if (obj == null || typeof obj !== 'object') return;
    for (const [key, value] of Object.entries(obj)) {
      const label = prefix ? `${prefix}.${key}` : key;
      if (typeof value === 'number' && Number.isFinite(value)) flat.push([label, value]);
      else if (typeof value === 'object' && prefix === '') walk(value, key);
    }
  };
  walk(metrics);
  return flat.sort((a, b) => rank(a[0]) - rank(b[0])).slice(0, 3);
}

function num(value: number | null | undefined, digits = 4): string {
  return value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
}

function CalibrationRows({
  entries,
  shipped,
}: {
  entries: Record<string, CalibrationEntry>;
  shipped: boolean;
}) {
  return (
    <>
      {Object.entries(entries).map(([head, entry]) => (
        <tr key={head}>
          <td className="name">{head}</td>
          <td>
            {entry.method}
            {entry.dataset ? ` · ${entry.dataset}` : ''}
            {entry.split_note && <span className="cav">{entry.split_note}</span>}
          </td>
          <td className="num">
            {num(entry.ece_before)} → {num(entry.ece_after)}
          </td>
          <td className="num">
            {entry.n_fit ?? '—'} / {entry.n_eval ?? '—'}
          </td>
          <td>
            <span className={`tag ${shipped ? 'ok' : 'no'}`}>
              {shipped ? 'SHIPPED' : 'REJECTED'}
            </span>
          </td>
        </tr>
      ))}
    </>
  );
}

export default function ModelsPage() {
  const [registry, setRegistry] = useState<Registry | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/models`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setRegistry)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <main className="shell">
        <div className="page-head">
          <h1>Model registry</h1>
        </div>
        <p className="load-error">Could not load the registry: {error}</p>
      </main>
    );
  }

  if (!registry) {
    return (
      <main className="shell">
        <div className="page-head">
          <span className="label">/models</span>
          <h1>Model registry</h1>
          <p>Reading what the training runs wrote…</p>
        </div>
      </main>
    );
  }

  const rejected = registry.calibration?.rejected ?? {};
  const calibrated = registry.calibration?.calibrated ?? {};

  return (
    <main className="shell">
      <div className="page-head">
        <span className="label">/models</span>
        <h1>
          Every checkpoint, <span className="accent">with its caveat</span>
        </h1>
        <p>{registry.note}</p>
      </div>

      <div className="deck">
        <section className="panel">
          <div className="panel-head">
            <span className="label">Trained checkpoints</span>
            <span className="spacer" />
            <span className="meta">{registry.checkpoints.length} directories</span>
          </div>

          {registry.checkpoints.length === 0 ? (
            <p className="cap">
              No checkpoint directory on this machine — train a head, or mount the
              checkpoints volume, and this table fills itself.
            </p>
          ) : (
            <div className="tablewrap">
              <table className="reg">
                <thead>
                  <tr>
                    <th>Checkpoint</th>
                    <th>Task</th>
                    <th style={{ textAlign: 'right' }}>Measured</th>
                    <th>Steps</th>
                    <th>What the number does not say</th>
                  </tr>
                </thead>
                <tbody>
                  {registry.checkpoints.map((entry) => (
                    <tr key={entry.name}>
                      <td className="name">{entry.name}</td>
                      <td>{entry.task}</td>
                      <td className="num">
                        {headlineMetrics(entry.metrics).map(([label, value]) => (
                          <div key={label}>
                            <span style={{ color: 'var(--dim)' }}>{label}</span>{' '}
                            {formatMetric(value)}
                          </div>
                        ))}
                        {headlineMetrics(entry.metrics).length === 0 && '—'}
                      </td>
                      <td className="num">
                        {entry.checkpoints}
                        {entry.latest_checkpoint && (
                          <div style={{ color: 'var(--dim)', fontSize: 10 }}>
                            {entry.latest_checkpoint}
                          </div>
                        )}
                      </td>
                      <td>
                        {entry.caveat ? (
                          <span className="cav">{entry.caveat}</span>
                        ) : (
                          <span className="cap">no caveat recorded</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <div className="deck-row row-2b">
          <section className="panel">
            <div className="panel-head">
              <span className="label">Downloaded weights</span>
              <span className="spacer" />
              <span className="meta">configs/model_lock.json</span>
            </div>
            {registry.downloaded_models.length === 0 ? (
              <p className="cap">No lock file on this machine.</p>
            ) : (
              <div className="tablewrap">
                <table className="reg" style={{ minWidth: 420 }}>
                  <thead>
                    <tr>
                      <th>Key</th>
                      <th>Repository</th>
                      <th>Digest</th>
                    </tr>
                  </thead>
                  <tbody>
                    {registry.downloaded_models.map((model) => (
                      <tr key={model.key}>
                        <td className="name">{model.key}</td>
                        <td>
                          {model.repo ?? '—'}
                          {model.used_for && <span className="cav">{model.used_for}</span>}
                        </td>
                        <td className="num">
                          {model.sha256 ? (
                            <span title={model.sha256}>{model.sha256.slice(0, 10)}…</span>
                          ) : (
                            <span className="tag hold">NO DIGEST</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="panel">
            <div className="panel-head">
              <span className="label">Calibration</span>
              <span className="spacer" />
              <span className="meta">configs/calibration.json</span>
            </div>
            <div className="tablewrap">
              <table className="reg" style={{ minWidth: 520 }}>
                <thead>
                  <tr>
                    <th>Head</th>
                    <th>Method &amp; split</th>
                    <th style={{ textAlign: 'right' }}>ECE before → after</th>
                    <th style={{ textAlign: 'right' }}>n fit / eval</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  <CalibrationRows entries={calibrated} shipped />
                  <CalibrationRows entries={rejected} shipped={false} />
                </tbody>
              </table>
            </div>
            {Object.keys(rejected).length > 0 && (
              <p className="caveat">
                <b>Rejected fits are listed, not hidden.</b> “We measured this and
                declined to ship it” is a stronger claim than silence, and it stops
                anyone re-deriving the same rejected temperature later.
              </p>
            )}
          </section>
        </div>
      </div>

      <footer className="foot">
        <span>SatQuery AI · model registry</span>
        <span>read from disk on every request — nothing here is recomputed</span>
      </footer>
    </main>
  );
}
