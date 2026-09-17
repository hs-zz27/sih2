'use client';

/**
 * Benchmark page (plan task 3.12).
 *
 * Reads `/benchmarks`, which aggregates the JSON reports under docs/assets/.
 * Nothing is recomputed here and nothing is hardcoded.
 *
 * Two deliberate choices:
 *
 * - **Missing reports are listed, not omitted.** A page that silently drops a
 *   report that has not been generated looks complete when it is not.
 * - **Every headline number is shown with the caveat recorded next to it in
 *   its source report.** The reports carry `note`, `caveat` and `provenance`
 *   fields for exactly this reason.
 */

import { useEffect, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

type Benchmarks = {
  available: Record<string, { source: string; data: any }>;
  missing: { name: string; expected_at: string }[];
  regenerate_with: string;
};

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export default function BenchmarksPage() {
  const [data, setData] = useState<Benchmarks | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/benchmarks`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <main className="page">
        <p className="load-error">Could not load benchmarks: {error}</p>
      </main>
    );
  }
  if (!data) {
    return (
      <main className="page">
        <p>Loading…</p>
      </main>
    );
  }

  const a = data.available;
  const adversarial = a.adversarial?.data;
  const ablations = a.ablations?.data?.ablations ?? [];
  const entail = a.entailment?.data;
  const selective = a.selective?.data ?? [];
  const calibration = a.calibration?.data ?? [];
  const stress = a.confidence_stress?.data;
  const soak = a.soak?.data;

  const agentArm = ablations.find((x: any) => x.name?.startsWith('agent'))?.arms?.[
    'monolith (classifier alone)'
  ];
  const cleanHybrid = entail?.scores?.find(
    (s: any) => s.suite === 'clean' && s.backend === 'deterministic+nli',
  );

  return (
    <main className="page">
      <span className="label">/benchmarks</span>
      <h1>Measured, and not yet measured</h1>
      <p className="note">
        Every number is read from the report that produced it. Regenerate with{' '}
        <code>{data.regenerate_with}</code>.
      </p>

      {data.missing.length > 0 && (
        <section className="card warn">
          <h3>Reports not generated</h3>
          <ul>
            {data.missing.map((m) => (
              <li key={m.name}>
                <strong>{m.name}</strong> — expected at <code>{m.expected_at}</code>
              </li>
            ))}
          </ul>
        </section>
      )}

      <h2>Headline</h2>
      <div className="stats">
        {adversarial && (
          <Stat
            label="illegal plans"
            value={`${adversarial.illegal_plans} / ${adversarial.n_plans}`}
            sub={`${adversarial.n_queries} adversarial queries x ${adversarial.n_configs} configs`}
          />
        )}
        {agentArm && (
          <Stat
            label="ungated classifier"
            value={`${(agentArm.illegal_plan_rate * 100).toFixed(1)}%`}
            sub="impossible plans without config gating"
          />
        )}
        {cleanHybrid && (
          <Stat
            label="entailment gate (clean suite)"
            value={`${(cleanHybrid.accuracy * 100).toFixed(0)}%`}
            sub={`${cleanHybrid.dangerous_false_retained} false sentences retained`}
          />
        )}
        {soak?.steady_state && (
          <Stat
            label="soak RSS slope"
            value={`${soak.steady_state.rss_slope_mb_per_iteration.toFixed(4)} MB/query`}
            sub={`${soak.iterations} iterations, ${soak.warmup_excluded} warm-up excluded`}
          />
        )}
      </div>

      <h2>Calibration</h2>
      <table className="table">
        <thead>
          <tr>
            <th>head</th>
            <th>method</th>
            <th>ECE before</th>
            <th>ECE after</th>
            <th>Brier before</th>
            <th>Brier after</th>
            <th>accepted</th>
          </tr>
        </thead>
        <tbody>
          {calibration.map((r: any, i: number) => (
            <tr key={i} className={r.accepted ? '' : 'muted'}>
              <td>{r.head}</td>
              <td>{r.method}</td>
              <td>{r.before.ece.toFixed(4)}</td>
              <td>{r.after.ece.toFixed(4)}</td>
              <td>{r.before.brier.toFixed(5)}</td>
              <td>{r.after.brier.toFixed(5)}</td>
              <td>{r.accepted ? 'yes' : `no - ${r.rejection_reason}`}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Selective prediction</h2>
      <p className="note">
        E-AURC, not AURC, is the comparable number: AURC mostly reflects
        accuracy, so a model that is often wrong scores high even with a perfect
        confidence ranking.
      </p>
      <table className="table">
        <thead>
          <tr>
            <th>signal</th>
            <th>n</th>
            <th>base error</th>
            <th>AURC</th>
            <th>optimal</th>
            <th>E-AURC</th>
          </tr>
        </thead>
        <tbody>
          {selective.map((r: any) => (
            <tr key={r.name}>
              <td>{r.name}</td>
              <td>{r.n.toLocaleString()}</td>
              <td>{r.base_error.toFixed(4)}</td>
              <td>{r.aurc.toFixed(4)}</td>
              <td>{r.aurc_optimal.toFixed(4)}</td>
              <td>
                <strong>{r.e_aurc.toFixed(4)}</strong>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {selective.map((r: any) => (
        <p key={r.name} className="caveat">
          ⚠ {r.note}
        </p>
      ))}

      <h2>Entailment gate</h2>
      <table className="table">
        <thead>
          <tr>
            <th>suite</th>
            <th>backend</th>
            <th>accuracy</th>
            <th>false retained</th>
            <th>true flagged</th>
          </tr>
        </thead>
        <tbody>
          {(entail?.scores ?? []).map((s: any, i: number) => (
            <tr key={i} className={s.suite === 'clean' ? '' : 'muted'}>
              <td>{s.suite}</td>
              <td>{s.backend}</td>
              <td>{(s.accuracy * 100).toFixed(0)}%</td>
              <td>{s.dangerous_false_retained}</td>
              <td>{s.destructive_true_flagged}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {entail?.provenance && (
        <p className="caveat">⚠ tuned suite: {entail.provenance.tuned}</p>
      )}

      <h2>Ablations</h2>
      {ablations.map((ab: any) => (
        <section key={ab.name} className="card">
          <h3>
            {ab.name} <span className="task">{ab.status}</span>
          </h3>
          <p className="question">{ab.question}</p>
          {ab.verdict && (
            <p>
              <strong>{ab.verdict}</strong>
            </p>
          )}
          {ab.caveat && <p className="caveat">⚠ {ab.caveat}</p>}
        </section>
      ))}

      <h2>Confidence stress response</h2>
      {stress && (
        <>
          <p className="note">
            sensitivity {stress.sensitivity_passed}/{stress.total} · specificity{' '}
            {stress.specificity_passed}/{stress.total}
          </p>
          <table className="table">
            <thead>
              <tr>
                <th>stressor</th>
                <th>targets</th>
                <th>Δmodel</th>
                <th>Δagreement</th>
                <th>Δinput_quality</th>
                <th>Δfinal</th>
              </tr>
            </thead>
            <tbody>
              {stress.stressors.map((s: any) => (
                <tr key={s.stressor}>
                  <td>{s.stressor}</td>
                  <td>{s.targets}</td>
                  <td>{s.deltas.model.toFixed(2)}</td>
                  <td>{s.deltas.agreement.toFixed(2)}</td>
                  <td>{s.deltas.input_quality.toFixed(2)}</td>
                  <td>{s.deltas.final.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </main>
  );
}
