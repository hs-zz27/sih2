'use client';

/**
 * The confidence panel.
 *
 * "Confidence score", not "Confidence": the combiner reports an uncalibrated
 * score whenever no learned head contributed, and
 * satquery/controller/calibration.py says so plainly. Rendering it as a
 * percentage asserted a calibrated probability the system does not claim to
 * produce, so the figure stays a two-decimal score and the caveat travels with
 * it.
 *
 * An abstained run returned no answer, so a confidence *for that answer* is
 * not applicable (limitation L18). The figure is withheld rather than shown as
 * zero — the run abstained on input validation, not on low confidence — but
 * the components stay visible, because they are the diagnosis of why.
 */

import { motion, useReducedMotion } from 'framer-motion';

import type { Confidence } from '../lib/events';
import { isCalibrated } from '../lib/events';

const LABELS: Record<string, string> = {
  model: 'Model',
  agreement: 'Agreement',
  input_quality: 'Input',
};

export default function ConfidenceCard({
  confidence,
  abstained,
}: {
  confidence: Confidence | null;
  abstained: boolean;
}) {
  const reduce = useReducedMotion();
  const calibrated = isCalibrated(confidence);

  return (
    <section className="panel">
      <div className="panel-head">
        <span className="label">
          <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 2a10 10 0 1 0 10 10" />
            <path d="M12 7v5l3 3" />
          </svg>
          Confidence score
        </span>
        <span className="spacer" />
        {confidence && (
          <span className="meta">
            {calibrated ? confidence.calibration!.method : 'uncalibrated'}
          </span>
        )}
      </div>

      {!confidence ? (
        <p className="answer empty">No confidence event yet.</p>
      ) : (
        <>
          <div className="score">
            <span className="n">{abstained ? '—' : confidence.final.toFixed(2)}</span>
            <span className={`band ${abstained ? 'NA' : confidence.band}`}>
              {abstained ? 'NOT APPLICABLE' : confidence.band}
            </span>
          </div>

          <div className="comps">
            {Object.entries(confidence.components ?? {}).map(([key, value]) => (
              <div className="comp" key={key}>
                <span className="cl">{LABELS[key] ?? key}</span>
                <span className="track">
                  <motion.i
                    initial={false}
                    animate={{ width: `${Math.max(0, Math.min(1, Number(value))) * 100}%` }}
                    transition={{ duration: reduce ? 0 : 0.45, ease: [0.16, 0.9, 0.28, 1] }}
                  />
                </span>
                <span className="cv">{Number(value).toFixed(2)}</span>
              </div>
            ))}
          </div>

          <p className="caveat">
            {abstained ? (
              <>
                <b>Withheld, not zero.</b> This run abstained, so there is no answer
                for a confidence to be about. The components above still stand — they
                are the diagnosis, and input quality is usually the low one.
              </>
            ) : calibrated ? (
              <>
                <b>Calibrated</b> with {confidence.calibration!.method} (T{' '}
                {confidence.calibration!.T.toFixed(3)}, ECE after{' '}
                {confidence.calibration!.ece_after.toFixed(4)}) on the split recorded
                in configs/calibration.json.
              </>
            ) : (
              <>
                <b>Uncalibrated.</b> No learned head contributed to this run, so the
                score ranks runs against one another — it is not a probability, and
                the interface never presents it as one.
              </>
            )}
          </p>
        </>
      )}
    </section>
  );
}
