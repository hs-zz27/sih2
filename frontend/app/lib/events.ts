/**
 * The SSE contract, in one place.
 *
 * The executor emits eight event names and the frontend previously handled
 * five of them: `step` and `verification` fell on the floor, so the two events
 * that say what the pipeline actually did and whether the answer is supported
 * never reached the interface. They are typed here so a component that wants
 * them cannot mistype the name.
 */

export const EVENT_NAMES = [
  'run_started',
  'ingest',
  'routing',
  'step',
  'verification',
  'confidence',
  'complete',
  'error',
] as const;

export type EventName = (typeof EVENT_NAMES)[number];

export type TraceEvent = { name: EventName | string; data: any };

export type Check = { name: string; status: 'PASS' | 'WARN' | 'FAIL'; message: string };

export type Confidence = {
  final: number;
  band: 'HIGH' | 'MEDIUM' | 'LOW';
  components: { model: number; agreement: number; input_quality: number };
  calibration?: { method: string; T: number; ece_after: number };
};

export type EntailmentGate = {
  sentences: number;
  retained: number;
  flagged: number;
  unverifiable: number;
  backend: string;
  action: string;
  flagged_detail: { sentence: string; reason: string; backend: string; score: number | null }[];
};

export type Verification = {
  physics_agreement: Record<string, number>;
  built_up_path: string;
  complementarity: Record<string, unknown>;
  conflicts: string[];
  entailment_gate: EntailmentGate;
};

/**
 * Whether a confidence score was actually calibrated.
 *
 * The combiner does not write the bare string "uncalibrated". A real run
 * reports `method: "uncalibrated (score is not a calibratable probability)"`
 * with `T: 1.0` and `ece_after: -1.0`, so an equality test against
 * "uncalibrated" reads that as a calibrated fit and prints a temperature of
 * 1.000 and an ECE of -1.0000 as though they were measurements. The sentinel
 * negative ECE is the reliable signal; the prefix check covers the wording.
 */
export function isCalibrated(confidence: Confidence | null): boolean {
  const calibration = confidence?.calibration;
  if (!calibration) return false;
  if (calibration.method.toLowerCase().startsWith('uncalibrated')) return false;
  return Number(calibration.ece_after) >= 0;
}

/**
 * Parses an SSE byte stream into discrete events.
 *
 * EventSource cannot issue a POST with a file body, so the run is started with
 * fetch() and the response stream is decoded by hand. Events are separated by
 * a blank line; a partial event left in the buffer is carried to the next read.
 */
export async function* parseSSE(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<TraceEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split: number;
    while ((split = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);

      let name = 'message';
      const dataLines: string[] = [];
      for (const line of raw.split('\n')) {
        if (line.startsWith('event: ')) name = line.slice(7);
        else if (line.startsWith('data: ')) dataLines.push(line.slice(6));
      }
      if (dataLines.length === 0) continue;
      try {
        yield { name, data: JSON.parse(dataLines.join('\n')) };
      } catch {
        yield { name, data: dataLines.join('\n') };
      }
    }
  }
}
