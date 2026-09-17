'use client';

/**
 * Device telemetry — a streaming area chart over a real endpoint.
 *
 * The published design carried this panel with no endpoint behind it, drawing
 * two smooth lines from a client-side random walk. A chart of an invented
 * quantity is worse than no chart, so rather than dropping the panel the
 * missing half was built: `GET /device` returns one sample of what the API
 * process can actually measure.
 *
 * What it can measure differs by machine, and the panel says so:
 *
 * * **VRAM in use** is real — `torch.cuda.mem_get_info` on the active device.
 *   It is the solid series.
 * * **GPU utilisation** needs NVML, which is not a dependency. When the
 *   endpoint returns `null` the legend reads "not instrumented" and no second
 *   line is drawn, instead of deriving something plausible from memory.
 *
 * Chart choices follow the >=1 Hz guidance: canvas, a rolling 180 s buffer,
 * the current value in text next to the chart, a pause control, and two series
 * separated by line style as well as hue — never colour alone.
 */

import { useReducedMotion } from 'framer-motion';
import { useCallback, useEffect, useRef, useState } from 'react';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
const BUFFER = 180;
const CHAMP = '#E8C39E';
const SAGE = '#9DBBA4';

type Sample = {
  device: string;
  name: string | null;
  vram_used_fraction: number | null;
  vram_total_bytes: number | null;
  utilisation: number | null;
  utilisation_source: string | null;
};

type Point = { vram: number | null; util: number | null };

export default function Telemetry() {
  const reduce = useReducedMotion();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const buffer = useRef<Point[]>([]);
  // Flashing content needs a reduced-motion path, so the chart starts paused
  // when the preference is set. Everything else on the panel still reads.
  const [running, setRunning] = useState(true);
  const [sample, setSample] = useState<Sample | null>(null);
  const [reachable, setReachable] = useState<boolean | null>(null);

  useEffect(() => {
    if (reduce) setRunning(false);
  }, [reduce]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return;
    canvas.width = w * ratio;
    canvas.height = h * ratio;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const pad = 10;
    const y = (v: number) => h - pad - (v / 100) * (h - pad * 2);
    const x = (i: number) => pad + (i / (BUFFER - 1)) * (w - pad * 2);

    // Grid at every label the axis names, so a reading can be placed by eye.
    ctx.strokeStyle = 'rgba(154,146,158,0.10)';
    ctx.lineWidth = 1;
    [0, 25, 50, 75, 100].forEach((v) => {
      ctx.beginPath();
      ctx.moveTo(pad, y(v));
      ctx.lineTo(w - pad, y(v));
      ctx.stroke();
    });

    const points = buffer.current;
    const vram = points.map((p) => p.vram);
    const util = points.map((p) => p.util);
    const offset = BUFFER - points.length;

    const line = (series: (number | null)[], color: string, dash: number[]) => {
      ctx.save();
      ctx.setLineDash(dash);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      let open = false;
      series.forEach((v, i) => {
        if (v == null) {
          open = false;
          return;
        }
        const px = x(offset + i);
        const py = y(v);
        if (open) ctx.lineTo(px, py);
        else ctx.moveTo(px, py);
        open = true;
      });
      ctx.stroke();
      ctx.restore();

      // The endpoint marker is the current value, emphasised.
      for (let i = series.length - 1; i >= 0; i--) {
        if (series[i] != null) {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(x(offset + i), y(series[i]!), 2.6, 0, Math.PI * 2);
          ctx.fill();
          break;
        }
      }
    };

    // Area under the primary series, fading into the ground.
    if (vram.some((v) => v != null)) {
      const grad = ctx.createLinearGradient(0, pad, 0, h - pad);
      grad.addColorStop(0, 'rgba(232,195,158,0.16)');
      grad.addColorStop(1, 'rgba(232,195,158,0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(x(offset), h - pad);
      vram.forEach((v, i) => {
        if (v != null) ctx.lineTo(x(offset + i), y(v));
      });
      ctx.lineTo(x(BUFFER - 1), h - pad);
      ctx.closePath();
      ctx.fill();
    }

    line(vram, CHAMP, []);
    line(util, SAGE, [4, 3]);
  }, []);

  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const response = await fetch(`${API}/device`);
        if (!response.ok) throw new Error(String(response.status));
        const data: Sample = await response.json();
        if (cancelled) return;
        setReachable(true);
        setSample(data);
        buffer.current = [
          ...buffer.current,
          {
            vram: data.vram_used_fraction == null ? null : data.vram_used_fraction * 100,
            util: data.utilisation,
          },
        ].slice(-BUFFER);
        draw();
      } catch {
        // One failed poll stops the loop. A panel that keeps asking a dead
        // endpoint once a second fills the console with 404s and tells the
        // reader nothing new; RESUME retries when the API is back.
        if (!cancelled) {
          setReachable(false);
          setRunning(false);
        }
      }
    };

    poll();
    if (!running) return () => {
      cancelled = true;
    };
    const id = window.setInterval(poll, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [running, draw]);

  useEffect(() => {
    const onResize = () => draw();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [draw]);

  const latest = buffer.current[buffer.current.length - 1];
  const vramText =
    latest?.vram == null ? 'unavailable' : `${latest.vram.toFixed(0)}%`;
  const utilInstrumented = sample?.utilisation != null;
  const totalGiB = sample?.vram_total_bytes
    ? `${(sample.vram_total_bytes / 1024 ** 3).toFixed(1)} GB`
    : null;

  return (
    <section className="panel" aria-label="Device telemetry">
      <div className="panel-head">
        <span className="label">
          <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M3 12h4l3 8 4-16 3 8h4" />
          </svg>
          Device telemetry
        </span>
        <span className="spacer" />
        <span className="meta">
          {sample ? `${sample.device}${totalGiB ? ` · ${totalGiB}` : ''} · 1 Hz` : '1 Hz'}
        </span>
        <button
          className="tele-btn"
          type="button"
          aria-pressed={!running}
          onClick={() => setRunning((r) => !r)}
        >
          <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
            {running ? <path d="M7 4v16M17 4v16" /> : <path d="M5 3l14 9-14 9V3z" />}
          </svg>
          {running ? 'PAUSE' : 'RESUME'}
        </button>
      </div>

      {reachable === false ? (
        <p className="cap">
          /device did not answer. Start the API — or restart it if it has been
          running since before this endpoint existed — then press RESUME.
        </p>
      ) : (
        <>
          <canvas
            ref={canvasRef}
            className="tele-canvas"
            role="img"
            aria-label={`VRAM in use over the last ${BUFFER} seconds, currently ${vramText}.`}
          />
          <div className="tele-legend">
            <span className="leg">
              <span className="swatch solid" />
              VRAM in use <b>{vramText}</b>
            </span>
            <span className={`leg${utilInstrumented ? '' : ' off'}`}>
              <span className="swatch dashed" />
              GPU utilisation{' '}
              <b>{utilInstrumented ? `${sample!.utilisation}%` : 'not instrumented'}</b>
            </span>
            <span className="leg" style={{ color: running ? 'var(--sage)' : 'var(--amber)' }}>
              {running ? 'streaming' : 'paused'}
            </span>
          </div>
          {!utilInstrumented && (
            <p className="cap">
              Utilisation needs NVML (pip install pynvml). Until it is there the
              series is absent rather than estimated.
            </p>
          )}
          <p className="sr" aria-live="polite">
            {running
              ? `Telemetry streaming. VRAM in use ${vramText}.`
              : `Telemetry paused at VRAM in use ${vramText}.`}
          </p>
        </>
      )}
    </section>
  );
}
