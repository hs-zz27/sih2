'use client';

/**
 * Site navigation.
 *
 * Four routes existed - the query page, /models, /benchmarks and
 * /runs/[runId] - and nothing linked to any of them. The model registry and
 * the benchmark page were reachable only by typing the URL, which for a demo
 * means a judge cannot get to the evidence pages unless someone drives.
 *
 * `usePathname` marks the current route so the header says where you are;
 * `prefetch` is left at the Next default. /runs/[runId] is deliberately
 * absent: it needs an id, and the query page links to it once a run exists.
 *
 * The device chip reads `/device` once on mount. It shows what the API
 * process actually reports - free VRAM on the active CUDA device, or "CPU"
 * when there is none - rather than a decorative status light that is green
 * whether or not anything is running.
 */

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';

import { focusQuery } from './lib/focusQuery';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

const ROUTES = [
  { href: '/', label: 'Query' },
  { href: '/models', label: 'Models' },
  { href: '/benchmarks', label: 'Benchmarks' },
];

type Device = {
  device: string;
  name: string | null;
  vram_free_bytes: number | null;
  vram_total_bytes: number | null;
};

function gib(bytes: number | null): string | null {
  if (bytes == null) return null;
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

export default function Nav() {
  const pathname = usePathname();
  const [device, setDevice] = useState<Device | null>(null);
  const navRef = useRef<HTMLElement>(null);

  /**
   * Publish the header's height as `--nav-h`.
   *
   * Anything that scrolls itself into view has to clear the sticky header,
   * and the header is not a fixed height: it wraps to two rows on a narrow
   * viewport, and grows again when the device chip arrives from `/device`.
   * Measuring beats guessing — a hardcoded offset tuned to the one-row case
   * parks the first line of the target underneath the second row.
   */
  useEffect(() => {
    const el = navRef.current;
    if (!el) return;
    const publish = () =>
      document.documentElement.style.setProperty(
        '--nav-h',
        `${Math.round(el.getBoundingClientRect().height)}px`,
      );
    publish();
    const observer = new ResizeObserver(publish);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetch(`${API}/device`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d) => {
        if (!cancelled) setDevice(d);
      })
      // A missing device reading is not worth an error state in the header.
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, []);

  const free = gib(device?.vram_free_bytes ?? null);
  const chip = !device
    ? null
    : free
      ? `${device.device.toUpperCase()} · ${free} FREE`
      : device.device.toUpperCase();

  return (
    <nav className="nav" aria-label="Main" ref={navRef}>
      <Link className="nav-brand" href="/">
        SatQuery&nbsp;AI
      </Link>
      <ul>
        {ROUTES.map((route) => {
          const active =
            route.href === '/' ? pathname === '/' : pathname?.startsWith(route.href);
          return (
            <li key={route.href}>
              <Link
                href={route.href}
                className={`link${active ? ' active' : ''}`}
                aria-current={active ? 'page' : undefined}
                onClick={(event) => {
                  // Already on the query page: the useful thing is not a
                  // route change to where you already are, it is getting to
                  // the composer. Any other route still navigates normally.
                  if (route.href === '/' && pathname === '/') {
                    event.preventDefault();
                    focusQuery();
                  }
                }}
              >
                {route.label}
              </Link>
            </li>
          );
        })}
      </ul>
      {chip && (
        <div className="right">
          <span className="device" title={device?.name ?? undefined}>
            <span className={`dot${device?.device === 'cpu' ? ' idle' : ''}`} />
            {chip}
          </span>
        </div>
      )}
    </nav>
  );
}
