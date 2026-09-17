'use client';

/**
 * The query composer.
 *
 * Structure follows the 21st.dev "Input Bar" pattern — one bounded composer,
 * a collapsing attachment row above the text, and a toolbar below with attach
 * on the left and send on the right — rebuilt on this project's own tokens
 * rather than installed, because the app has no shadcn/Radix layer and adding
 * one for a single control would be a bigger change than the control.
 *
 * It is deliberately the one bordered, filled object on the page. Everything
 * else is set with rules; this is the thing you are meant to touch, so it gets
 * the affordance and nothing else competes for it.
 *
 * What it fixes about the field it replaces:
 *
 * * The old "choose one or two scenes" was a mono label that looked like a
 *   caption. Attaching is now a button, and the composer is a real drop
 *   target — the previous dashed outline promised a drop that never existed.
 * * Files are listed with their sizes and can be removed one at a time.
 *   Picking again adds to the set instead of silently replacing it.
 * * A file over the API's per-file limit is refused here, with the number, in
 *   the composer — rather than after a 300 MB upload comes back a 413.
 * * "Scenes", not "files", because the API groups them: a Cartosat-2S MX
 *   product is four BAND*.tif plus a sidecar and counts as ONE scene.
 *   Counting files is what made the target sensor's own product format
 *   unuploadable (limitation L17), so the interface counts the way the
 *   backend does.
 */

import dynamic from 'next/dynamic';
import { useCallback, useEffect, useRef, useState } from 'react';

import { QUERY_FIELD_ID } from '../lib/focusQuery';
import type { Bounds } from '../lib/footprint';
import type { ProbedScene } from './AreaPicker';

const API = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

// OpenLayers has no business in the first paint of a page whose main control
// is a text field.
const AreaPicker = dynamic(() => import('./AreaPicker'), { ssr: false });

export const ACCEPTED = '.tif,.tiff,.img,.jp2,.png,.jpg,.jpeg';

// Mirrors satquery/api/main.py: MAX_IMAGES, MAX_UPLOAD_FILES,
// MAX_UPLOAD_BYTES and MAX_TOTAL_UPLOAD_BYTES. Checking here is a courtesy —
// the server still enforces them, and it is the one that decides.
const MAX_FILES = 32;
const MAX_FILE_BYTES = 256 * 1024 * 1024;
const MAX_TOTAL_BYTES = 1024 * 1024 * 1024;

// Above this, the scene is not uploaded for a preview until asked.
const AUTO_PROBE_BYTES = 64 * 1024 * 1024;

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

/** Two files are the same upload if the name, size and mtime all match. */
function sameFile(a: File, b: File): boolean {
  return a.name === b.name && a.size === b.size && a.lastModified === b.lastModified;
}

export default function QueryComposer({
  query,
  onQueryChange,
  files,
  onFilesChange,
  running,
  onSubmit,
  error,
  aoi,
  onAoiChange,
}: {
  query: string;
  onQueryChange: (value: string) => void;
  files: File[];
  onFilesChange: (files: File[]) => void;
  running: boolean;
  onSubmit: () => void;
  error: string;
  aoi: Bounds | null;
  onAoiChange: (box: Bounds | null) => void;
}) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [reject, setReject] = useState('');
  // Drag events fire for every child element, so a boolean flag flickers as
  // the pointer crosses the chips. A depth counter does not.
  const dragDepth = useRef(0);
  const [probe, setProbe] = useState<ProbedScene[] | null>(null);
  const [probing, setProbing] = useState(false);
  const [probeError, setProbeError] = useState('');

  /**
   * Ask the server where the attached scenes are, and for a picture of them.
   *
   * Runs on attach rather than behind a button: the scene should appear
   * because you attached it, not because you found a control. The footprint is
   * only knowable once something has opened the file, and the server is the
   * authority on that — it is the one that will refuse the run if the scene
   * turns out not to be georeferenced.
   *
   * Held back above AUTO_PROBE_BYTES. Uploading a quarter-gigabyte scene the
   * moment it is attached, for a preview, is not a favour; past that size the
   * button asks first.
   */
  const runProbe = useCallback(async (targets: File[]) => {
    if (targets.length === 0) return;
    setProbing(true);
    setProbeError('');
    try {
      const form = new FormData();
      targets.forEach((f) => form.append('images', f));
      const res = await fetch(`${API}/probe`, { method: 'POST', body: form });
      if (!res.ok) {
        const detail = await res.json().catch(() => null);
        throw new Error(detail?.detail ?? `server returned ${res.status}`);
      }
      const data = await res.json();
      setProbe(data.scenes ?? []);
    } catch (err: any) {
      setProbeError(err?.message ?? String(err));
    } finally {
      setProbing(false);
    }
  }, []);

  /* An area belongs to the scenes it was drawn over. Change the attachments
     and it is no longer meaningful, so it goes with them — and the new set is
     located straight away. */
  useEffect(() => {
    onAoiChange(null);
    setProbe(null);
    setProbeError('');
    if (files.length === 0) return;
    const total = files.reduce((sum, f) => sum + f.size, 0);
    if (total <= AUTO_PROBE_BYTES) runProbe(files);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [files]);

  // Grow with the question, up to a ceiling, then scroll.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = '0';
    const next = Math.min(el.scrollHeight, 160);
    el.style.height = `${next}px`;
    el.style.overflowY = el.scrollHeight > 160 ? 'auto' : 'hidden';
  }, [query]);

  const addFiles = useCallback(
    (incoming: FileList | File[] | null) => {
      if (!incoming) return;
      const next = [...files];
      const refused: string[] = [];

      for (const file of Array.from(incoming)) {
        if (next.some((existing) => sameFile(existing, file))) continue;
        if (file.size > MAX_FILE_BYTES) {
          refused.push(
            `${file.name} is ${formatBytes(file.size)} — the limit is ${formatBytes(MAX_FILE_BYTES)} per file`,
          );
          continue;
        }
        next.push(file);
      }

      if (next.length > MAX_FILES) {
        refused.push(`at most ${MAX_FILES} files per run`);
        next.length = MAX_FILES;
      }
      const total = next.reduce((sum, f) => sum + f.size, 0);
      if (total > MAX_TOTAL_BYTES) {
        refused.push(
          `${formatBytes(total)} in total — the limit is ${formatBytes(MAX_TOTAL_BYTES)}`,
        );
      }

      setReject(refused.join('. '));
      onFilesChange(next);
    },
    [files, onFilesChange],
  );

  const removeAt = useCallback(
    (index: number) => {
      setReject('');
      onFilesChange(files.filter((_, i) => i !== index));
    },
    [files, onFilesChange],
  );

  const submit = useCallback(() => {
    if (running) return;
    onSubmit();
  }, [running, onSubmit]);

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter runs, Shift+Enter writes a newline — the convention for a
    // composer. The Run button stays, so the shortcut is never the only way.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const total = files.reduce((sum, f) => sum + f.size, 0);
  const notice = reject || error;

  return (
    <form
      className={`composer${dragging ? ' dragging' : ''}`}
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
      onDragEnter={(e) => {
        e.preventDefault();
        dragDepth.current += 1;
        setDragging(true);
      }}
      onDragOver={(e) => e.preventDefault()}
      onDragLeave={(e) => {
        e.preventDefault();
        dragDepth.current -= 1;
        if (dragDepth.current <= 0) setDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        addFiles(e.dataTransfer.files);
      }}
    >
      <div className="composer-head">
        <span className="label">Query</span>
        <span className="spacer" />
        <span className="meta">POST /runs/stream</span>
      </div>

      <div
        className="composer-box"
        onClick={(e) => {
          if (!(e.target as HTMLElement).closest('button, textarea')) {
            textareaRef.current?.focus();
          }
        }}
      >
        {/* The attachment rail collapses to nothing when there is nothing on
            it, so an empty composer is one clean field rather than a field
            with a permanently reserved gap above it. */}
        <div className={`composer-rail${files.length || aoi ? ' open' : ''}`}>
          <div className="composer-rail-inner">
            <ul className="attachments">
              {aoi && (
                <li className="attachment aoi">
                  <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M4 4h16v16H4z" />
                    <path d="M9 9h6v6H9z" />
                  </svg>
                  <span className="attachment-name">
                    area · {aoi.map((v) => v.toFixed(3)).join(', ')}
                  </span>
                  <button
                    type="button"
                    className="attachment-remove"
                    onClick={() => onAoiChange(null)}
                    aria-label="Remove the selected area"
                  >
                    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M6 6l12 12M18 6L6 18" />
                    </svg>
                  </button>
                </li>
              )}
              {files.map((file, i) => (
                <li className="attachment" key={`${file.name}-${file.lastModified}-${i}`}>
                  <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M4 5h7l2 3h7v11H4z" />
                  </svg>
                  <span className="attachment-name" title={file.name}>
                    {file.name}
                  </span>
                  <span className="attachment-size">{formatBytes(file.size)}</span>
                  <button
                    type="button"
                    className="attachment-remove"
                    onClick={() => removeAt(i)}
                    aria-label={`Remove ${file.name}`}
                  >
                    <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                      <path d="M6 6l12 12M18 6L6 18" />
                    </svg>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <label className="sr" htmlFor={QUERY_FIELD_ID}>
          Your question about the imagery
        </label>
        <textarea
          id={QUERY_FIELD_ID}
          ref={textareaRef}
          className="composer-input"
          rows={1}
          value={query}
          disabled={running}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="What changed between these two scenes?"
        />

        <div className="composer-tools">
          {/* Attach and its hint are one group, the keyboard hint and Run are
              another, so a narrow viewport wraps them as two blocks instead of
              stranding the Run button on a line of its own. */}
          <div className="composer-group">
            <button
              type="button"
              className="composer-attach"
              onClick={() => inputRef.current?.click()}
              disabled={running}
            >
              <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                <path d="M21.4 11.1l-9.2 9.2a6 6 0 01-8.5-8.5l9.2-9.2a4 4 0 015.7 5.7l-9.2 9.2a2 2 0 01-2.8-2.9l8.5-8.4" />
              </svg>
              {files.length ? 'Add more' : 'Attach imagery'}
            </button>

            {files.length > 0 && !probe && (
              <button
                type="button"
                className="composer-attach"
                onClick={() => runProbe(files)}
                disabled={running || probing}
                title="Upload the scenes to locate them and show a preview"
              >
                <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M3 6l6-3 6 3 6-3v15l-6 3-6-3-6 3z" />
                  <path d="M9 3v15M15 6v15" />
                </svg>
                {probing ? 'Locating…' : 'Show on map'}
              </button>
            )}

            <span className="composer-hint">
              {files.length
                ? `${files.length} file${files.length === 1 ? '' : 's'} · ${formatBytes(total)}`
                : 'or drop files here · one or two scenes'}
            </span>
          </div>

          <div className="composer-group right">
            <kbd className="composer-kbd" title="Press Enter to run">
              ↵
            </kbd>
            <button className="btn btn-solid" type="submit" disabled={running}>
            {running ? (
              <>
                <span className="composer-spinner" aria-hidden="true" />
                Running…
              </>
            ) : (
              <>
                <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M5 3l14 9-14 9V3z" />
                </svg>
                Run
              </>
            )}
            </button>
          </div>
        </div>

        <input
          ref={inputRef}
          className="sr"
          type="file"
          accept={ACCEPTED}
          multiple
          onChange={(e) => {
            addFiles(e.target.files);
            // Reset, so picking the same file twice in a row still fires.
            e.target.value = '';
          }}
        />

        {dragging && (
          <div className="composer-drop" aria-hidden="true">
            Drop imagery to attach
          </div>
        )}
      </div>

      <div className="composer-foot">
        <span className="composer-formats">{ACCEPTED.replace(/,/g, '  ')}</span>
        <span className="spacer" />
        <span className="composer-note">
          A multi-band vendor product counts as one scene.
        </span>
      </div>

      {(notice || probeError) && (
        <p className="composer-error" role="alert">
          {probeError || notice}
        </p>
      )}

      {probe && probe.length > 0 && (
        <AreaPicker scenes={probe} value={aoi} onChange={onAoiChange} />
      )}
    </form>
  );
}
