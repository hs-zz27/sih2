"""FastAPI application with SSE trace streaming (plan task 1.5).

Endpoints:
    GET  /health              liveness
    POST /runs                run the pipeline, return the full trace
    POST /runs/stream         run the pipeline, stream the trace live as SSE
    GET  /runs                recent runs
    GET  /runs/{run_id}       one stored run including its trace

The streaming endpoint runs the pipeline on a worker thread and drains its
events through a queue, so the client sees ingest, routing and each tool step
as they happen rather than waiting for the whole run.
"""

from __future__ import annotations

import json
import queue
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Iterator

import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from satquery.contracts.trace import Trace
from satquery.controller.pipeline import Controller
from satquery.api.store import RunStore
# Product filename patterns, so the upload path groups a vendor product the
# same way `discover` assembles it rather than keeping a second copy of the rule.
from satquery.ingest.product import _BAND_FILE_RE, _POL_FILE_RE

app = FastAPI(title="SatQuery AI API", version="0.2.0-phase1")

# The frontend is served from a different origin (:3000) than the API (:8000),
# so the browser blocks fetch() without an explicit CORS allowance. Origins are
# configurable and default to local development only - never "*", which would
# also have to disable credentials and is wrong for a deployed system.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "SATQUERY_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-Extent", "X-Projection", "X-Overlay-Rendering"],
)

# Built once: the controller fits the intent classifier on construction, and
# refitting per request would dominate latency.
_controller: Controller | None = None
_store: RunStore | None = None
_lock = threading.Lock()

# Logical *images*, not files. A vendor product ships one file per band - a
# Cartosat-2S MX scene is four BAND*.tif plus BAND_META.txt - and all five
# together are one image. Counting files here rejected the target sensor's own
# product format with "expected 1 to 2 images, got 4" (limitation L17), so the
# upload path groups files into products first and applies this limit to the
# groups.
MAX_IMAGES = 2

# Files per request. Bounds the request independently of the image count: a
# ScanSAR product legitimately carries several polarisation rasters plus
# sidecars, and an unbounded count is a denial-of-service surface however
# small each file is.
MAX_UPLOAD_FILES = int(os.getenv("SATQUERY_MAX_UPLOAD_FILES", 32))

# Per-file upload cap. Without one, `shutil.copyfileobj` writes whatever the
# client sends, so a single request can fill the disk - and neither Starlette
# nor uvicorn imposes a default. 256 MB comfortably clears a full Cartosat-2E
# scene (7687x7640 px, 4 bands, uint16 ~ 470 MB uncompressed but far less as a
# compressed GeoTIFF) while bounding the damage.
MAX_UPLOAD_BYTES = int(os.getenv("SATQUERY_MAX_UPLOAD_BYTES", 256 * 1024 * 1024))

# Total across the request. The per-file cap alone bounds one file at 256 MB;
# with a raised file count that is 8 GB per request, so the request needs its
# own ceiling.
MAX_TOTAL_UPLOAD_BYTES = int(
    os.getenv("SATQUERY_MAX_TOTAL_UPLOAD_BYTES", 1024 * 1024 * 1024)
)
UPLOAD_CHUNK = 1024 * 1024

# How many completed run directories to keep. Each holds the uploaded rasters
# and the artifacts the run produced, and nothing deleted them: a demo session
# grew the temp directory without bound and kept user-supplied imagery on disk
# indefinitely. They cannot be deleted immediately - /runs/{id}/preview and
# /runs/{id}/report.pdf both read from them - so the fix is bounded retention,
# oldest evicted first.
MAX_RETAINED_RUNS = int(os.getenv("SATQUERY_MAX_RETAINED_RUNS", 20))

# Overlay colours for categorical rasters, class 1 upward. Chosen to stay
# legible over an OSM basemap, whose greens, greys and blues would swallow a
# muted palette. Class 0 is background and is never drawn.
_MASK_PALETTE = (
    (220, 38, 38),    # 1 - red, the conventional "changed" colour
    (37, 99, 235),    # 2 - blue
    (217, 119, 6),    # 3 - amber
    (147, 51, 234),   # 4 - purple
    (5, 150, 105),    # 5 - green
    (219, 39, 119),   # 6 - pink
    (14, 165, 233),   # 7 - cyan
)
# Slightly translucent so the basemap underneath stays readable for
# orientation, while the changed pixels remain unmistakable.
MASK_ALPHA = 235
# Sentinel that closes the SSE stream from the worker thread.
_DONE = object()


def get_controller() -> Controller:
    global _controller
    if _controller is None:
        with _lock:
            if _controller is None:
                _controller = Controller()
    return _controller


def get_store() -> RunStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = RunStore()
    return _store


def _safe_segment(segment: str) -> str:
    """One path segment, stripped of anything that could escape a directory.

    Client-supplied names are never trusted: `Path(...).name` discards any
    directory part, and the result is rejected if it is empty or a relative
    marker, so "../../etc/passwd" cannot become a path component.
    """
    cleaned = Path(segment.replace("\\", "/")).name.strip()
    return "" if cleaned in ("", ".", "..") else cleaned


def group_uploads(filenames: list[str]) -> list[list[int]]:
    """Group uploaded filenames into logical images, returning their indices.

    A vendor product is one image spread over several files, so the request's
    file count is not its image count. Two grouping signals, in order:

    1. **A client-supplied directory.** A browser directory picker
       (`webkitdirectory`) sends names like `5132611/BAND1.tif`. The first
       segment groups the files, which is the only signal that can separate
       *two* products in one request.
    2. **A recognised vendor filename.** Flat uploads carry no directory, so
       if any file matches a known product layout - `BAND<n>.tif` for Cartosat
       MX, `imagery_<POL>.tif` for EOS-04 - the whole upload is taken as one
       product. Sidecars like `BAND_META.txt` travel with it, which matters
       because `discover` reads them for the acquisition date and bit depth.

    Otherwise each file is its own image, which is the ordinary case of
    uploading one or two GeoTIFFs.

    **Known boundary:** two flat-uploaded vendor products cannot be told
    apart - both are `BAND1.tif`, `BAND2.tif`, ... with no directory to
    separate them. Upload them through a directory picker, or use the CLI.
    """
    if not filenames:
        return []

    directories = [
        Path(name.replace("\\", "/")).parent.as_posix() for name in filenames
    ]
    if any(d not in ("", ".") for d in directories):
        groups: dict[str, list[int]] = {}
        for index, directory in enumerate(directories):
            groups.setdefault(directory, []).append(index)
        return list(groups.values())

    names = [Path(n.replace("\\", "/")).name for n in filenames]
    if any(_BAND_FILE_RE.match(n) or _POL_FILE_RE.search(n) for n in names):
        return [list(range(len(filenames)))]

    return [[index] for index in range(len(filenames))]


def _validate_upload_shape(images: list[UploadFile]) -> None:
    """Reject a malformed request before any bytes are written to disk.

    The image count is the count of *logical images* after grouping, not of
    files: a Cartosat-2S MX product is four BAND*.tif plus a metadata sidecar
    and is one image. Counting files here is what made the target sensor's own
    product format unuploadable (limitation L17).
    """
    if len(images) > MAX_UPLOAD_FILES:
        raise HTTPException(
            400, f"expected at most {MAX_UPLOAD_FILES} files, got {len(images)}"
        )
    groups = group_uploads([f.filename or "" for f in images])
    if not 1 <= len(groups) <= MAX_IMAGES:
        raise HTTPException(
            400,
            f"expected 1 to {MAX_IMAGES} images, got {len(groups)} "
            f"(from {len(images)} file(s))",
        )


# The longest edge of the preview PNG the picker draws over. Big enough to
# choose an area from, small enough to sit in a JSON response.
PROBE_PREVIEW_PX = 512

# A crop smaller than this on either side is not an input. Below roughly a
# tile the model has nothing to look at, and the ingest checks that follow
# would be describing noise.
MIN_AOI_PX = 64


def _parse_aoi(raw: str | None) -> tuple[float, float, float, float] | None:
    """Parse the optional `aoi` form field: [west, south, east, north] in EPSG:4326.

    Rejected here rather than deep in rasterio, so a malformed box comes back
    as a 400 naming the problem instead of a 500 naming a GDAL internal.
    """
    if raw is None or not raw.strip():
        return None
    try:
        box = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"aoi is not valid JSON: {exc.msg}") from exc

    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise HTTPException(400, "aoi must be [west, south, east, north]")
    try:
        west, south, east, north = (float(v) for v in box)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "aoi values must be numbers") from exc

    if not (-180 <= west < east <= 180) or not (-90 <= south < north <= 90):
        raise HTTPException(
            400,
            "aoi must be [west, south, east, north] in degrees, with west < east "
            "and south < north",
        )
    return (west, south, east, north)


def _crop_to_aoi(path: Path, aoi: tuple[float, float, float, float]) -> Path:
    """Write the part of `path` inside `aoi`, and return the new file.

    A windowed read, so the crop keeps the source resolution, dtype, band
    count and nodata: only the extent changes. That matters because every
    check downstream - GSD, radiometry, overlap - then describes the pixels
    the model was actually given.

    The original is left on disk. The trace records both extents, so a run on
    a crop can never be mistaken for a run on the whole scene.
    """
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import Window, from_bounds, intersection

    from satquery.ingest.reader import wgs84_bounds

    if path.is_dir():
        # A multi-file vendor product is several rasters plus the metadata
        # that describes them; cropping the bands without rewriting the
        # sidecars would leave a product whose own header disagrees with its
        # pixels. Refused rather than half-done.
        raise HTTPException(
            400,
            "area selection is not supported for multi-file products yet - "
            "upload a single raster, or run the whole scene",
        )

    with rasterio.open(path) as src:
        if src.crs is None:
            raise HTTPException(
                400,
                f"{path.name} carries no CRS, so an area on the map cannot be "
                "located in it",
            )

        left, bottom, right, top = transform_bounds(
            "EPSG:4326", src.crs, *aoi, densify_pts=21
        )
        try:
            window = intersection(
                from_bounds(left, bottom, right, top, src.transform),
                Window(0, 0, src.width, src.height),
            )
        except Exception as exc:  # noqa: BLE001 - rasterio raises when disjoint
            raise HTTPException(
                400,
                f"the selected area does not overlap {path.name}",
            ) from exc

        window = window.round_lengths().round_offsets()
        if window.width < MIN_AOI_PX or window.height < MIN_AOI_PX:
            raise HTTPException(
                400,
                f"the selected area covers only {int(window.width)}x"
                f"{int(window.height)} px of {path.name}; at least "
                f"{MIN_AOI_PX}x{MIN_AOI_PX} is needed",
            )

        source_bounds = wgs84_bounds(src)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            height=int(window.height),
            width=int(window.width),
            transform=src.window_transform(window),
        )
        data = src.read(window=window)

    target = path.with_name(f"{path.stem}_aoi.tif")
    with rasterio.open(target, "w", **profile) as dst:
        dst.write(data)
        # The provenance goes in the file, not in a variable that has to be
        # carried through three layers to reach the trace. Ingest reads these
        # back into the manifest, so a crop can never be presented as though
        # the whole scene had been examined.
        dst.update_tags(
            SATQUERY_AOI=json.dumps([round(v, 6) for v in aoi]),
            SATQUERY_SOURCE_BOUNDS=json.dumps(
                [round(v, 6) for v in source_bounds] if source_bounds else None
            ),
            SATQUERY_SOURCE_FILE=path.name,
        )
    return target


def _apply_aoi(
    paths: list[Path], aoi: tuple[float, float, float, float] | None
) -> list[Path]:
    """Crop every scene to the AOI, or return the paths untouched.

    All scenes are cropped to the *same* box, which is the point for a
    bi-temporal pair: it guarantees the two cover the same ground, which is
    the condition the overlap check exists to test.
    """
    if aoi is None:
        return paths
    return [_crop_to_aoi(path, aoi) for path in paths]


def _save_uploads(files: list[UploadFile], dest: Path) -> list[Path]:
    """Persist uploads under `dest`, grouped into logical images.

    Returns one path per image: a **file** when the image is a single raster,
    and a **directory** when it is a multi-file vendor product, which is what
    `satquery.ingest.discover` expects in order to assemble the bands.
    """
    dest.mkdir(parents=True, exist_ok=True)
    groups = group_uploads([f.filename or "" for f in files])
    total_written = 0
    paths: list[Path] = []

    for group_index, indices in enumerate(groups):
        multi = len(indices) > 1
        group_dir = dest / f"image_{group_index:02d}" if multi else dest
        group_dir.mkdir(parents=True, exist_ok=True)

        for i in indices:
            upload = files[i]
            # Never trust a client-supplied filename for a path: take the
            # basename only, so "../../etc/passwd" cannot escape the upload
            # directory.
            safe = _safe_segment(upload.filename or "") or f"image_{i}"
            # A multi-file product must keep its real names - `discover`
            # matches BAND1.tif and imagery_HH.tif by pattern - so only the
            # single-file case gets an index prefix to avoid collisions.
            target = group_dir / (safe if multi else f"{i:02d}_{safe}")
            written = 0
            # Copied in chunks with a running total rather than in one call:
            # the point is to stop BEFORE the disk fills, so the limit has to
            # be enforced during the write, not checked afterwards.
            with target.open("wb") as fh:
                while chunk := upload.file.read(UPLOAD_CHUNK):
                    written += len(chunk)
                    total_written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        fh.close()
                        shutil.rmtree(dest, ignore_errors=True)
                        raise HTTPException(
                            413,
                            f"{safe} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} "
                            f"MB per-file upload limit",
                        )
                    if total_written > MAX_TOTAL_UPLOAD_BYTES:
                        fh.close()
                        shutil.rmtree(dest, ignore_errors=True)
                        raise HTTPException(
                            413,
                            f"request exceeds the "
                            f"{MAX_TOTAL_UPLOAD_BYTES // (1024 * 1024)} MB total "
                            f"upload limit",
                        )
                    fh.write(chunk)

        paths.append(group_dir if multi else group_dir / f"{indices[0]:02d}_"
                     f"{_safe_segment(files[indices[0]].filename or '') or f'image_{indices[0]}'}")

    return paths


def _prune_run_dirs(keep: int = MAX_RETAINED_RUNS) -> None:
    """Delete the oldest run directories beyond `keep`.

    Best-effort and never fatal: failing to reclaim disk must not fail the run
    that just succeeded. A directory still being read by an in-flight preview
    request will refuse to delete on Windows, which is the correct outcome -
    it is simply retried on the next run.
    """
    root = Path(tempfile.gettempdir())
    try:
        dirs = sorted(
            # Both prefixes: /runs creates satquery_run_*, and
            # /runs/{id}/report.pdf creates satquery_report_*. The second was
            # leaking too.
            (
                p for p in root.iterdir()
                if p.is_dir() and p.name.startswith(("satquery_run_", "satquery_report_"))
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return
    for stale in dirs[keep:]:
        shutil.rmtree(stale, ignore_errors=True)

    # The uploads were bounded here; the index rasters the run wrote into
    # `artifacts/<run_id>` were not, and they are the larger half - ~526 MB
    # for a full scene. Same retention count, same "named directories are
    # never touched" rule; see satquery/controller/retention.py.
    from satquery.controller.retention import auto_prune

    auto_prune(keep=keep)


def _sse(event: str, data: dict | list) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _client_safe_error(exc: Exception) -> str:
    """A message safe to send to a client.

    Raw exception text from rasterio/GDAL embeds absolute server paths, which
    discloses the filesystem layout to anyone who can upload a bad file. The
    full detail is still recorded in the run store for operators; only the
    outward-facing string is reduced.
    """
    name = type(exc).__name__
    if isinstance(exc, (ValueError, KeyError)):
        # These carry our own validation messages, which are safe and useful.
        return str(exc).replace(str(Path.cwd()), ".")
    return f"the input could not be processed ({name})"


@app.get("/")
def root():
    return {
        "service": "SatQuery AI API",
        "status": "online",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "web_ui": "http://localhost:3000",
    }


@app.get("/health")
def health_check():
    return {"status": "ok", "version": app.version}


@app.post("/runs")
async def create_run(
    query: str = Form(...),
    images: list[UploadFile] = File(...),
    aoi: str | None = Form(None),
):
    """Run the pipeline synchronously and return the complete trace."""
    _validate_upload_shape(images)
    area = _parse_aoi(aoi)

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    work_dir = Path(tempfile.mkdtemp(prefix=f"satquery_{run_id}_"))
    store = get_store()
    store.create(run_id, query)

    try:
        paths = _apply_aoi(_save_uploads(images, work_dir), area)
        trace = get_controller().run(paths, query, run_id=run_id)
    except HTTPException as exc:
        # A deliberate status - 413 for an oversized upload - must survive.
        # The blanket handler below was converting it to a 500, which told the
        # client "server error" for what is squarely a client error.
        store.fail(run_id, exc.detail)
        raise
    except Exception as exc:  # noqa: BLE001 - reported, not swallowed
        store.fail(run_id, str(exc))  # full detail retained server-side
        raise HTTPException(500, _client_safe_error(exc)) from exc

    store.complete(run_id, trace)
    _prune_run_dirs()
    return json.loads(trace.model_dump_json())


@app.post("/runs/stream")
async def stream_run(
    query: str = Form(...),
    images: list[UploadFile] = File(...),
    aoi: str | None = Form(None),
):
    """Run the pipeline and stream trace stages as server-sent events."""
    _validate_upload_shape(images)
    area = _parse_aoi(aoi)

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    work_dir = Path(tempfile.mkdtemp(prefix=f"satquery_{run_id}_"))
    paths = _apply_aoi(_save_uploads(images, work_dir), area)

    store = get_store()
    store.create(run_id, query)
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            from satquery.ingest import ingest as ingest_fn

            controller = get_controller()
            manifest = ingest_fn(paths, run_id=run_id)
            # Through the controller, not around it. Reaching into
            # router/executor here duplicated the pipeline and the copies
            # drifted - config_excluded was never passed, so the streamed
            # answer silently lost the task 3.8 exclusion notice.
            trace = controller.run_on_manifest(
                manifest,
                query,
                on_event=lambda name, data: events.put((name, data)),
            )
            store.complete(run_id, trace)
            events.put(("complete", json.loads(trace.model_dump_json())))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            store.fail(run_id, str(exc))  # full detail retained server-side
            events.put(
                ("error", {"run_id": run_id, "message": _client_safe_error(exc)})
            )
        finally:
            events.put(_DONE)
            _prune_run_dirs()

    threading.Thread(target=worker, daemon=True).start()

    def generate() -> Iterator[str]:
        yield _sse("run_started", {"run_id": run_id, "query": query})
        while True:
            item = events.get()
            if item is _DONE:
                break
            name, data = item
            yield _sse(name, data)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/runs/{run_id}/preview/{role}")
def get_preview(run_id: str, role: str, max_edge: int = 512):
    """Render one of a run's input images as a PNG.

    Browsers cannot display GeoTIFF, and the bi-temporal swipe and
    optical-SAR blend comparators (task 2.12) need something they can draw.
    This reuses the same `to_rgb_preview` the VQA tool feeds the model, so
    what the user compares is what the model saw - band selection and
    stretch included, rather than a separately-tuned display rendering that
    could look better or worse than the actual input.
    """
    from satquery.ingest.reader import read_image
    from satquery.tools.imaging import to_rgb_preview

    record = get_store().get(run_id)
    if record is None or not record.get("trace"):
        raise HTTPException(404, f"no such run: {run_id}")

    images = record["trace"]["ingest"]["images"]
    match = next((i for i in images if i.get("role") == role), None)
    if match is None:
        raise HTTPException(
            404,
            f"no image with role {role!r}; available: "
            f"{[i.get('role') for i in images]}",
        )

    path = Path(match["path"])
    if not path.exists():
        raise HTTPException(410, "the source image is no longer on disk")

    try:
        image, _ = to_rgb_preview(read_image(path), max_edge=max(64, min(max_edge, 2048)))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, _client_safe_error(exc)) from exc

    import io

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/runs/{run_id}/overlay/{key}")
def get_overlay(run_id: str, key: str, max_edge: int = 1024):
    """A run artifact as a web-map overlay (plan task 1.6).

    Returns a PNG **already reprojected to EPSG:3857** together with its
    extent in the `X-Extent` header. Reprojecting server-side is what lets the
    frontend place the layer exactly without carrying proj4 and a projection
    registry for every UTM zone an Indian scene might land in - the Cartosat
    sample alone is zone 45N.

    The alpha channel encodes nodata, so a mask overlays the basemap instead
    of covering it with a black rectangle.
    """
    import io

    import numpy as np
    import rasterio
    from PIL import Image
    from rasterio.warp import transform_bounds
    from rasterio.vrt import WarpedVRT

    record = get_store().get(run_id)
    if record is None or not record.get("trace"):
        raise HTTPException(404, f"no such run: {run_id}")

    trace = record["trace"]
    paths = trace.get("artifact_paths") or {}
    if key not in paths:
        raise HTTPException(
            404, f"no artifact {key!r}; available: {sorted(paths)}"
        )
    path = Path(paths[key])
    if not path.exists():
        raise HTTPException(410, "the artifact is no longer on disk")

    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                raise HTTPException(
                    422, f"{key} has no CRS and cannot be placed on a map"
                )
            with WarpedVRT(src, crs="EPSG:3857") as vrt:
                scale = min(1.0, max_edge / max(vrt.width, vrt.height))
                width = max(1, int(vrt.width * scale))
                height = max(1, int(vrt.height * scale))
                count = min(vrt.count, 3)
                data = vrt.read(
                    list(range(1, count + 1)),
                    out_shape=(count, height, width),
                    masked=True,
                ).astype("float32")
                bounds = transform_bounds(
                    src.crs, "EPSG:3857", *src.bounds, densify_pts=21
                )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, _client_safe_error(exc)) from exc

    valid = np.ma.getmaskarray(data)
    finite = data.compressed() if hasattr(data, "compressed") else data.ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise HTTPException(422, f"{key} contains no valid pixels")

    # A binary or small-categorical raster is a *mask*, and a mask must not be
    # rendered as a photograph. Stretching one to grayscale gave alpha=255 and
    # RGB (0,0,0) on every unchanged pixel - a black rectangle laid over the
    # basemap (limitation L19). The alpha channel was already carrying nodata,
    # but "unchanged" is data, not nodata, so it painted.
    #
    # Categorical rendering keeps the evidence and drops only the background:
    # class 0 becomes transparent, every asserted class keeps full opacity in
    # a distinct colour. A mask with real change is *more* legible than
    # before, not less - the changed pixels now sit in colour against the map
    # instead of inside a black square.
    distinct = np.unique(finite)
    categorical = (
        np.issubdtype(np.asarray(finite).dtype, np.floating) is False
        or np.allclose(distinct, np.round(distinct))
    ) and distinct.size <= len(_MASK_PALETTE) + 1 and float(distinct.min()) >= 0

    if categorical:
        codes = np.asarray(data.filled(0))[0].round().astype("int32")
        rgb = np.zeros((3, height, width), dtype="uint8")
        for value in distinct:
            code = int(round(float(value)))
            if code <= 0:
                continue
            colour = _MASK_PALETTE[(code - 1) % len(_MASK_PALETTE)]
            selected = codes == code
            for channel in range(3):
                rgb[channel][selected] = colour[channel]
        # Transparent where the class is background OR the pixel is nodata.
        alpha = np.where((codes > 0) & (~valid[0]), MASK_ALPHA, 0).astype("uint8")
        rendering = "categorical"
    else:
        # Percentile stretch, matching the preview endpoint: a single hot pixel
        # or a nodata sentinel would otherwise flatten the image to black.
        low, high = np.percentile(finite, [2, 98])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((np.asarray(data.filled(low)) - low) / (high - low), 0, 1)
        scaled = (scaled * 255).astype("uint8")

        if scaled.shape[0] == 1:
            rgb = np.repeat(scaled, 3, axis=0)
        else:
            rgb = np.zeros((3, height, width), dtype="uint8")
            rgb[: scaled.shape[0]] = scaled[:3]
        alpha = (~valid[0] * 255).astype("uint8")
        rendering = "continuous"

    rgba = np.concatenate([rgb, alpha[None, ...]], axis=0)
    image = Image.fromarray(np.transpose(rgba, (1, 2, 0)), mode="RGBA")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return Response(
        content=buffer.getvalue(),
        media_type="image/png",
        headers={
            # minx,miny,maxx,maxy in EPSG:3857, which is what ol needs to
            # place a Static image source.
            "X-Extent": ",".join(f"{v:.3f}" for v in bounds),
            "X-Projection": "EPSG:3857",
            # Which of the two rendering paths ran. A client can style a mask
            # differently from an index, and a test can assert that a binary
            # mask did not silently fall back to the grayscale stretch that
            # produced the opaque black rectangle of limitation L19.
            "X-Overlay-Rendering": rendering,
            "Access-Control-Expose-Headers": (
                "X-Extent,X-Projection,X-Overlay-Rendering"
            ),
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.get("/runs/{run_id}/overlays")
def list_overlays(run_id: str):
    """Which artifacts of a run can be drawn on a map."""
    record = get_store().get(run_id)
    if record is None or not record.get("trace"):
        raise HTTPException(404, f"no such run: {run_id}")
    paths = (record["trace"].get("artifact_paths") or {})
    return {
        "run_id": run_id,
        "overlays": [
            {"key": k, "available": Path(v).exists()}
            for k, v in sorted(paths.items())
            if Path(v).suffix.lower() in {".tif", ".tiff"}
        ],
    }


@app.get("/models")
def get_model_registry():
    """Model registry page data (task 3.12).

    Read-only and assembled from what the training runs wrote. Registered
    before /runs/{run_id} would matter only if the paths collided; they do
    not, but the ordering is kept explicit anyway.
    """
    from satquery.report.registry import model_registry

    return model_registry()


@app.get("/benchmarks")
def get_benchmarks():
    """Benchmark page data (task 3.12): every Phase 3 measurement."""
    from satquery.report.registry import benchmarks

    return benchmarks()


@app.get("/runs/{run_id}/report.pdf")
def get_run_report(run_id: str):
    """Render a completed run to a PDF (task 3.12)."""
    from fastapi.responses import FileResponse

    from satquery.report.pdf_report import export_pdf

    record = get_store().get(run_id)
    if record is None or not record.get("trace"):
        raise HTTPException(404, f"no completed run {run_id}")

    trace = Trace.model_validate(
        record["trace"] if isinstance(record["trace"], dict)
        else json.loads(record["trace"])
    )
    out_dir = Path(tempfile.mkdtemp(prefix=f"satquery_report_{run_id}_"))
    try:
        pdf = export_pdf(trace, out_dir / f"{run_id}.pdf")
    except RuntimeError as exc:
        # reportlab is optional; say so rather than returning a 500.
        raise HTTPException(503, str(exc)) from exc
    return FileResponse(
        pdf, media_type="application/pdf", filename=f"{run_id}.pdf"
    )


@app.get("/runs")
def list_runs(limit: int = 50):
    return {"runs": get_store().list(limit=min(max(limit, 1), 500))}


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    record = get_store().get(run_id)
    if record is None:
        raise HTTPException(404, f"no such run: {run_id}")
    return record


@app.post("/probe")
async def probe_uploads(images: list[UploadFile] = File(...)):
    """Where the uploaded scenes are, without running anything.

    The area picker has to draw a box over the scene's footprint, and the
    footprint is only known once something has opened the file. Running the
    whole pipeline to find that out would cost minutes and a stored trace, so
    this reads the header and throws the pixels away.

    Nothing is persisted: no run id, no trace, no artifacts. The temporary
    directory goes with the response.
    """
    _validate_upload_shape(images)

    work_dir = Path(tempfile.mkdtemp(prefix="satquery_probe_"))
    try:
        paths = _save_uploads(images, work_dir)
        import base64
        import io

        import rasterio

        from rasterio.vrt import WarpedVRT
        from rasterio.warp import transform_bounds

        from satquery.ingest.reader import read_image, wgs84_bounds
        from satquery.tools.imaging import to_rgb_preview

        def preview_for(target: Path, src) -> tuple[str | None, list[float] | None]:
            """A small PNG of the scene in EPSG:3857, and the extent to draw it at.

            Reprojected, not just bounded. A UTM scene's footprint is a rotated
            quadrilateral in web-mercator, so dropping its PNG into the
            axis-aligned lon/lat envelope misregisters it against the basemap -
            measured at 164 m on an 8.7 km Landsat scene at 83.4E, where the
            grid convergence is 1.08 degrees. You cannot choose an area off a
            picture that is in the wrong place, so this warps first, the same
            way `/runs/{id}/overlay/{key}` already does.

            Rendered by `to_rgb_preview`, the same path the VQA tool feeds the
            model, so the pixels being selected are the pixels that will be
            read - band choice and stretch included.
            """
            try:
                from affine import Affine

                with WarpedVRT(src, crs="EPSG:3857") as vrt:
                    scale = min(1.0, PROBE_PREVIEW_PX / max(vrt.width, vrt.height))
                    width = max(1, int(vrt.width * scale))
                    height = max(1, int(vrt.height * scale))
                    data = vrt.read(out_shape=(vrt.count, height, width))
                    profile = vrt.profile.copy()
                    profile.update(
                        driver="GTiff",
                        width=width,
                        height=height,
                        transform=vrt.transform
                        * Affine.scale(vrt.width / width, vrt.height / height),
                    )

                warped = work_dir / f"{target.stem}_3857.tif"
                with rasterio.open(warped, "w", **profile) as dst:
                    dst.write(data)
                    # Band names decide which bands `to_rgb_preview` picks, and
                    # the VRT does not carry them across on its own.
                    if any(src.descriptions):
                        dst.descriptions = src.descriptions
                    dst.update_tags(**src.tags())

                image, _ = to_rgb_preview(read_image(warped), max_edge=PROBE_PREVIEW_PX)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                extent = list(
                    transform_bounds(src.crs, "EPSG:3857", *src.bounds, densify_pts=21)
                )
            except Exception:  # noqa: BLE001 - a scene that will not render is
                # still worth reporting the bounds for; the picker falls back
                # to the outline alone.
                return None, None
            return (
                "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
                extent,
            )

        scenes = []
        for path in paths:
            # A multi-file product is a directory; probe its first raster,
            # which is enough to place the product on a map.
            target = path
            if path.is_dir():
                rasters = sorted(
                    p
                    for p in path.iterdir()
                    if p.suffix.lower() in {".tif", ".tiff", ".jp2", ".img"}
                )
                if not rasters:
                    scenes.append({"name": path.name, "georeferenced": False})
                    continue
                target = rasters[0]

            try:
                with rasterio.open(target) as src:
                    located = src.crs is not None
                    preview, preview_extent = (
                        preview_for(target, src) if located else (None, None)
                    )
                    scenes.append({
                        "name": path.name,
                        "crs": str(src.crs) if src.crs else "UNKNOWN",
                        "georeferenced": located,
                        "lonlat_bounds": wgs84_bounds(src),
                        "width": src.width,
                        "height": src.height,
                        "multi_file": path.is_dir(),
                        # Only for a scene that can be placed on a map: there
                        # is nowhere to draw a preview of one that cannot.
                        "preview": preview,
                        # In EPSG:3857 already, because the PNG is reprojected.
                        "preview_extent": preview_extent,
                    })
            except Exception:  # noqa: BLE001 - an unreadable file is not a 500
                scenes.append({"name": path.name, "georeferenced": False})

        return {"scenes": scenes}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/device")
def get_device():
    """One sample of what the process actually knows about its device.

    The frontend's telemetry chart polls this at 1 Hz. It reports only
    measurements this process can take, and says `null` for the rest:

    * **VRAM** comes from `torch.cuda.mem_get_info`, which is a real reading
      of free and total bytes on the active device.
    * **Utilisation** needs NVML (`pynvml`), which is not a dependency here.
      Rather than deriving a plausible-looking number from memory - which
      would be a chart of an invented quantity - the field is null and the
      chart labels that series "not instrumented".

    A dashboard that draws a smooth line for something it never measured is
    worse than a dashboard with one line, so the honest null is the point of
    this endpoint.
    """
    import time

    sample = {
        "t": time.time(),
        "device": "cpu",
        "name": None,
        "vram_free_bytes": None,
        "vram_total_bytes": None,
        "vram_used_fraction": None,
        "utilisation": None,
        "utilisation_source": None,
    }

    try:
        import torch
    except Exception:  # noqa: BLE001 - torch missing is a valid answer here
        return sample

    if not torch.cuda.is_available():
        return sample

    index = torch.cuda.current_device()
    sample["device"] = f"cuda:{index}"
    try:
        sample["name"] = torch.cuda.get_device_name(index)
        free, total = torch.cuda.mem_get_info(index)
        sample["vram_free_bytes"] = int(free)
        sample["vram_total_bytes"] = int(total)
        sample["vram_used_fraction"] = round(1.0 - (free / total), 4) if total else None
    except Exception:  # noqa: BLE001 - a driver hiccup is not a 500
        pass

    # Utilisation, from whichever of the two sources is actually present.
    # NVML is the better one - it is an API, not a subprocess - but it is a
    # dependency this project does not carry, whereas `nvidia-smi` ships with
    # the driver and is therefore in every CUDA image already. Reporting
    # "not instrumented" while a working nvidia-smi sits on the PATH was
    # honest about the wrong thing.
    try:
        import importlib

        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        sample["utilisation"] = int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
        sample["utilisation_source"] = "nvml"
        return sample
    except Exception:  # noqa: BLE001 - NVML is optional; fall through to the CLI
        pass

    try:
        import shutil
        import subprocess

        smi = shutil.which("nvidia-smi")
        if smi:
            out = subprocess.run(
                [
                    smi,
                    f"--id={index}",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                # A hung driver query must not hold the request open: this is
                # polled once a second by the telemetry chart.
                timeout=2,
                check=True,
            )
            sample["utilisation"] = int(out.stdout.strip().splitlines()[0])
            sample["utilisation_source"] = "nvidia-smi"
    except Exception:  # noqa: BLE001 - no driver tool is data, not a failure
        pass

    return sample
