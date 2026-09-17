"""Evidence pack export (plan task 2.11).

Bundles everything needed to independently check an answer into one ZIP:
the trace, the index rasters as COGs, footprints and detections as GeoJSON,
and a manifest listing what each file is and where it came from.

The design constraint the plan sets is that the mask must **open correctly
georeferenced in QGIS**. That drives two decisions:

* GeoJSON is written in **EPSG:4326**, because RFC 7946 mandates it and a
  GeoJSON carrying projected UTM coordinates will silently land in the Gulf of
  Guinea when opened. The source CRS is recorded in properties so nothing is
  lost.
* Rasters are copied as-is rather than re-encoded. They are already COGs from
  the index engine, and a re-encode risks changing values in a bundle whose
  entire purpose is verifiability.

`evidence.json` records a SHA-256 for every file, so a reviewer can prove the
bundle was not altered after export.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

from satquery.contracts.trace import Trace

GEOJSON_CRS = "EPSG:4326"
CHUNK = 1024 * 1024


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def raster_footprint(path: Path) -> dict | None:
    """Footprint of a raster as a GeoJSON Feature in EPSG:4326."""
    try:
        with rasterio.open(path) as src:
            if src.crs is None:
                return None
            bounds = src.bounds
            west, south, east, north = transform_bounds(
                src.crs, GEOJSON_CRS, *bounds, densify_pts=21
            )
            source_crs = str(src.crs)
            width, height, gsd = src.width, src.height, abs(src.transform.a)
    except Exception:  # noqa: BLE001 - a bad raster must not kill the export
        return None

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [west, south], [east, south], [east, north],
                [west, north], [west, south],
            ]],
        },
        "properties": {
            "name": path.name,
            # The reprojection to 4326 is lossy for area calculations, so the
            # native CRS is preserved for anyone who needs to go back.
            "source_crs": source_crs,
            "width": width,
            "height": height,
            "gsd_m": round(gsd, 4),
        },
    }


def boxes_to_geojson(trace: Trace) -> dict | None:
    """Detections as a GeoJSON FeatureCollection, if any were produced.

    Boxes are in pixel coordinates, so they are converted through the source
    image's affine transform. Without a georeferenced source they are omitted
    rather than written as raw pixel indices masquerading as coordinates.
    """
    boxes = []
    for step in trace.execution:
        for box in step.outputs.get("bounding_boxes", []) or []:
            boxes.append(box)
    if not boxes:
        return None

    image = next((i for i in trace.ingest.images if i.get("path")), None)
    if image is None:
        return None

    try:
        with rasterio.open(image["path"]) as src:
            if src.crs is None:
                return None
            transform, crs = src.transform, src.crs
    except Exception:  # noqa: BLE001
        return None

    features = []
    for i, box in enumerate(boxes):
        try:
            xmin, ymin = float(box["xmin"]), float(box["ymin"])
            xmax, ymax = float(box["xmax"]), float(box["ymax"])
        except (KeyError, TypeError, ValueError):
            continue
        corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        projected = [transform * (x, y) for x, y in corners]
        lons, lats = zip(*projected)
        west, south, east, north = transform_bounds(
            crs, GEOJSON_CRS, min(lons), min(lats), max(lons), max(lats)
        )
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[
                [west, south], [east, south], [east, north],
                [west, north], [west, south],
            ]]},
            "properties": {
                "id": i,
                "label": box.get("label"),
                "score": box.get("score"),
                "pixel_bbox": [xmin, ymin, xmax, ymax],
            },
        })

    return {"type": "FeatureCollection", "features": features} if features else None


def export(
    trace: Trace,
    out_dir: str | Path,
    artifact_dir: str | Path | None = None,
    zip_output: bool = True,
) -> Path:
    """Write an evidence pack for `trace`. Returns the ZIP (or directory)."""
    out_dir = Path(out_dir)
    pack = out_dir / f"evidence_{trace.run_id}"
    rasters_dir = pack / "rasters"
    pack.mkdir(parents=True, exist_ok=True)
    rasters_dir.mkdir(exist_ok=True)

    files: list[dict] = []

    def record(path: Path, kind: str, description: str) -> None:
        files.append({
            "file": str(path.relative_to(pack)).replace("\\", "/"),
            "kind": kind,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "description": description,
        })

    trace_path = pack / "trace.json"
    trace_path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    record(trace_path, "json", "Full execution trace: routing, tools, verification")

    # Index rasters, copied byte-for-byte. They are already COGs; re-encoding
    # could change values in a bundle whose purpose is verification.
    if artifact_dir:
        source = Path(artifact_dir) / trace.run_id
        if source.is_dir():
            for raster in sorted(source.glob("*.tif")):
                target = rasters_dir / raster.name
                shutil.copy2(raster, target)
                record(target, "cog", f"{raster.stem.upper()} index raster")

    footprints = []
    for image in trace.ingest.images:
        path = Path(image.get("path", ""))
        if path.exists():
            feature = raster_footprint(path)
            if feature:
                feature["properties"]["role"] = image.get("role")
                feature["properties"]["modality"] = image.get("modality")
                footprints.append(feature)

    if footprints:
        fp = pack / "footprint.geojson"
        fp.write_text(
            json.dumps({"type": "FeatureCollection", "features": footprints}, indent=2),
            encoding="utf-8",
        )
        record(fp, "geojson", f"Input footprints in {GEOJSON_CRS}")

    detections = boxes_to_geojson(trace)
    if detections:
        det = pack / "detections.geojson"
        det.write_text(json.dumps(detections, indent=2), encoding="utf-8")
        record(det, "geojson", f"Detected regions in {GEOJSON_CRS}")

    answer_path = pack / "answer.txt"
    answer_path.write_text(
        f"Query:\n{trace.query}\n\nAnswer:\n{trace.answer}\n\n"
        f"Confidence: {trace.confidence.final} ({trace.confidence.band})\n"
        f"Task: {trace.routing.selected_task}\n"
        + (f"Abstained: {trace.abstain_reason}\n" if trace.abstained else ""),
        encoding="utf-8",
    )
    record(answer_path, "text", "Query, answer and confidence in plain text")

    evidence = {
        "run_id": trace.run_id,
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "code_version": trace.code_version,
        "matrix_version": trace.routing.capability_matrix_version,
        "query": trace.query,
        "answer": trace.answer,
        "task": trace.routing.selected_task,
        "confidence": {
            "final": trace.confidence.final,
            "band": trace.confidence.band,
            "components": trace.confidence.components.model_dump(),
        },
        "verification": {
            "physics_agreement": trace.verification.physics_agreement,
            "built_up_path": trace.verification.built_up_path,
            "conflicts": trace.verification.conflicts,
        },
        "abstained": trace.abstained,
        "abstain_reason": trace.abstain_reason,
        "geojson_crs": GEOJSON_CRS,
        "files": files,
        "integrity_note": (
            "Each sha256 is of the file as written. Recomputing them proves the "
            "bundle was not altered after export."
        ),
    }
    (pack / "evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    if not zip_output:
        return pack

    archive = out_dir / f"evidence_{trace.run_id}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(pack.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(pack))
    shutil.rmtree(pack, ignore_errors=True)
    return archive
