"""PDF report generation (plan task 3.12).

Renders a `Trace` into a self-contained PDF: the query, the answer, the
confidence breakdown, the routing decision, every executed step, the
verification result including the entailment gate, the measured index
statistics, and previews of the rasters the run produced.

Two design rules, both consequences of what the rest of Phase 3 established:

**The PDF reports the trace, it does not re-derive anything.** Every number
here comes from a field the pipeline already wrote. A report that recomputed
statistics could disagree with the trace it claims to describe, and then there
would be two answers and no way to tell which the system actually gave.

**A missing or unmeasured value is printed as such.** `ece_after = -1.0` is
rendered "not measured", an empty complementarity block says "not computed",
and an abstention prints its trigger and resolving input. The sentinels exist
precisely so a reader can tell an unmeasured value from a measured one, and a
report that silently omitted them would undo that.

reportlab is an optional dependency: `satquery.report` is not on the runtime
path, and a machine that only serves queries should not need a PDF engine.
The import is deferred and the error names the fix.
"""

from __future__ import annotations

from pathlib import Path

from satquery.contracts.trace import Trace

PAGE_MARGIN = 42
PREVIEW_MAX_PX = 900

BAND_COLOURS = {
    "HIGH": (0.16, 0.55, 0.28),
    "MEDIUM": (0.80, 0.55, 0.10),
    "LOW": (0.70, 0.18, 0.16),
}


def _require_reportlab():
    try:
        from reportlab.lib import colors  # noqa: F401
        from reportlab.lib.pagesizes import A4  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PDF reporting needs reportlab, which is an optional dependency: "
            "pip install reportlab==5.0.1"
        ) from exc


def raster_preview(path: Path, out_dir: Path) -> Path | None:
    """Render a raster to a PNG the PDF can embed.

    Returns None rather than raising when the file is missing or unreadable:
    a report is a summary of a run that already happened, and losing one
    preview must not lose the whole document.
    """
    try:
        import numpy as np
        import rasterio
        from PIL import Image
    except ImportError:  # pragma: no cover - environment dependent
        return None

    try:
        with rasterio.open(path) as src:
            count = min(src.count, 3)
            data = src.read(list(range(1, count + 1)), masked=True).astype("float32")
    except Exception:  # noqa: BLE001 - a missing preview is not a failed report
        return None

    # Percentile stretch, not min/max: a single hot pixel or a nodata sentinel
    # would otherwise flatten the whole image to black.
    finite = data.compressed() if hasattr(data, "compressed") else data.ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, [2, 98])
    if high <= low:
        high = low + 1.0

    scaled = np.clip((np.asarray(data.filled(low)) - low) / (high - low), 0, 1)
    scaled = (scaled * 255).astype("uint8")

    if scaled.shape[0] == 1:
        image = Image.fromarray(scaled[0], mode="L")
    else:
        rgb = np.zeros((3, *scaled.shape[1:]), dtype="uint8")
        rgb[: scaled.shape[0]] = scaled[:3]
        image = Image.fromarray(np.transpose(rgb, (1, 2, 0)), mode="RGB")

    image.thumbnail((PREVIEW_MAX_PX, PREVIEW_MAX_PX))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{Path(path).stem}_preview.png"
    image.save(target)
    return target


class _Writer:
    """Thin layout helper: a cursor, a page break, and consistent styles."""

    def __init__(self, canvas, page_size):
        self.c = canvas
        self.width, self.height = page_size
        self.y = self.height - PAGE_MARGIN

    def space(self, amount: float = 10) -> None:
        self.y -= amount

    def _ensure(self, needed: float) -> None:
        if self.y - needed < PAGE_MARGIN:
            self.c.showPage()
            self.y = self.height - PAGE_MARGIN

    def title(self, text: str) -> None:
        self._ensure(30)
        self.c.setFillColorRGB(0, 0, 0)
        self.c.setFont("Helvetica-Bold", 16)
        self.c.drawString(PAGE_MARGIN, self.y, text)
        self.y -= 22

    def heading(self, text: str) -> None:
        self._ensure(26)
        self.c.setFillColorRGB(0.15, 0.15, 0.15)
        self.c.setFont("Helvetica-Bold", 11)
        self.c.drawString(PAGE_MARGIN, self.y, text.upper())
        self.y -= 6
        self.c.setStrokeColorRGB(0.8, 0.8, 0.8)
        self.c.line(PAGE_MARGIN, self.y, self.width - PAGE_MARGIN, self.y)
        self.y -= 12

    def line(self, text: str, bold: bool = False, colour=None) -> None:
        self._ensure(14)
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", 9)
        self.c.setFillColorRGB(*(colour or (0.1, 0.1, 0.1)))
        self.c.drawString(PAGE_MARGIN, self.y, text[:170])
        self.y -= 12

    def paragraph(self, text: str, indent: float = 0) -> None:
        """Wrap on width, because a truncated answer is a misleading answer."""
        from reportlab.pdfbase.pdfmetrics import stringWidth

        self.c.setFont("Helvetica", 9)
        self.c.setFillColorRGB(0.1, 0.1, 0.1)
        limit = self.width - 2 * PAGE_MARGIN - indent
        words = str(text).split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if stringWidth(candidate, "Helvetica", 9) > limit and current:
                self._ensure(13)
                self.c.setFont("Helvetica", 9)
                self.c.drawString(PAGE_MARGIN + indent, self.y, current)
                self.y -= 12
                current = word
            else:
                current = candidate
        if current:
            self._ensure(13)
            self.c.setFont("Helvetica", 9)
            self.c.drawString(PAGE_MARGIN + indent, self.y, current)
            self.y -= 12

    def key_values(self, pairs: list[tuple[str, str]]) -> None:
        for key, value in pairs:
            self._ensure(14)
            self.c.setFont("Helvetica-Bold", 9)
            self.c.setFillColorRGB(0.35, 0.35, 0.35)
            self.c.drawString(PAGE_MARGIN, self.y, f"{key}")
            self.c.setFont("Helvetica", 9)
            self.c.setFillColorRGB(0.1, 0.1, 0.1)
            self.c.drawString(PAGE_MARGIN + 170, self.y, str(value)[:120])
            self.y -= 12

    def image(self, path: Path, caption: str, max_w: float = 240) -> None:
        from reportlab.lib.utils import ImageReader

        try:
            reader = ImageReader(str(path))
            iw, ih = reader.getSize()
        except Exception:  # noqa: BLE001
            return
        scale = min(max_w / iw, 1.0)
        w, h = iw * scale, ih * scale
        self._ensure(h + 22)
        self.c.drawImage(reader, PAGE_MARGIN, self.y - h, width=w, height=h)
        self.y -= h + 4
        self.c.setFont("Helvetica-Oblique", 8)
        self.c.setFillColorRGB(0.4, 0.4, 0.4)
        self.c.drawString(PAGE_MARGIN, self.y, caption[:140])
        self.y -= 14


def _calibration_text(trace: Trace) -> str:
    calibration = trace.confidence.calibration
    if calibration.ece_after < 0:
        # The documented sentinel. Printing "-1.0" would look like a measured
        # value; printing nothing would look like a passed check.
        return f"{calibration.method} (ECE not measured)"
    return (
        f"{calibration.method}, T={calibration.T:.4f}, "
        f"held-out ECE={calibration.ece_after:.4f}"
    )


def export_pdf(
    trace: Trace,
    out_path: str | Path,
    preview_dir: str | Path | None = None,
    compress: bool = True,
) -> Path:
    """Render `trace` to a PDF at `out_path`.

    `compress=False` writes uncompressed content streams. That exists so the
    tests can assert what the document actually SAYS by reading the bytes,
    without adding a PDF parser as a test dependency. A test that only
    checked the file was non-empty would pass for a blank page.
    """
    _require_reportlab()
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as pdfcanvas

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(preview_dir or out_path.parent / "previews")

    c = pdfcanvas.Canvas(str(out_path), pagesize=A4)
    c.setPageCompression(1 if compress else 0)
    w = _Writer(c, A4)

    w.title("SatQuery AI - Run Report")
    w.key_values([
        ("Run ID", trace.run_id),
        ("Timestamp (UTC)", trace.timestamp_utc),
        ("Code version", trace.code_version),
        ("Capability matrix", trace.routing.capability_matrix_version),
    ])
    w.space(6)

    w.heading("Query")
    w.paragraph(trace.query or "(empty)")
    w.space(4)

    w.heading("Answer")
    w.paragraph(trace.answer or "(no answer)")
    if trace.abstained:
        w.space(4)
        w.line("ABSTAINED", bold=True, colour=BAND_COLOURS["LOW"])
        w.key_values([
            ("Trigger", trace.abstain_trigger or "unset"),
            ("Limiting component", trace.abstain_limiting_component or "n/a"),
        ])
        w.paragraph(f"Reason: {trace.abstain_reason or 'unstated'}")
        w.paragraph(
            f"What would resolve it: {trace.abstain_resolving_input or 'unstated'}"
        )
    w.space(6)

    w.heading("Confidence")
    components = trace.confidence.components
    w.line(
        f"{trace.confidence.final:.4f}   band {trace.confidence.band}",
        bold=True,
        colour=BAND_COLOURS.get(trace.confidence.band, (0.1, 0.1, 0.1)),
    )
    w.key_values([
        ("model", f"{components.model:.4f}"),
        ("agreement", f"{components.agreement:.4f}"),
        ("input_quality", f"{components.input_quality:.4f}"),
        ("calibration", _calibration_text(trace)),
    ])
    w.paragraph(
        "Components are combined with a weighted geometric mean, so any one "
        "of them collapsing collapses the score. Weights are equal until "
        "there is labelled data to fit them on.",
    )
    w.space(6)

    w.heading("Routing")
    w.key_values([
        ("Selected task", trace.routing.selected_task),
        ("Legal tasks", ", ".join(trace.routing.legal_tasks)),
        ("Classifier", trace.routing.classifier.name),
        ("Top-1 / margin",
         f"{trace.routing.classifier.top1:.3f} / "
         f"{trace.routing.classifier.margin:.3f}"),
        ("LLM tiebreak", str(trace.routing.llm_tiebreak_invoked)),
    ])
    if trace.routing.config_excluded_task:
        w.paragraph(
            f"The query's most likely task, {trace.routing.config_excluded_task}, "
            f"is not supported by this input configuration and was excluded by "
            f"config gating before the classifier chose.",
        )
    w.space(6)

    w.heading("Ingest")
    w.key_values([
        ("Mode", trace.ingest.mode),
        ("Configuration", trace.ingest.config),
        ("Images", str(len(trace.ingest.images))),
    ])
    for check in trace.ingest.checks:
        if check.get("status") != "PASS":
            w.line(
                f"  {check.get('status')}  {check.get('name')}: "
                f"{check.get('message', '')}",
                colour=BAND_COLOURS["MEDIUM"],
            )
    w.space(6)

    w.heading("Execution")
    if not trace.execution:
        w.line("no steps executed")
    for step in trace.execution:
        w.line(
            f"{step.step}  {step.tool} v{step.version}  "
            f"conf {step.confidence:.3f} ({step.confidence_method})  "
            f"{step.runtime_ms} ms",
        )
        w.line(f"    rationale: {step.rationale_tag}", colour=(0.4, 0.4, 0.4))
    w.space(6)

    w.heading("Verification")
    verification = trace.verification
    w.key_values([
        ("Built-up path", verification.built_up_path),
        ("Complementarity",
         str(verification.complementarity) if verification.complementarity
         else "not computed (task 2.3)"),
    ])
    for name, value in sorted(verification.physics_agreement.items()):
        w.line(f"  {name}: {value:.3f}")
    for conflict in verification.conflicts:
        w.line(f"  conflict: {conflict}", colour=BAND_COLOURS["MEDIUM"])

    gate = verification.entailment_gate
    w.space(4)
    w.line("Entailment gate", bold=True)
    w.key_values([
        ("Backend", gate.backend),
        ("Sentences", str(gate.sentences)),
        ("Retained", str(gate.retained)),
        ("Flagged", str(gate.flagged)),
        ("Unverifiable", str(gate.unverifiable)),
        ("Action taken", gate.action),
    ])
    if gate.unverifiable:
        w.paragraph(
            "'Unverifiable' means no premise in the payload could speak to "
            "the sentence. It is reported separately from 'retained' so a "
            "retention rate is not mistaken for a verification rate.",
        )
    for flagged in gate.flagged_detail:
        w.paragraph(f"removed: \"{flagged.sentence}\" - {flagged.reason}", indent=10)
    w.space(6)

    if trace.artifacts:
        w.heading("Artifacts and previews")
        for artifact in trace.artifacts:
            w.line(f"  {artifact}  ->  "
                   f"{trace.artifact_paths.get(artifact, 'path not recorded')}")
        w.space(4)
        for key, raw in sorted(trace.artifact_paths.items()):
            path = Path(raw)
            if path.suffix.lower() not in {".tif", ".tiff", ".png"}:
                continue
            preview = raster_preview(path, preview_dir)
            if preview is not None:
                w.image(preview, f"{key} - {path.name} (2-98% percentile stretch)")

    c.showPage()
    c.save()
    return out_path
