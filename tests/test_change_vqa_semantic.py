"""The semantic path of `change_vqa_v1`, and its precedence (task 2.6).

Two properties matter here and neither is visible in an aggregate score:

1. **The semantic path is reachable.** A precedence rule that silently
   shadows the second path is a mistake this project has already made once -
   the task-3.5 hybrid entailment gate scored identically to deterministic
   alone because NLI was never consulted on the cases that mattered. So the
   ordering is asserted directly, in both directions.
2. **The segmenter is fed what it was trained on.** It learned on 8-bit RGB
   divided by 255. `change_mask_v1` reads through a 2-98 percentile stretch,
   and reusing that reader here would hand the model a different input
   distribution - a silent degradation rather than a loud failure.

The model itself is stubbed: these test the wiring, not the weights. The
weights are measured end to end by the CDVQA benchmark run.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from satquery.ingest import ingest
from satquery.tools import change_vqa
from satquery.tools.change_vqa import ChangeVQASemantic
from satquery.verify.semantic_change import CLASSES

IDX = {name: i for i, name in enumerate(CLASSES)}


def write_rgb(path, value: int = 100, size: int = 64, dtype="uint8"):
    """An 8-bit RGB raster, which is what CDVQA supplies."""
    data = np.full((3, size, size), value, dtype=dtype)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=3,
        dtype=dtype, crs="EPSG:32645", transform=from_origin(0, 0, 1.0, 1.0),
    ) as dst:
        dst.write(data)
        dst.descriptions = ("RED", "GREEN", "BLUE")
    return path


@pytest.fixture
def rgb_pair(tmp_path):
    return [
        write_rgb(tmp_path / "t1.tif", 100),
        write_rgb(tmp_path / "t2.tif", 160),
    ]


def fake_maps(t1_spec: dict[str, int], t2_spec: dict[str, int], size: int = 10):
    """Patch in a known pair of class maps in place of the segmenter."""
    def build(spec):
        flat = np.zeros(size * size, dtype="int64")
        cursor = 0
        for name, count in spec.items():
            flat[cursor : cursor + count] = IDX[name]
            cursor += count
        return flat.reshape(size, size)

    return build(t1_spec), build(t2_spec)


class TestPrecedence:
    def test_the_semantic_path_runs_when_the_index_path_defers(
        self, rgb_pair, monkeypatch
    ):
        """RGB has no NIR, so no index is computable and the deterministic
        path defers. That is exactly the CDVQA case."""
        t1, t2 = fake_maps({"buildings": 30}, {"trees": 30})
        monkeypatch.setattr(
            change_vqa, "predict_class_maps", lambda a, b, size=512: (t1, t2, "fake")
        )

        result = ChangeVQASemantic().run(
            ingest(rgb_pair), {"_query": "Did the areas of trees change?"}
        )
        assert result.payload.data["answer"] == "yes"
        assert result.payload.data["path"] == "semantic_change_map"
        assert result.version == change_vqa.SEMANTIC_VERSION

    def test_the_index_path_still_wins_when_it_can_measure(
        self, msi_6band, msi_6band_t2, monkeypatch
    ):
        """On multispectral imagery the closed-form measurement is the better
        answer, and the segmenter must not be consulted at all."""
        called = []
        monkeypatch.setattr(
            change_vqa,
            "predict_class_maps",
            lambda *a, **k: called.append(1) or (None, None, ""),
        )

        result = ChangeVQASemantic().run(
            ingest([msi_6band, msi_6band_t2]),
            {"_query": "How much did the vegetation change?"},
        )
        assert not called, "the segmenter was consulted despite a measurable index"
        assert result.payload.data["path"] == "deterministic_template"

    def test_an_unanswerable_shape_defers_with_the_right_reason(
        self, rgb_pair, monkeypatch
    ):
        """Deferring with the index path's "no NIR band" reason would be the
        wrong explanation once the segmenter has run."""
        t1, t2 = fake_maps({"buildings": 30}, {"trees": 30})
        monkeypatch.setattr(
            change_vqa, "predict_class_maps", lambda a, b, size=512: (t1, t2, "fake")
        )

        result = ChangeVQASemantic().run(
            ingest(rgb_pair), {"_query": "How many aircraft are on the apron?"}
        )
        assert result.payload.data["deferred"] is True
        assert "semantic change head ran" in result.payload.data["reason"]

    def test_a_single_image_never_reaches_the_segmenter(self, tmp_path, monkeypatch):
        called = []
        monkeypatch.setattr(
            change_vqa,
            "predict_class_maps",
            lambda *a, **k: called.append(1) or (None, None, ""),
        )
        single = ingest([write_rgb(tmp_path / "only.tif")])
        result = ChangeVQASemantic().run(single, {"_query": "Did the trees change?"})
        assert not called
        assert result.payload.data["deferred"] is True


class TestPayload:
    def test_the_measurement_is_carried_into_the_trace(self, rgb_pair, monkeypatch):
        """The answer is one token; the counts behind it are what makes it
        auditable."""
        # Deliberately unequal: 30 buildings against 20 trees. An exact tie
        # defers by design, which is its own test below.
        t1, t2 = fake_maps({"buildings": 30}, {"trees": 20})
        monkeypatch.setattr(
            change_vqa, "predict_class_maps", lambda a, b, size=512: (t1, t2, "fake")
        )
        data = ChangeVQASemantic().run(
            ingest(rgb_pair), {"_query": "What is the largest change?"}
        ).payload.data

        assert data["answer"] == "buildings"
        measurement = data["measurement"]
        assert measurement["question_kind"] == "largest_change"
        assert measurement["class_areas_t1"]["buildings"] == 30
        assert measurement["class_areas_t2"]["trees"] == 20
        assert measurement["changed_fraction"] == pytest.approx(0.3)

    def test_a_tie_between_classes_defers_through_the_tool(
        self, rgb_pair, monkeypatch
    ):
        """The derivation's refusal to break a tie has to survive the tool
        wrapper, not just the module."""
        t1, t2 = fake_maps({"buildings": 30}, {"trees": 30})
        monkeypatch.setattr(
            change_vqa, "predict_class_maps", lambda a, b, size=512: (t1, t2, "fake")
        )
        result = ChangeVQASemantic().run(
            ingest(rgb_pair), {"_query": "What is the largest change?"}
        )
        assert result.payload.data["deferred"] is True

    def test_confidence_is_not_claimed_to_be_certainty(self, rgb_pair, monkeypatch):
        """The arithmetic is exact but the segmentation is not, so a
        confidence of 1.0 here would hide where the error actually lives."""
        t1, t2 = fake_maps({"buildings": 30}, {"trees": 30})
        monkeypatch.setattr(
            change_vqa, "predict_class_maps", lambda a, b, size=512: (t1, t2, "fake")
        )
        result = ChangeVQASemantic().run(
            ingest(rgb_pair), {"_query": "Did the areas of trees change?"}
        )
        assert result.confidence < 1.0
        assert result.confidence_method == "segmentation_derived"


class TestPreprocessing:
    def test_eight_bit_input_is_scaled_by_255_not_stretched(self, tmp_path):
        """A flat 8-bit image must read as its true level. A percentile
        stretch would map a constant image to zeros and a two-level image to
        full range - neither is what the model trained on."""
        path = write_rgb(tmp_path / "flat.tif", value=128, size=32)
        meta = ingest([path]).images[0]
        arr = change_vqa.read_rgb_as_trained(meta, 32)
        assert arr.shape == (3, 32, 32)
        assert arr.min() == pytest.approx(128 / 255, abs=1e-3)
        assert arr.max() == pytest.approx(128 / 255, abs=1e-3)

    def test_higher_bit_depth_still_gets_a_stretch(self, tmp_path):
        """16-bit ranges are sensor-dependent, so those do need normalising."""
        size = 32
        data = np.linspace(0, 4000, size * size, dtype="uint16").reshape(size, size)
        path = tmp_path / "wide.tif"
        with rasterio.open(
            path, "w", driver="GTiff", height=size, width=size, count=3,
            dtype="uint16", crs="EPSG:32645",
            transform=from_origin(0, 0, 1.0, 1.0),
        ) as dst:
            for band in range(1, 4):
                dst.write(data, band)
            dst.descriptions = ("RED", "GREEN", "BLUE")

        meta = ingest([path]).images[0]
        arr = change_vqa.read_rgb_as_trained(meta, size)
        assert arr.max() == pytest.approx(1.0, abs=1e-3)
        assert arr.min() == pytest.approx(0.0, abs=1e-3)


class TestAvailability:
    def test_unset_env_reports_why(self, monkeypatch):
        monkeypatch.delenv(change_vqa.ENV_SEMANTIC, raising=False)
        ok, reason = change_vqa.semantic_available()
        assert not ok
        assert change_vqa.ENV_SEMANTIC in reason

    def test_a_missing_checkpoint_reports_why(self, monkeypatch, tmp_path):
        monkeypatch.setenv(change_vqa.ENV_SEMANTIC, str(tmp_path / "absent.pt"))
        ok, reason = change_vqa.semantic_available()
        assert not ok
        assert "not found" in reason

    def test_the_registry_falls_back_to_the_template_without_a_checkpoint(
        self, monkeypatch
    ):
        """The fallback is a working deterministic answerer, not a stub."""
        monkeypatch.delenv(change_vqa.ENV_SEMANTIC, raising=False)
        from satquery.tools.stubs import _change_vqa_tool

        assert type(_change_vqa_tool()).__name__ == "ChangeVQATemplate"
