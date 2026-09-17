"""General natural-language questions reach a capability (PS-26167).

The system is meant to be a general-purpose assistant for remote-sensing
imagery, not a front end for nine benchmark phrasings. Before this suite,
"Is this an urban or rural scene?" classified as CLARIFY_OR_ABSTAIN at
**0.436 top-1 with a 0.162 margin** - confidently enough that the router
took it - because the intent bank's abstain class was trained on content-free
filler ("What is this?", "Anything interesting?") and that was the nearest
neighbour of any short unadorned question. The product refused a question
`rs_vqa_v1` answers directly.

Two things are pinned here, and the second matters as much as the first:

* general questions reach a usable single-image capability, and
* the specialist routes and the honest refusals are **unchanged** - a fix
  that made everything answerable by widening the abstain gate would pass
  the first half of this file and fail the second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio

from satquery.contracts.input_manifest import IngestMode
from satquery.controller.matrix_loader import load_matrix
from satquery.controller.router import Router
from satquery.ingest import ingest

MATRIX_PATH = Path("configs/capability_matrix.yaml")

# Tasks that answer a question about what is visible in one image. Which of
# them serves a given general question is the router's business; that it is
# one of them, rather than a refusal, is the requirement.
SINGLE_IMAGE_VISUAL = {
    "SINGLE_VQA", "SINGLE_CAPTION", "SINGLE_GROUND", "SINGLE_LANDCOVER",
}


@pytest.fixture(scope="module")
def matrix():
    return load_matrix(MATRIX_PATH)


@pytest.fixture(scope="module")
def router(matrix):
    return Router(matrix)


def _write_plain_image(path: Path, driver: str) -> Path:
    """A PNG or JPEG with no georeferencing, as a user would upload one."""
    array = (np.random.default_rng(11).random((3, 128, 128)) * 255).astype("uint8")
    with rasterio.open(
        path, "w", driver=driver, height=128, width=128, count=3, dtype="uint8"
    ) as dst:
        dst.write(array)
    return path


class TestGeneralQuestionsReachACapability:
    """PS items 1-3: general single-image questions must not be refused."""

    # The shapes the PS lists, plus paraphrases that appear in none of the
    # training templates, so this measures generalisation rather than recall.
    GENERAL_QUERIES = [
        "Is this an urban or rural scene?",
        "Is this urban or rural?",
        "Are there buildings?",
        "What do you see?",
        "What is in this image?",
        "What is happening here?",
        "Are there roads?",
        "Does this area contain vegetation?",
        "What kind of landscape is this?",
        "Does this look like farmland?",
        "What is unusual in this image?",
        "How dense is the built-up area?",
        "What can you tell me about this image?",
        "What objects are visible?",
        "Describe the visible environment.",
        # Unseen phrasings, deliberately not in the template bank.
        "Is this an industrial estate or a residential neighbourhood?",
        "Would you say this area has been developed?",
        "Can you see any water here?",
    ]

    @pytest.mark.parametrize("query", GENERAL_QUERIES)
    def test_a_general_question_is_not_refused(self, router, msi_6band, query):
        plan = router.route(query, ingest([msi_6band]))
        assert plan.tasks[0] in SINGLE_IMAGE_VISUAL, (
            f"{query!r} routed to {plan.tasks[0]}, refusing a question the "
            f"single-image capabilities can answer"
        )

    def test_the_reported_defect_query(self, router, msi_6band):
        """The exact query from the defect report."""
        plan = router.route("Is this an urban or rural scene?", ingest([msi_6band]))
        assert plan.tasks[0] != "CLARIFY_OR_ABSTAIN"
        assert plan.tasks[0] == "SINGLE_VQA"

    def test_presence_question_reaches_a_usable_visual_tool(self, router, msi_6band):
        plan = router.route("Are there buildings?", ingest([msi_6band]))
        assert plan.tasks[0] in SINGLE_IMAGE_VISUAL
        assert plan.steps, "a usable capability must have at least one step"

    def test_what_is_in_this_image_still_works(self, router, msi_6band):
        plan = router.route("What is in this image?", ingest([msi_6band]))
        assert plan.tasks[0] in SINGLE_IMAGE_VISUAL


class TestConversationalQuestionsDoNotNeedTheCaptioner:
    """The second routing defect, found on the GPU deployment.

    "Describe what you see.", "Describe the landscape." and "Give me an
    overview of this image." are conversational questions, but every
    `describe`/`overview` phrasing lived in the caption bank, so they routed
    to SINGLE_CAPTION and came back as the caption stub - on a deployment
    where `rs_vqa_v1` was loaded and could have answered. The boundary is now
    *what was asked for*, not the presence of the verb "describe":

    * a caption of **the image** -> SINGLE_CAPTION (unchanged, and still the
      mandatory second single-image capability), and
    * what **you can see**, or something **in** the scene -> SINGLE_VQA.
    """

    CONVERSATIONAL = [
        "What is in this image?",
        "What can you tell me about this image?",
        "What can you tell me about this satellite image?",
        "Describe what you see.",
        "Describe the landscape.",
        "What features are visible?",
        "What is happening in this image?",
        "Give me an overview of this image.",
        # Unseen phrasings on the same side of the boundary.
        "Describe what you can make out here.",
        "Tell me about this area.",
    ]

    @pytest.mark.parametrize("query", CONVERSATIONAL)
    def test_conversational_questions_use_the_vqa_path(
        self, router, msi_6band, query
    ):
        plan = router.route(query, ingest([msi_6band]))
        assert plan.tasks[0] == "SINGLE_VQA", (
            f"{query!r} routed to {plan.tasks[0]}; a conversational question "
            f"must not require the captioning model to be loaded"
        )
        assert [s.tool for s in plan.steps] == ["rs_vqa_v1"]

    EXPLICIT_CAPTION = [
        "Generate a formal caption for this image.",
        "Caption this scene.",
        "Write a caption for this satellite image.",
        "Describe this image.",
        "Write a few sentences about this image.",
        "In a sentence or two, describe this scene.",
        # The PS's own representative query, which is a caption request.
        "Describe the land-cover and major objects visible in this image.",
    ]

    @pytest.mark.parametrize("query", EXPLICIT_CAPTION)
    def test_explicit_caption_requests_still_reach_the_captioner(
        self, router, msi_6band, query
    ):
        """The mandatory second single-image capability is not cannibalised."""
        plan = router.route(query, ingest([msi_6band]))
        assert plan.tasks[0] == "SINGLE_CAPTION"
        assert "caption_v1" in {s.tool for s in plan.steps}


class TestExistingVQARoutesAreUnchanged:
    """PS items 4-5: the QLoRA VQA model keeps the queries it already served."""

    @pytest.mark.parametrize(
        "query",
        [
            "Is there water in this image?",
            "Is there vegetation in this image?",
            "How many buildings are visible?",
            "Is there a bridge in this scene?",
        ],
    )
    def test_vqa_queries_still_use_rs_vqa_v1(self, router, msi_6band, query):
        plan = router.route(query, ingest([msi_6band]))
        assert plan.tasks[0] == "SINGLE_VQA"
        assert [s.tool for s in plan.steps] == ["rs_vqa_v1"]


class TestSpecialistRoutesAreUnchanged:
    """PS items 6-9: widening the general path must not cannibalise these."""

    @pytest.mark.parametrize(
        "query",
        ["Describe this image.", "Caption this scene.",
         "Write a caption for this satellite image."],
    )
    def test_caption_requests_reach_captioning(self, router, msi_6band, query):
        plan = router.route(query, ingest([msi_6band]))
        assert plan.tasks[0] == "SINGLE_CAPTION"
        assert "caption_v1" in {s.tool for s in plan.steps}

    @pytest.mark.parametrize(
        "query",
        ["Show me where the roads are.", "Locate the water bodies.",
         "Draw boxes around the rooftops."],
    )
    def test_location_requests_reach_grounding(self, router, msi_6band, query):
        plan = router.route(query, ingest([msi_6band]))
        assert plan.tasks[0] == "SINGLE_GROUND"
        assert "grounding_v1" in {s.tool for s in plan.steps}

    @pytest.mark.parametrize(
        "query",
        ["Describe what changed between the two images.",
         "Produce a change mask.",
         "How much did the built-up area change?"],
    )
    def test_change_queries_reach_the_change_workflow(
        self, router, msi_6band, msi_6band_t2, query
    ):
        plan = router.route(query, ingest([msi_6band, msi_6band_t2]))
        assert plan.tasks[0].startswith("TEMPORAL_")

    @pytest.mark.parametrize(
        "query",
        ["Which areas became built up?", "Which areas became urban?",
         "Which parts turned into farmland?", "Which areas were cleared?",
         "What became built up?", "What changed?"],
    )
    def test_transition_phrasings_reach_the_change_workflow(
        self, router, msi_6band, msi_6band_t2, query
    ):
        """"Which areas became X" is a change question in conversational dress.

        It scored 0.326 for change against 0.273 for SINGLE_VQA - under the
        confidence bar - so the router fell back to the single-image default
        and rs_vqa_v1 answered "only one image was provided" about a
        two-image input. A wrong statement about the input is worse than a
        worse answer, which is why this is pinned.
        """
        plan = router.route(query, ingest([msi_6band, msi_6band_t2]))
        assert plan.tasks[0].startswith("TEMPORAL_")

    def test_optical_sar_query_reaches_the_crossmodal_workflow(
        self, router, msi_6band, sar_dualpol
    ):
        plan = router.route(
            "Combine the optical and radar images to find buildings.",
            ingest([msi_6band, sar_dualpol]),
        )
        assert plan.tasks[0] == "XMODAL_JOINT_EXTRACT"
        assert "optsar_fusion_v1" in {s.tool for s in plan.steps}


class TestLocationIsAnsweredFromTheManifest:
    """"Where is this?" was answered with a falsehood.

    `rs_vqa_v1` declines location questions with "Satellite imagery does not
    carry that information." On a GeoTIFF that is simply untrue - the file
    carries a CRS and an affine transform, which is exactly that information -
    and the adapter cannot know the difference, because it never sees the
    georeferencing. Measured on a real Cartosat crop whose header puts it at
    20.32N, 85.84E: the answer still said the information was not there.

    The correction is made from the manifest, and only when the question was
    about location, the tool declined, AND the input is georeferenced.
    """

    REFUSAL = (
        "I cannot answer that from this image. Satellite imagery does not "
        "carry that information."
    )

    def test_the_manifest_measures_the_geographic_extent(self, msi_6band):
        image = ingest([msi_6band]).images[0]
        assert image.lonlat_bounds is not None
        west, south, east, north = image.lonlat_bounds
        assert -180 <= west < east <= 180
        assert -90 <= south < north <= 90

    def test_an_ungeoreferenced_png_has_no_extent(self, tmp_path):
        image = ingest([_write_plain_image(tmp_path / "s.png", "PNG")]).images[0]
        assert image.lonlat_bounds is None

    @pytest.mark.parametrize(
        "query",
        ["What place is this?", "Which city is this?", "Where is this?",
         "What are the coordinates of this scene?", "Which country is this?"],
    )
    def test_a_location_refusal_is_corrected_with_measured_coordinates(
        self, msi_6band, query
    ):
        from satquery.controller.executor import _location_disclosure

        manifest = ingest([msi_6band])
        disclosure = _location_disclosure(query, self.REFUSAL, manifest)
        assert disclosure is not None
        assert "location IS known" in disclosure
        # The honest half: coordinates are measured, a place NAME is not.
        assert "gazetteer" in disclosure

    def test_the_false_claim_is_dropped_rather_than_argued_with(self):
        """Appending a correction after a falsehood leaves both on screen."""
        from satquery.controller.executor import _drop_false_location_claim

        cleaned = _drop_false_location_claim(self.REFUSAL)
        assert "does not carry that information" not in cleaned
        # The true half survives: the PIXELS really do not carry the location.
        assert "cannot answer that from this image" in cleaned

    def test_the_decline_stands_when_there_is_no_georeferencing(self, tmp_path):
        """For a PNG the model's decline is TRUE, so it must not be overridden."""
        from satquery.controller.executor import _location_disclosure

        manifest = ingest([_write_plain_image(tmp_path / "s.png", "PNG")])
        assert _location_disclosure("Where is this?", self.REFUSAL, manifest) is None

    def test_a_non_location_question_is_untouched(self, msi_6band):
        from satquery.controller.executor import _location_disclosure

        manifest = ingest([msi_6band])
        assert _location_disclosure("Are there buildings?", self.REFUSAL, manifest) is None

    def test_an_answered_question_is_untouched(self, msi_6band):
        """Only a decline is corrected; a real answer is left alone."""
        from satquery.controller.executor import _location_disclosure

        manifest = ingest([msi_6band])
        assert _location_disclosure("Where is this?", "urban", manifest) is None


class TestAbstentionIsNotWeakened:
    """PS item 10, and the constraint the whole change is bounded by.

    Nothing here is about making a query pass. If widening the general path
    had been done by lowering the abstain gate, these are what would fail.
    """

    @pytest.mark.parametrize(
        "query",
        ["Hello.", "Hi there.", "Do the thing.", "Sort it out.",
         "What should I do next?", "Which one is better?",
         "What about the other one?", "Continue.",
         # Lower-case, unpunctuated filler. Two of these ("ok", "thoughts?")
         # did NOT abstain at any point before this change either; they are
         # here because widening the general path is exactly the change that
         # could have made them worse, and it did not.
         "hey", "ok", "thoughts?", "just have a look and let me know"],
    )
    def test_content_free_requests_still_abstain(self, router, msi_6band, query):
        plan = router.route(query, ingest([msi_6band]))
        assert plan.tasks[0] == "CLARIFY_OR_ABSTAIN"

    @pytest.mark.parametrize(
        "query", ["", "   ", "\n\n", "hmm", "asdfghjkl"]
    )
    def test_contentless_queries_abstain_regardless_of_classifier_margin(
        self, router, msi_6band, query
    ):
        """The case that regressed twice while the template bank was tuned.

        A query with no features lands on the class prior: CLARIFY_OR_ABSTAIN
        top-1 at ~0.35 with a ~0.04 margin, under the generic confidence bar.
        Whether it abstained was therefore decided by where a linear model's
        prior happened to sit, and it flipped every time the bank changed
        size. The router now honours an abstain pick whatever its margin, so
        this is a property of the routing rule rather than of the fit.
        """
        plan = router.route(query, ingest([msi_6band]))
        assert plan.tasks[0] == "CLARIFY_OR_ABSTAIN"
        assert plan.steps == []

    def test_failed_input_checks_still_force_abstention(self, router, no_crs_raster):
        """Input validation abstention is untouched by the routing change."""
        plan = router.route("Is this urban or rural?", ingest([no_crs_raster]))
        assert plan.tasks[0] == "CLARIFY_OR_ABSTAIN"

    def test_a_change_query_on_one_image_still_cannot_route_to_change(
        self, router, msi_6band
    ):
        plan = router.route("Produce a change mask.", ingest([msi_6band]))
        assert not plan.tasks[0].startswith("TEMPORAL_")


class TestPlainImageFormats:
    """PS items 11-13: PNG/JPEG work, and claim nothing they cannot support."""

    def test_png_ingests_without_blocking(self, tmp_path):
        manifest = ingest([_write_plain_image(tmp_path / "scene.png", "PNG")])
        assert manifest.blocking_failures == []
        assert manifest.config == "SINGLE"

    def test_jpeg_ingests_without_blocking(self, tmp_path):
        manifest = ingest([_write_plain_image(tmp_path / "scene.jpg", "JPEG")])
        assert manifest.blocking_failures == []

    def test_a_png_answers_a_general_question(self, router, tmp_path):
        manifest = ingest([_write_plain_image(tmp_path / "scene.png", "PNG")])
        plan = router.route("What do you see in this image?", manifest)
        assert plan.tasks[0] in SINGLE_IMAGE_VISUAL

    def test_a_png_answers_a_presence_question(self, router, tmp_path):
        manifest = ingest([_write_plain_image(tmp_path / "scene.png", "PNG")])
        plan = router.route("Are there buildings?", manifest)
        assert plan.tasks[0] in SINGLE_IMAGE_VISUAL

    def test_a_png_claims_no_geospatial_capability(self, tmp_path):
        """The point of accepting it is honesty, not silence.

        CRS is UNKNOWN, the image is flagged ungeoreferenced, no
        band-arithmetic index is offered, and no sensor is guessed. Passing
        the check silently would have been the wrong fix: it would let a
        reader assume an output could be placed on a map.
        """
        manifest = ingest([_write_plain_image(tmp_path / "scene.png", "PNG")])
        image = manifest.images[0]
        assert image.crs == "UNKNOWN"
        assert image.georeferenced is False
        assert image.container_format == "PNG"
        assert image.sensor_guess is None
        for index in ("ndvi", "ndwi", "mndwi", "ndbi", "sigma0"):
            assert manifest.index_availability[index] is False

        crs = next(c for c in manifest.checks if c.name == "crs_present")
        assert crs.status == "WARN"
        assert "no geospatial metadata" in crs.message

    def test_the_trace_does_not_report_a_gsd_nobody_measured(self, tmp_path):
        """GDAL's identity transform reads as 1.0 m; that is not a measurement."""
        from satquery.controller.pipeline import Controller

        trace = Controller().run(
            [_write_plain_image(tmp_path / "scene.png", "PNG")],
            "Is this urban or rural?",
        )
        assert trace.ingest.images[0]["gsd_m"] is None
        assert trace.ingest.images[0]["georeferenced"] is False

    def test_an_rgba_png_does_not_advertise_ndvi_from_its_alpha_channel(
        self, tmp_path
    ):
        """A 4-channel PNG is RGB + opacity, not RGB + near-infrared.

        Band-count inference called it MSI and the positional fallback named
        band 4 NIR, so `index_availability` advertised ndvi and ndwi as
        computable - from an opacity channel. Screenshots and any export with
        transparency are 4-channel, so this is the common case, not an edge
        one.
        """
        path = tmp_path / "rgba.png"
        rgb = (np.random.default_rng(4).random((3, 128, 128)) * 255).astype("uint8")
        rgba = np.concatenate([rgb, np.full((1, 128, 128), 255, dtype="uint8")])
        with rasterio.open(
            path, "w", driver="PNG", height=128, width=128, count=4, dtype="uint8"
        ) as dst:
            dst.write(rgba)

        manifest = ingest([path])
        image = manifest.images[0]
        assert image.bands == ["RED", "GREEN", "BLUE"]
        assert manifest.index_availability["ndvi"] is False
        assert manifest.index_availability["ndwi"] is False
        # Dropping a channel is disclosed, not silent.
        assert image.modality_evidence.get("alpha_band_dropped") is True

    def test_a_png_reaches_the_model_with_its_channels_in_order(self, tmp_path):
        """The image the model sees must be the image that was uploaded.

        GDAL reports band 1 of a PNG as `red`, but with no band descriptions
        the fallback assumed the GeoTIFF convention [BLUE, GREEN, RED], so
        `to_rgb_preview` looked "RED" up at index 3 and handed the VQA model a
        channel-reversed picture. Measured before the fix: the preview's red
        channel correlated +1.000 with the source's BLUE.
        """
        from satquery.tools.imaging import to_rgb_preview

        # Three distinguishable, non-flat channels. Flat ones are stretched to
        # mid-grey, which would hide a swap rather than expose it.
        ramp_x = np.tile(np.linspace(0, 255, 128, dtype="uint8"), (128, 1))
        ramp_y = ramp_x.T
        middle = np.tile(np.linspace(60, 90, 128, dtype="uint8"), (128, 1))

        path = tmp_path / "probe.png"
        with rasterio.open(
            path, "w", driver="PNG", height=128, width=128, count=3, dtype="uint8"
        ) as dst:
            dst.write(np.stack([ramp_x, middle, ramp_y]))

        image = ingest([path]).images[0]
        assert image.bands == ["RED", "GREEN", "BLUE"]

        preview, provenance = to_rgb_preview(image)
        rendered = np.array(preview).astype(float)
        assert provenance["bands_shown"] == ["RED", "GREEN", "BLUE"]

        for channel, source in enumerate([ramp_x, middle, ramp_y]):
            correlation = np.corrcoef(
                rendered[:, :, channel].ravel(), source.astype(float).ravel()
            )[0, 1]
            assert correlation > 0.99, (
                f"preview channel {channel} does not match its source band"
            )

    def test_geotiff_handling_is_unchanged(self, msi_6band):
        manifest = ingest([msi_6band], mode=IngestMode.OPERATIONAL)
        image = manifest.images[0]
        assert manifest.blocking_failures == []
        assert image.georeferenced is True
        assert image.crs != "UNKNOWN"
        assert image.gsd_m == pytest.approx(10.0)
        assert manifest.index_availability["ndvi"] is True
        crs = next(c for c in manifest.checks if c.name == "crs_present")
        assert crs.status == "PASS"

    def test_a_geotiff_without_a_crs_is_still_a_defect(self, no_crs_raster):
        """The relaxation is per container format, not a blanket one."""
        manifest = ingest([no_crs_raster], mode=IngestMode.OPERATIONAL)
        assert "crs_present" in manifest.blocking_failures
