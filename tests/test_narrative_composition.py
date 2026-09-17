"""Answer composition tests.

The executor used to let the last answer-bearing tool in the plan overwrite
`final_answer` outright, and synthesised prose only when *no* tool had
produced any. For SINGLE_CAPTION that meant `index_engine_v1` and
`landcover_v1` both ran, both produced numbers, and both were discarded
because `caption_v1` - last in the plan - emitted one short sentence. These
tests pin the composition that replaced it, and, more importantly, pin the
places it must NOT reach.
"""

from __future__ import annotations

from satquery.synth.narrative import (
    ENRICHABLE_TASKS,
    compose_answer,
    describe_georeferencing,
    describe_indices,
    describe_location,
    describe_place,
)

from satquery.geo import Place

# 82% water, 18% vegetation - the shape the index engine emits.
INDEX_PAYLOAD = {
    "indices": {
        "ndvi": {"fraction_above_threshold": 0.18},
        "mndwi": {"fraction_above_threshold": 0.82},
    }
}


class TestEnrichment:
    def test_a_caption_keeps_the_measured_description(self):
        answer = compose_answer(
            "SINGLE_CAPTION", "a river runs through the scene", [], INDEX_PAYLOAD
        )
        assert answer.startswith("a river runs through the scene.")
        assert "82% water" in answer
        assert "18% vegetation" in answer

    def test_an_unpunctuated_caption_is_terminated_before_joining(self):
        """The captioner emits `" ".join(words)` with no full stop, which
        concatenated into "...through the scene Index thresholds indicate"."""
        answer = compose_answer(
            "SINGLE_CAPTION", "a river runs through the scene", [], INDEX_PAYLOAD
        )
        assert "scene Index" not in answer

    def test_an_already_punctuated_caption_gains_no_second_full_stop(self):
        answer = compose_answer(
            "SINGLE_CAPTION", "A river runs through the scene.", [], INDEX_PAYLOAD
        )
        assert ".." not in answer

    def test_landcover_classes_reach_a_caption_answer(self):
        """`landcover_v1` is in the SINGLE_CAPTION plan; its labels were
        computed and then dropped from the prose."""
        answer = compose_answer(
            "SINGLE_CAPTION",
            "a coastal scene",
            [{"labels": ["Marine waters"]}],
            INDEX_PAYLOAD,
        )
        assert "Marine waters" in answer

    def test_a_missing_tool_answer_still_synthesises(self):
        answer = compose_answer("SINGLE_CAPTION", "", [], INDEX_PAYLOAD)
        assert "82% water" in answer
        assert not answer.startswith(".")


class TestScoping:
    """The guard that keeps composition out of tasks it would damage."""

    def test_a_vqa_answer_is_left_alone(self):
        """"Did vegetation increase?" wants a direction, not scene
        statistics. `synthesise_answer` returns "" for VQA tasks, and that
        is the signal to leave the model's own answer standing alone."""
        answer = compose_answer(
            "TEMPORAL_CHANGE_VQA", "Vegetation increased.", [], INDEX_PAYLOAD
        )
        assert answer == "Vegetation increased."

    def test_change_map_does_not_append_a_contradicting_sentence(self):
        """TEMPORAL_CHANGE_MAP's synthesised string explains why a tool did
        NOT run. Appending it to a real tool answer would contradict it, so
        that task stays replace-only."""
        answer = compose_answer(
            "TEMPORAL_CHANGE_MAP",
            "The reservoir shrank along its western shore.",
            [],
            INDEX_PAYLOAD,
            artifacts=["change_mask"],
        )
        assert answer == "The reservoir shrank along its western shore."
        assert "did not run in this profile" not in answer

    def test_xmodal_stays_replace_only(self):
        answer = compose_answer(
            "XMODAL_JOINT_EXTRACT", "Fused optical and SAR bands.", [], INDEX_PAYLOAD
        )
        assert answer == "Fused optical and SAR bands."

    def test_the_enrichable_set_is_the_additive_tasks_only(self):
        """Pinned deliberately. Adding a task here changes what every answer
        of that shape says, and the two exclusions above are the ones that
        would break if the set grew carelessly."""
        assert ENRICHABLE_TASKS == {
            "SINGLE_CAPTION",
            "SINGLE_LANDCOVER",
            "TEMPORAL_CHANGE_DESC",
            # Joined the set when grounding answers gained the scene
            # coordinate: its synthesised sentence IS the whole answer, so
            # enriching it appends location without displacing anything.
            "SINGLE_GROUND",
        }


class TestGeoreferencingDisclosure:
    def test_an_ungeoreferenced_file_says_so(self):
        """A PNG cannot carry a CRS, so there is no latitude, longitude or
        area in metres to report - `gsd_m` is GDAL's identity placeholder,
        not a measurement. Silence would read as a system that declined to
        mention where the scene is."""
        note = describe_georeferencing(False, "PNG")
        assert "PNG" in note
        assert "no georeferencing" in note

    def test_a_georeferenced_file_adds_nothing(self):
        assert describe_georeferencing(True, "GTiff") == ""

    def test_the_disclosure_reaches_a_composed_answer(self):
        answer = compose_answer(
            "SINGLE_CAPTION",
            "a coastal scene",
            [],
            INDEX_PAYLOAD,
            georeferenced=False,
            container_format="PNG",
        )
        assert "no georeferencing" in answer

    def test_no_coordinates_are_invented_for_a_georeferenced_file(self):
        """Composition adds measurement, never geography. Nothing in the
        answer path knows where the scene is."""
        answer = compose_answer(
            "SINGLE_CAPTION", "a coastal scene", [], INDEX_PAYLOAD
        )
        for word in ("latitude", "longitude", "°", "EPSG"):
            assert word not in answer


class TestLocation:
    """The location sentence. Every number in it is transformed from the
    file's own CRS and geotransform at ingest; nothing is inferred."""

    def test_a_located_scene_reports_its_centre_and_extent(self):
        sentence = describe_location(True, "GTiff", (18.0829, 75.0060), (1280.0, 1280.0))
        assert "18.0829° N" in sentence
        assert "75.0060° E" in sentence
        assert "1.3 km by 1.3 km" in sentence

    def test_southern_and_western_hemispheres_get_the_right_letters(self):
        """A bare minus sign in front of a coordinate is easy to misread,
        and easy to lose entirely when the text is reflowed."""
        sentence = describe_location(True, "GTiff", (-33.86, -70.66), None)
        assert "33.8600° S" in sentence
        assert "70.6600° W" in sentence
        assert "-" not in sentence

    def test_a_geographic_crs_gets_a_centre_but_no_extent(self):
        """`extent_m` is None for a geographic CRS because `gsd_m` is only
        approximate there. The centre is exact either way."""
        sentence = describe_location(True, "GTiff", (18.08, 75.01), None)
        assert "centred at" in sentence
        assert "on the ground" not in sentence

    def test_a_georeferenced_file_that_will_not_transform_says_nothing(self):
        """The alternative is a coordinate we do not have."""
        assert describe_location(True, "GTiff", None, None) == ""

    def test_an_ungeoreferenced_file_still_gets_the_disclosure(self):
        sentence = describe_location(False, "PNG", None, None)
        assert "no georeferencing" in sentence

    def test_a_sub_kilometre_scene_is_reported_in_metres(self):
        sentence = describe_location(True, "GTiff", (18.08, 75.01), (640.0, 320.0))
        assert "640 m by 320 m" in sentence

    def test_the_location_reaches_a_composed_answer(self):
        answer = compose_answer(
            "SINGLE_CAPTION",
            "a coastal scene",
            [],
            INDEX_PAYLOAD,
            centroid=(18.0829, 75.0060),
            extent_m=(1280.0, 1280.0),
        )
        assert "a coastal scene." in answer
        assert "82% water" in answer
        assert "18.0829° N" in answer

    def test_no_location_is_claimed_without_a_footprint(self):
        """compose_answer defaults carry no centroid, and must not invent
        one to fill the slot."""
        answer = compose_answer("SINGLE_CAPTION", "a coastal scene", [], INDEX_PAYLOAD)
        assert "centred at" not in answer
        for word in ("latitude", "longitude", "°"):
            assert word not in answer


class TestPlace:
    """`describe_place`. The gazetteer measures against a dataset; this
    renders what it found without adding confidence it did not report."""

    def test_a_country_and_climate_are_named(self):
        sentence = describe_place(
            Place(country="Lithuania", climate="Dfb",
                  climate_description="cold, no dry season, warm summer")
        )
        assert "in Lithuania" in sentence
        assert "Dfb" in sentence
        assert "cold, no dry season, warm summer" in sentence

    def test_an_ambiguous_country_is_hedged_not_asserted(self):
        """A scene on a national border is exactly the case where naming one
        country confidently is wrong."""
        sentence = describe_place(
            Place(country="Lithuania", ambiguous=frozenset({"country"}))
        )
        assert "nearest match is Lithuania" in sentence
        assert "in Lithuania" not in sentence

    def test_an_ambiguous_climate_is_hedged(self):
        sentence = describe_place(
            Place(climate="Dfb", ambiguous=frozenset({"climate"}))
        )
        assert "near the edge of" in sentence

    def test_a_climate_without_a_description_still_renders(self):
        assert "Dfb" in describe_place(Place(climate="Dfb"))

    def test_an_empty_place_says_nothing(self):
        assert describe_place(Place()) == ""

    def test_no_place_at_all_says_nothing(self):
        assert describe_place(None) == ""

    def test_the_place_reaches_a_composed_answer(self):
        answer = compose_answer(
            "SINGLE_CAPTION", "a coastal scene", [], INDEX_PAYLOAD,
            centroid=(54.9, 21.08),
            place=Place(country="Lithuania", climate="Dfb"),
        )
        assert "Lithuania" in answer
        assert "54.9000" in answer

    def test_no_region_is_named_without_a_gazetteer(self):
        """The default path - no gazetteer installed - must add nothing."""
        answer = compose_answer(
            "SINGLE_CAPTION", "a coastal scene", [], INDEX_PAYLOAD,
            centroid=(54.9, 21.08),
        )
        assert "The scene lies" not in answer


class TestAreas:
    """Ground areas alongside the percentages. Every one is
    `fraction_above_threshold` x the scene's own footprint area, so it is as
    measured as the fraction it came from."""

    def test_percentages_alone_without_a_scene_area(self):
        sentence = describe_indices(INDEX_PAYLOAD)
        assert "82% water" in sentence
        assert "km2" not in sentence and "m2" not in sentence

    def test_an_area_is_added_when_the_scene_area_is_known(self):
        # 1280 m x 1280 m = 1,638,400 m2; 82% of that is 1.34 km2.
        sentence = describe_indices(INDEX_PAYLOAD, 1280.0 * 1280.0)
        assert "82% water (1.34 km2)" in sentence

    def test_small_areas_stay_in_square_metres(self):
        sentence = describe_indices(INDEX_PAYLOAD, 250_000.0)
        assert "205,000 m2" in sentence
        assert "km2" not in sentence

    def test_the_overlap_caveat_survives(self):
        """It is load-bearing once these are areas: a reader who would not
        sum percentages might well try to sum square kilometres."""
        sentence = describe_indices(INDEX_PAYLOAD, 1280.0 * 1280.0)
        assert "measured independently and may overlap" in sentence

    def test_areas_reach_a_composed_answer(self):
        answer = compose_answer(
            "SINGLE_CAPTION", "a coastal scene", [], INDEX_PAYLOAD,
            extent_m=(1280.0, 1280.0),
        )
        assert "1.34 km2" in answer

    def test_no_area_without_a_ground_extent(self):
        """A geographic CRS withholds `extent_m` because `gsd_m` is
        approximate there, so those answers keep percentages and gain no
        areas rather than gaining wrong ones."""
        answer = compose_answer(
            "SINGLE_CAPTION", "a coastal scene", [], INDEX_PAYLOAD
        )
        assert "km2" not in answer


class TestGroundingIsEnriched:
    def test_a_grounding_answer_carries_the_scene_location(self):
        """"Where is the water" got "Localised 1 matching region." and
        nothing else - no coordinate, on an input that carried one."""
        answer = compose_answer(
            "SINGLE_GROUND", "", [{"bounding_boxes": [[0, 0, 1, 1]]}], INDEX_PAYLOAD,
            centroid=(18.0829, 75.0060),
        )
        assert "Localised 1 matching region." in answer
        assert "18.0829" in answer

    def test_grounding_does_not_gain_index_prose(self):
        """`synthesise_answer` returns only the localisation sentence for
        this task; enriching it adds location, not scene statistics."""
        answer = compose_answer(
            "SINGLE_GROUND", "", [{"bounding_boxes": []}], INDEX_PAYLOAD,
            centroid=(18.0829, 75.0060),
        )
        assert "Index thresholds" not in answer
