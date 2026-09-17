"""Deterministic narrative synthesis.

Turns structured tool output into a human-readable sentence. Every number in
the text comes from the deterministic index engine, so the narrative cannot
assert something the physics does not support - which is the property the
entailment gate (task 3.5) will later check automatically.

This is deliberately template-based rather than generative. A generated
summary of already-structured numbers adds a hallucination risk for no
benefit; the learned models are for tasks that genuinely need language
understanding.
"""

from __future__ import annotations

# Fraction of the scene above an index threshold, below which a class is not
# worth mentioning at all.
MENTION_THRESHOLD = 0.05

INDEX_MEANING = {
    "ndvi": "vegetation",
    "ndwi": "water",
    "mndwi": "water",
    "ndbi": "built-up land",
    "builtup_proxy": "likely built-up land",
}


def _percent(fraction: float) -> str:
    return f"{fraction * 100:.0f}%"


def _area(square_metres: float) -> str:
    """A ground area, in the unit that keeps it readable.

    Two decimals of a square kilometre is 10,000 m2 - finer than any
    threshold-derived fraction deserves, and coarse enough not to imply the
    boundary between "vegetation" and "not vegetation" is known to the pixel.
    """
    if square_metres >= 1_000_000.0:
        return f"{square_metres / 1_000_000.0:.2f} km2"
    return f"{round(square_metres, -3):,.0f} m2"


def describe_indices(index_payload: dict, scene_area_m2: float | None = None) -> str:
    """One sentence describing scene composition from index statistics.

    `scene_area_m2` turns each fraction into a ground area. It is optional
    because it is only *knowable* for a projected CRS - it comes from
    `ImageMeta.ground_extent_m`, which is withheld where `gsd_m` is the
    degrees-to-metres approximation. Without it the sentence reports
    percentages alone, exactly as before.

    The areas inherit the overlap the fractions have: they are per-index
    thresholds measured independently, so they do not partition the scene
    and must not be added up. The closing clause already says so, and it is
    load-bearing once these are areas rather than percentages - a reader who
    would not sum percentages might well try to sum square kilometres.
    """
    indices = index_payload.get("indices", {})
    parts: list[str] = []

    # MNDWI supersedes NDWI when both exist: it is the better water index.
    water_key = "mndwi" if "mndwi" in indices else "ndwi"
    ordered = ["ndvi", water_key, "ndbi", "builtup_proxy"]

    seen_meanings: set[str] = set()
    for key in ordered:
        entry = indices.get(key)
        if not entry:
            continue
        meaning = INDEX_MEANING.get(key)
        if meaning is None or meaning in seen_meanings:
            continue
        fraction = entry.get("fraction_above_threshold")
        if fraction is None or fraction < MENTION_THRESHOLD:
            continue
        seen_meanings.add(meaning)
        described = f"{_percent(fraction)} {meaning}"
        if scene_area_m2:
            described += f" ({_area(fraction * scene_area_m2)})"
        parts.append(described)

    if not parts:
        return "No dominant land-cover class exceeded its detection threshold."

    if len(parts) == 1:
        return f"Index thresholds indicate {parts[0]} coverage."

    body = ", ".join(parts[:-1]) + f" and {parts[-1]}"
    # These fractions come from independent per-index thresholds, so they
    # overlap and will not sum to 100%. Saying "the scene is X% A and Y% B"
    # would imply a partition that the indices do not provide, so the
    # wording states the overlap explicitly instead.
    return (
        f"Index thresholds indicate {body}. These classes are measured "
        "independently and may overlap."
    )


def describe_labels(labels: list[str]) -> str:
    if not labels:
        return ""
    readable = [str(label).replace("_", " ") for label in labels]
    if len(readable) == 1:
        return f"Detected land-cover class: {readable[0]}."
    return "Detected land-cover classes: " + ", ".join(readable) + "."


def synthesise_answer(
    task: str,
    step_outputs: list[dict],
    index_payload: dict,
    artifacts: list[str] | None = None,
    scene_area_m2: float | None = None,
) -> str:
    """Build an answer for tasks whose tools return structure, not prose.

    `artifacts` is the list of files the run actually produced. It is a
    parameter rather than an assumption because the lite profile (task 3.10)
    sheds the learned tools, and this function previously asserted "see the
    exported raster artifact" for `TEMPORAL_CHANGE_MAP` whether or not a
    raster existed. In lite, none did.
    """
    labels: list[str] = []
    boxes: list = []
    for out in step_outputs:
        labels.extend(out.get("labels", []) or [])
        boxes.extend(out.get("bounding_boxes", []) or [])

    if task == "SINGLE_LANDCOVER":
        pieces = [describe_labels(labels), describe_indices(index_payload, scene_area_m2)]
        return " ".join(p for p in pieces if p)

    if task == "SINGLE_GROUND":
        if boxes:
            noun = "region" if len(boxes) == 1 else "regions"
            return f"Localised {len(boxes)} matching {noun}."
        return "No matching region was localised."

    if task == "SINGLE_CAPTION":
        # landcover_v1 is in this task's plan (see configs/capability_matrix.yaml),
        # so any class it was confident enough to assert belongs in the prose.
        # It usually asserts none - recall is 0.25% at the measured threshold -
        # and `describe_labels` returns "" for that, which is the honest result.
        pieces = [describe_labels(labels), describe_indices(index_payload, scene_area_m2)]
        return " ".join(p for p in pieces if p)

    if task == "TEMPORAL_CHANGE_DESC":
        return describe_indices(index_payload, scene_area_m2)

    if task == "TEMPORAL_CHANGE_MAP":
        # `artifacts` also carries the index engine's own COGs, so a bare
        # truthiness check passed in lite even though no change raster
        # existed. The claim has to be about the change mask specifically.
        if any("change" in str(a).lower() for a in artifacts or []):
            return "Produced a change mask; see the exported raster artifact."
        return (
            "No change raster was produced - the change tool did not run in "
            "this profile. " + describe_indices(index_payload, scene_area_m2)
        ).strip()

    if task == "XMODAL_JOINT_EXTRACT":
        # Previously fell through to "", which reached the user as an empty
        # answer in lite. The indices are computed from whichever modality
        # supplied the bands, so there is always something measured to say.
        return (
            "Optical-SAR fusion did not run in this profile, so this "
            "describes the optical bands alone. " + describe_indices(index_payload, scene_area_m2)
        ).strip()

    # VQA-style tasks are answered by the model itself. An empty string here
    # is a signal to the caller, not an answer - the executor turns it into a
    # named abstention rather than showing it.
    return ""


# Tasks whose synthesised prose is purely *additive* measurement, and so may be
# appended to a tool's own answer rather than only replacing a missing one.
#
# The list is deliberately short. `synthesise_answer` also covers
# TEMPORAL_CHANGE_MAP and XMODAL_JOINT_EXTRACT, but the strings it returns for
# those explain why a tool did NOT run ("Optical-SAR fusion did not run in this
# profile"). Appending that to a real tool answer would contradict it. Those
# tasks keep the replace-only behaviour.
#
# VQA tasks are absent because `synthesise_answer` returns "" for them, which
# already means "the model's own answer stands alone" - a question like "did
# vegetation increase?" wants a direction, not a paragraph of scene statistics.
ENRICHABLE_TASKS = frozenset(
    {"SINGLE_CAPTION", "SINGLE_LANDCOVER", "TEMPORAL_CHANGE_DESC", "SINGLE_GROUND"}
)


def _terminated(text: str) -> str:
    """Give a fragment a full stop so it can be joined to another sentence.

    The scene captioner emits `" ".join(words)` with no terminal punctuation,
    so concatenating it directly produced "a river runs through the scene Index
    thresholds indicate ...".
    """
    text = text.strip()
    if not text or text[-1] in ".!?":
        return text
    return text + "."


def describe_georeferencing(
    georeferenced: bool, container_format: str | None = None
) -> str:
    """Disclose a missing CRS, because it bounds what the answer can contain.

    A PNG or JPEG cannot carry georeferencing at all, so for those inputs there
    is no latitude, longitude, ground extent or area in metres to report - the
    `gsd_m` on the manifest is a placeholder from GDAL's identity transform,
    not a measurement (see satquery/ingest/reader.py). Saying so is better than
    a reader assuming the system simply declined to mention where the scene is.
    """
    if georeferenced:
        return ""
    container = f" ({container_format})" if container_format else ""
    return (
        f"This file{container} carries no georeferencing, so no coordinates, "
        "ground extent or areas in metres can be reported for it."
    )


def _coordinate(value: float, positive: str, negative: str) -> str:
    """A signed degree as a hemisphere-lettered coordinate.

    Four decimals is about 11 m at the equator - finer than the GSD of any
    product this system ingests, and coarse enough not to imply the footprint
    is known to the centimetre.
    """
    hemisphere = positive if value >= 0 else negative
    return f"{abs(value):.4f}° {hemisphere}"


def _distance(metres: float) -> str:
    if metres >= 1000.0:
        return f"{metres / 1000.0:.1f} km"
    return f"{round(metres, -1):.0f} m"


def describe_location(
    georeferenced: bool,
    container_format: str | None = None,
    centroid: tuple[float, float] | None = None,
    extent_m: tuple[float, float] | None = None,
) -> str:
    """One sentence saying where the scene is, or why that cannot be said.

    Every number here is transformed from the container's own CRS and
    geotransform at ingest (`ingest/reader._footprint`). Nothing is inferred:
    a file that does not carry a footprint produces the disclosure instead,
    and a file that carries one but whose CRS measures in degrees gets the
    centre without a ground extent, because `gsd_m` is only exact for a
    projected CRS.
    """
    if not georeferenced:
        return describe_georeferencing(georeferenced, container_format)
    if centroid is None:
        # Georeferenced, but the footprint would not transform. Saying
        # nothing is right: the alternative is a coordinate we do not have.
        return ""

    latitude, longitude = centroid
    sentence = (
        f"The scene is centred at {_coordinate(latitude, 'N', 'S')}, "
        f"{_coordinate(longitude, 'E', 'W')}"
    )
    if extent_m:
        sentence += (
            f" and covers about {_distance(extent_m[0])} by "
            f"{_distance(extent_m[1])} on the ground"
        )
    return sentence + "."


def describe_place(place) -> str:
    """Name the region a scene falls in, hedging where the data hedges.

    Takes a `geo.gazetteer.Place`. Duck-typed rather than imported so this
    module keeps its "no imports" property - every other function here turns
    numbers into words and needs nothing from the rest of the package.

    A label whose 3x3 window disagreed is reported as a nearest match, never
    as fact. The distinction matters most exactly where it is most tempting
    to drop: a scene on a national border is precisely the case where naming
    one country confidently is wrong.
    """
    if place is None or not place:
        return ""

    ambiguous = getattr(place, "ambiguous", frozenset())
    parts: list[str] = []

    if place.country:
        if "country" in ambiguous:
            parts.append(
                f"close to a national boundary in the reference data - the "
                f"nearest match is {place.country}"
            )
        else:
            parts.append(f"in {place.country}")

    if place.climate:
        zone = f"Koppen climate zone {place.climate}"
        if place.climate_description:
            zone += f" ({place.climate_description})"
        if "climate" in ambiguous:
            zone = f"near the edge of {zone}"
        parts.append(zone)

    if not parts:
        return ""
    return "The scene lies " + ", ".join(parts) + "."


def compose_answer(
    task: str,
    tool_answer: str,
    step_outputs: list[dict],
    index_payload: dict,
    artifacts: list[str] | None = None,
    georeferenced: bool = True,
    container_format: str | None = None,
    centroid: tuple[float, float] | None = None,
    extent_m: tuple[float, float] | None = None,
    place=None,
) -> str:
    """Combine a tool's prose with the measured description of the scene.

    The executor used to let the last answer-bearing tool win outright, and
    only synthesised prose when *no* tool had produced any. For SINGLE_CAPTION
    that meant `index_engine_v1` and `landcover_v1` both ran, both produced
    numbers, and both were then discarded from the answer because
    `caption_v1` - last in the plan - had emitted one short sentence. The
    measurements survived in the trace and never reached the reader.

    Composition keeps the captioner's sentence as the opening clause and adds
    what was actually measured after it, so the answer says what the model saw
    *and* what the physics found.
    """
    # The scene's own footprint area, which is what turns a fraction into
    # square kilometres. None for a geographic CRS, where `extent_m` is
    # withheld because `gsd_m` is approximate there - so those answers keep
    # percentages and gain no areas, rather than gaining wrong ones.
    scene_area_m2 = extent_m[0] * extent_m[1] if extent_m else None
    synthesised = synthesise_answer(
        task, step_outputs, index_payload, artifacts=artifacts,
        scene_area_m2=scene_area_m2,
    )
    tool_answer = tool_answer.strip()

    if task not in ENRICHABLE_TASKS or not synthesised:
        # Replace-only, exactly as before: the tool's answer if it produced
        # one, otherwise whatever the synthesiser could say.
        return tool_answer or synthesised

    parts = [_terminated(tool_answer), _terminated(synthesised)]
    parts.append(
        describe_location(georeferenced, container_format, centroid, extent_m)
    )
    parts.append(describe_place(place))
    return " ".join(p for p in parts if p)
