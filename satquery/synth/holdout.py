"""Hand-written out-of-template queries for honest classifier evaluation.

The synthetic bank's own test split measures template memorisation, not
generalisation: held-out examples are slot-variants of templates the model
trained on, so it scores ~100% and tells you almost nothing. These queries are
written by hand in deliberately different register - contractions, slang,
missing punctuation, indirect phrasing - to estimate real-world accuracy.

PROVENANCE MATTERS, so it is recorded here:

* `TUNED_HOLDOUT` was used to diagnose weak spots, after which the template
  bank was broadened. Its score is therefore optimistic and must NOT be quoted
  as a generalisation estimate.
* `CLEAN_HOLDOUT` was written afterwards and has never informed the templates.
  It is the honest number.

Neither set is large enough for a tight confidence interval. They are a smoke
test against template overfitting, not a benchmark. The real evaluation is
against the prescribed benchmarks in Phase 2.
"""

from __future__ import annotations

from .query_bank import QueryExample

# Used to diagnose weaknesses; templates were then expanded. Optimistic.
TUNED_HOLDOUT: list[tuple[str, str]] = [
    ("Whats the count of boats down by the docks?", "SINGLE_VQA"),
    ("any idea if theres a runway here", "SINGLE_VQA"),
    ("tell me roughly what share of this is trees", "SINGLE_VQA"),
    ("give me a rundown of this picture", "SINGLE_CAPTION"),
    ("in a sentence or two, whats going on in this shot", "SINGLE_CAPTION"),
    ("sum up the imagery", "SINGLE_CAPTION"),
    ("stick a box on every warehouse", "SINGLE_GROUND"),
    ("i want the pixel positions of the jetties", "SINGLE_GROUND"),
    ("whereabouts is the dam", "SINGLE_GROUND"),
    ("split the scene into surface categories", "SINGLE_LANDCOVER"),
    ("i need a thematic map of ground types", "SINGLE_LANDCOVER"),
    ("what share of each terrain category is here", "SINGLE_LANDCOVER"),
    ("stick the radar and the optical together and tell me about the crops",
     "XMODAL_JOINT_EXTRACT"),
    ("does the microwave sensor see anything the camera missed",
     "XMODAL_JOINT_EXTRACT"),
    ("run a joint analysis across both sensors", "XMODAL_JOINT_EXTRACT"),
    ("talk me through the differences across the two passes",
     "TEMPORAL_CHANGE_DESC"),
    ("whats moved since last time", "TEMPORAL_CHANGE_DESC"),
    ("write up how the area shifted", "TEMPORAL_CHANGE_DESC"),
    ("how many more houses are there now than before", "TEMPORAL_CHANGE_VQA"),
    ("did the lake shrink", "TEMPORAL_CHANGE_VQA"),
    ("put a number on the forest loss", "TEMPORAL_CHANGE_VQA"),
    ("i want a raster showing altered pixels", "TEMPORAL_CHANGE_MAP"),
    ("spit out a difference layer for gis", "TEMPORAL_CHANGE_MAP"),
    ("draw the footprint of what moved", "TEMPORAL_CHANGE_MAP"),
    ("hey", "CLARIFY_OR_ABSTAIN"),
    ("um", "CLARIFY_OR_ABSTAIN"),
    ("just have a look and let me know", "CLARIFY_OR_ABSTAIN"),
]

# Written after the templates were finalised. Never used for tuning.
CLEAN_HOLDOUT: list[tuple[str, str]] = [
    # SINGLE_VQA
    ("is that a power station in the corner", "SINGLE_VQA"),
    ("about how much of this is under water", "SINGLE_VQA"),
    ("tally up the silos", "SINGLE_VQA"),
    ("are the fields bigger than the woods here", "SINGLE_VQA"),
    # SINGLE_CAPTION
    ("just tell me what im looking at", "SINGLE_CAPTION"),
    ("put this scene into words", "SINGLE_CAPTION"),
    ("a short blurb about this frame please", "SINGLE_CAPTION"),
    # SINGLE_GROUND
    ("mark up wherever the quarries sit", "SINGLE_GROUND"),
    ("i need bounding boxes for the greenhouses", "SINGLE_GROUND"),
    ("which corner has the airstrip", "SINGLE_GROUND"),
    # SINGLE_LANDCOVER
    ("bucket every pixel into a cover type", "SINGLE_LANDCOVER"),
    ("i want the class breakdown for this tile", "SINGLE_LANDCOVER"),
    ("produce the thematic layer of surface classes", "SINGLE_LANDCOVER"),
    # XMODAL_JOINT_EXTRACT
    ("pull the two sensors together and check the embankments",
     "XMODAL_JOINT_EXTRACT"),
    ("what extra detail does the backscatter give over the photo",
     "XMODAL_JOINT_EXTRACT"),
    ("read the optical and the sar side by side for the settlements",
     "XMODAL_JOINT_EXTRACT"),
    # TEMPORAL_CHANGE_DESC
    ("give me the story of how this plot evolved", "TEMPORAL_CHANGE_DESC"),
    ("in prose, how do the two dates differ", "TEMPORAL_CHANGE_DESC"),
    ("recap whats different now versus then", "TEMPORAL_CHANGE_DESC"),
    # TEMPORAL_CHANGE_VQA
    ("by how much did the built area go up", "TEMPORAL_CHANGE_VQA"),
    ("were any orchards cleared between the passes", "TEMPORAL_CHANGE_VQA"),
    ("quantify the shoreline retreat", "TEMPORAL_CHANGE_VQA"),
    # TEMPORAL_CHANGE_MAP
    ("burn the changed pixels into a geotiff", "TEMPORAL_CHANGE_MAP"),
    ("i need the change polygons as a layer", "TEMPORAL_CHANGE_MAP"),
    ("render where things differ as a mask", "TEMPORAL_CHANGE_MAP"),
    # CLARIFY_OR_ABSTAIN
    ("ok", "CLARIFY_OR_ABSTAIN"),
    ("whats up with it", "CLARIFY_OR_ABSTAIN"),
    ("do your thing", "CLARIFY_OR_ABSTAIN"),
    ("thoughts?", "CLARIFY_OR_ABSTAIN"),
]


def as_examples(pairs: list[tuple[str, str]]) -> list[QueryExample]:
    return [QueryExample(text=t, task=task) for t, task in pairs]  # type: ignore[arg-type]


def clean_holdout() -> list[QueryExample]:
    """The honest generalisation set - never used to tune the templates."""
    return as_examples(CLEAN_HOLDOUT)


def tuned_holdout() -> list[QueryExample]:
    """Diagnostic set; its score is optimistic by construction."""
    return as_examples(TUNED_HOLDOUT)
