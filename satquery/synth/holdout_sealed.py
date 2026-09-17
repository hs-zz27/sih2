"""Sealed routing holdout: written before the query-bank revision, never tuned.

Why a third set
---------------
`holdout.CLEAN_HOLDOUT` (n=29) is the only honest routing number the project
had, and it has two problems. It is too small to move: one query is 3.4
points, so the documented 0.6552 -> 0.5862 drop was three flips. And it
carries no input configuration, so it can only score the *raw* classifier,
while the system never uses the raw prediction: the router restricts it to the
tasks the inputs make legal, then applies a confidence gate with a
configuration default. The number a user experiences is that second one, and
nothing measured it.

This set fixes both. Each query is paired with the input configuration it
would realistically arrive with, so `evaluation/routing_eval.py` can score raw
and system routing on the same items.

Provenance - read before quoting a number
------------------------------------------
* Written in one pass and committed **before** any change to
  `query_bank.py` in the same revision. The commit order is the evidence.
* It must never be used to choose, add or edit templates. If it ever is,
  move it into the tuned category and write a new sealed set.
* Seven queries per task. Register varies on purpose: field-report phrasing,
  Indian English ("kindly", "pls"), sensor names from the PS (Cartosat,
  RISAT), lowercase and missing punctuation.
* Some items are easy - close to how the templates phrase a task. That is
  deliberate: a holdout of only adversarial phrasings would understate real
  accuracy as badly as a template split overstates it.
"""

from __future__ import annotations

from dataclasses import dataclass

from satquery.contracts.plan import TaskID


@dataclass(frozen=True)
class RoutedQuery:
    text: str
    config: str
    task: TaskID


S, X, B = "SINGLE", "CROSSMODAL_PAIR", "BITEMPORAL_PAIR"

SEALED_HOLDOUT: list[RoutedQuery] = [
    # SINGLE_VQA
    RoutedQuery("how many cooling towers can u see", S, "SINGLE_VQA"),
    RoutedQuery("is there any standing water in the fields", S, "SINGLE_VQA"),
    RoutedQuery("roughly what percent of the tile is paved", S, "SINGLE_VQA"),
    RoutedQuery("kindly confirm whether a railway line passes through this area", S, "SINGLE_VQA"),
    RoutedQuery("are the ponds bigger than the canal", S, "SINGLE_VQA"),
    RoutedQuery("any sign of a helipad on the hospital roof", S, "SINGLE_VQA"),
    RoutedQuery("how many brick kilns are operating here", S, "SINGLE_VQA"),
    # SINGLE_CAPTION
    RoutedQuery("describe the scene for my field report", S, "SINGLE_CAPTION"),
    RoutedQuery("write a caption for this satellite tile", S, "SINGLE_CAPTION"),
    RoutedQuery("whats this image showing overall", S, "SINGLE_CAPTION"),
    RoutedQuery("give me a one paragraph overview of the area", S, "SINGLE_CAPTION"),
    RoutedQuery("summarise the landscape in plain english", S, "SINGLE_CAPTION"),
    RoutedQuery("narrate what's visible in this photo", S, "SINGLE_CAPTION"),
    RoutedQuery("a brief description of the town in this image would help", S, "SINGLE_CAPTION"),
    # SINGLE_GROUND
    RoutedQuery("locate the reservoir and draw a box around it", S, "SINGLE_GROUND"),
    RoutedQuery("point out where the stadium is", S, "SINGLE_GROUND"),
    RoutedQuery("highlight the flyover mentioned in the complaint", S, "SINGLE_GROUND"),
    RoutedQuery("find the petrol pumps and show their positions", S, "SINGLE_GROUND"),
    RoutedQuery("where exactly is the rail yard in this image", S, "SINGLE_GROUND"),
    RoutedQuery("outline the temple complex", S, "SINGLE_GROUND"),
    RoutedQuery("give coordinates of the biggest warehouse", S, "SINGLE_GROUND"),
    # SINGLE_LANDCOVER
    RoutedQuery("classify this scene into land use categories", S, "SINGLE_LANDCOVER"),
    RoutedQuery("what is the percentage split between cropland, forest and built-up", S, "SINGLE_LANDCOVER"),
    RoutedQuery("make a land use land cover map", S, "SINGLE_LANDCOVER"),
    RoutedQuery("segment the tile by surface type", S, "SINGLE_LANDCOVER"),
    RoutedQuery("give me an lulc breakdown", S, "SINGLE_LANDCOVER"),
    RoutedQuery("label each region as water, vegetation, soil or urban", S, "SINGLE_LANDCOVER"),
    RoutedQuery("which cover classes are present and in what proportion", S, "SINGLE_LANDCOVER"),
    # XMODAL_JOINT_EXTRACT
    RoutedQuery("combine the radar with the optical to map flooded fields", X, "XMODAL_JOINT_EXTRACT"),
    RoutedQuery("use sar where clouds block the optical and find the water", X, "XMODAL_JOINT_EXTRACT"),
    RoutedQuery("fuse both sensors to detect built-up land", X, "XMODAL_JOINT_EXTRACT"),
    RoutedQuery("what does the radar reveal that the optical image hides", X, "XMODAL_JOINT_EXTRACT"),
    RoutedQuery("cross check the settlements using optical and sar together", X, "XMODAL_JOINT_EXTRACT"),
    RoutedQuery("merge the microwave and multispectral data for this region", X, "XMODAL_JOINT_EXTRACT"),
    RoutedQuery("use both the risat and cartosat images to find wetlands", X, "XMODAL_JOINT_EXTRACT"),
    # TEMPORAL_CHANGE_DESC
    RoutedQuery("describe how this area changed between the two dates", B, "TEMPORAL_CHANGE_DESC"),
    RoutedQuery("what's different in the second image compared to the first", B, "TEMPORAL_CHANGE_DESC"),
    RoutedQuery("explain the changes over time in this region", B, "TEMPORAL_CHANGE_DESC"),
    RoutedQuery("summarise the development that happened between the acquisitions", B, "TEMPORAL_CHANGE_DESC"),
    RoutedQuery("compare the before and after images and describe the differences", B, "TEMPORAL_CHANGE_DESC"),
    RoutedQuery("what has happened to this place since the earlier pass", B, "TEMPORAL_CHANGE_DESC"),
    RoutedQuery("give an account of the changes along the riverbank", B, "TEMPORAL_CHANGE_DESC"),
    # TEMPORAL_CHANGE_VQA
    RoutedQuery("did the forest cover decrease between the two images", B, "TEMPORAL_CHANGE_VQA"),
    RoutedQuery("how many new buildings came up", B, "TEMPORAL_CHANGE_VQA"),
    RoutedQuery("has the lake grown or shrunk", B, "TEMPORAL_CHANGE_VQA"),
    RoutedQuery("by what percentage did farmland reduce", B, "TEMPORAL_CHANGE_VQA"),
    RoutedQuery("is there more construction now than before", B, "TEMPORAL_CHANGE_VQA"),
    RoutedQuery("was the road widened between the dates", B, "TEMPORAL_CHANGE_VQA"),
    RoutedQuery("how much area was lost to the flood", B, "TEMPORAL_CHANGE_VQA"),
    # TEMPORAL_CHANGE_MAP
    RoutedQuery("generate a change detection mask", B, "TEMPORAL_CHANGE_MAP"),
    RoutedQuery("show me a map of changed areas", B, "TEMPORAL_CHANGE_MAP"),
    RoutedQuery("export the change layer as geotiff", B, "TEMPORAL_CHANGE_MAP"),
    RoutedQuery("create a binary change map between the two scenes", B, "TEMPORAL_CHANGE_MAP"),
    RoutedQuery("highlight changed pixels on a map", B, "TEMPORAL_CHANGE_MAP"),
    RoutedQuery("produce the difference raster for gis", B, "TEMPORAL_CHANGE_MAP"),
    RoutedQuery("mark all areas of change as polygons", B, "TEMPORAL_CHANGE_MAP"),
    # CLARIFY_OR_ABSTAIN
    RoutedQuery("hello", S, "CLARIFY_OR_ABSTAIN"),
    RoutedQuery("can you help", S, "CLARIFY_OR_ABSTAIN"),
    RoutedQuery("?", S, "CLARIFY_OR_ABSTAIN"),
    RoutedQuery("what's the weather in delhi tomorrow", S, "CLARIFY_OR_ABSTAIN"),
    RoutedQuery("tell me something", B, "CLARIFY_OR_ABSTAIN"),
    RoutedQuery("analyse", X, "CLARIFY_OR_ABSTAIN"),
    RoutedQuery("nice", S, "CLARIFY_OR_ABSTAIN"),
]


def sealed_holdout() -> list[RoutedQuery]:
    return list(SEALED_HOLDOUT)
