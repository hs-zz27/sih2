"""Synthetic query bank for the Tier-1 intent classifier (plan task 1.4).

Templates with filled slots, expanded into thousands of paraphrases. The task
boundaries here define what the classifier can learn, so each task's templates
are written to be *linguistically* distinguishable, not just semantically:
- CAPTION asks for a free description of the whole scene
- VQA asks a specific question with a specific answer
- GROUND asks for a location, expecting boxes back
- LANDCOVER asks for classification into classes
- XMODAL explicitly invokes both sensors
- CHANGE_DESC / CHANGE_VQA / CHANGE_MAP split by requested output form:
  prose, an answer to a question, or a raster mask
- CLARIFY_OR_ABSTAIN covers greetings, vagueness and impossible requests

Deterministic given a seed, so classifier metrics are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from satquery.contracts.plan import TaskID

FEATURES = [
    "buildings", "roads", "water bodies", "farmland", "forest", "urban area",
    "vegetation", "bridges", "ships", "aircraft", "solar panels", "rooftops",
    "rivers", "lakes", "crops", "bare soil", "settlements", "runways",
    "storage tanks", "harbours",
]

SINGULAR = [
    "a building", "a road", "a river", "a bridge", "a ship", "an aircraft",
    "a lake", "a runway", "a storage tank", "a harbour",
]

LANDCOVER_CLASSES = [
    "land cover", "land use", "surface types", "terrain classes",
    "land cover classes",
]

CHANGE_NOUNS = [
    "urban growth", "deforestation", "new construction", "water extent",
    "vegetation loss", "flooding", "land clearing", "built-up expansion",
]

# Scene descriptors, for the "is this X or Y?" shape a user actually types.
SCENE_TYPES = [
    "urban", "rural", "residential", "industrial", "agricultural",
    "coastal", "forested", "built-up", "developed", "natural",
    "densely built", "sparsely populated",
]

_VQA = [
    "How many {feature} are visible in this image?",
    "How many {feature} can you count?",
    "Is there {singular} in this scene?",
    "Are there any {feature} present?",
    "What is the approximate area covered by {feature}?",
    "Does this image contain {feature}?",
    "What proportion of the scene is {feature}?",
    "Which is more prevalent, {feature} or {feature2}?",
    "Is the {feature} area larger than the {feature2} area?",
    "What colour are the {feature} in this image?",
    "Count the {feature} in the image.",
    "Tell me how many {feature} there are.",
    "What is the total extent of {feature}?",
    "Can you confirm whether {feature} appear here?",
    "Roughly what percentage of this scene is {feature}?",
    "Any idea if there are {feature} here?",
    "What share of this is {feature}?",
    "Give me a number for the {feature}.",
    "Whats the count of {feature}?",
]

# General natural-language visual questions (PS-26167).
#
# Everything in _VQA above was written from benchmark phrasings, so the
# classifier learned that "a question about one image" means "how many X" or
# "what proportion is X". The open-ended shapes a user actually types - "is
# this urban or rural?", "what do you see?", "are there buildings?" - matched
# none of them and fell into CLARIFY_OR_ABSTAIN, whose own templates are
# content-free filler ("What is this?", "Anything interesting?") and are the
# nearest neighbour of any short unadorned question. The system refused to
# answer questions rs_vqa_v1 answers perfectly well.
#
# These are templates over the same slots as the rest of the bank, not a list
# of accepted questions. They expand into hundreds of paraphrases, and the
# word/char n-gram features generalise across the *shape*, which is what lets
# an unseen "does this look like an industrial estate?" reach the same class
# as the "does this look like {feature}?" seen in training. Nothing here
# enumerates the questions the product must accept.
_GENERAL_VISUAL = [
    "Is this {scene} or {scene2}?",
    "Is this an {scene} or {scene2} scene?",
    "Is this area {scene} or {scene2}?",
    "Would you call this {scene} or {scene2}?",
    "Is this scene {scene}?",
    "Is this area {scene}?",
    "Does this look {scene} to you?",
    "Does this look like {feature}?",
    "What kind of landscape is this?",
    "What kind of area is this?",
    "What sort of terrain is this?",
    "What do you see in this image?",
    "What do you see here?",
    "What is in this image?",
    "What can you tell me about this image?",
    "What is happening in this scene?",
    "Do you see any {feature}?",

    # Conversational open-ended questions. These read like captioning and are
    # not: "describe what you see" asks the assistant what it can make out,
    # and "describe the landscape" asks about something *in* the scene. Both
    # sat in _CAPTION, so an ordinary conversational question needed the
    # captioner loaded and returned the caption stub when it was not, even
    # though rs_vqa_v1 was loaded and could answer. The boundary that stays
    # with _CAPTION is whether the user asked for a caption of *the image* -
    # "caption this scene", "write a few sentences about this image" - which
    # is the mandatory second single-image capability and is untouched.
    "Describe what you see.",
    "Describe what you can see here.",
    "Describe what is visible here.",
    "Tell me what you see.",
    "Tell me what you can see in this scene.",
    "Describe the {feature}.",
    "Describe the {feature} you can see.",
    "Describe the landscape.",
    "Describe the {scene} area.",
    "Tell me about the {feature} here.",
    "Tell me about this area.",
    "What features are visible?",
    "What features can you make out?",
    "Give me an overview of this image.",
    "Give me a rundown of this picture.",
    "Narrate what can be seen here.",
    "Whats going on in this shot?",
    "How dense is the {feature} here?",
    "How built up is this area?",
    "Is this a densely developed area?",
    "What is unusual in this image?",
    "Is there anything unusual in this scene?",
]

# A request to PRODUCE A CAPTION: the object is the image itself, and what is
# wanted back is a written artifact about it - a caption, a summary, a few
# sentences. See _GENERAL_VISUAL for the boundary against a conversational
# "what do you see?", which asks the same model-shaped question but is not a
# request for a caption and must not need the captioner to be loaded.
_CAPTION = [
    "Describe this image.",
    "Caption this scene.",
    "Generate a caption for this image.",
    "Produce a caption for this scene.",
    "Generate a formal caption for this image.",
    "Write an image caption for this scene.",
    "Give me a caption for this satellite image.",
    "Give me a description of what this image shows.",
    "Write a caption for this satellite image.",
    "Summarise what is visible in this scene.",
    "What does this image show?",
    "Provide an overall description of the imagery.",
    "Explain what this scene contains.",
    "Give a short summary of this image.",
    "Describe the overall content of this scene.",
    "Write a few sentences about this image.",
    "Describe the landscape in this image.",
    "What would you say this image depicts?",
    "Sum up the imagery.",
    "In a sentence or two, describe this scene.",
    "Talk me through what this image shows.",
]

_GROUND = [
    "Show me where the {feature} are.",
    "Locate the {feature} in this image.",
    "Highlight all {feature}.",
    "Point to {singular}.",
    "Where is {singular} in this scene?",
    "Draw boxes around the {feature}.",
    "Mark the location of the {feature}.",
    "Find the {feature} and show their positions.",
    "Identify where {feature} appear.",
    "Give me the coordinates of the {feature}.",
    "Outline the {feature} in the image.",
    "Where exactly are the {feature} located?",
    "Detect and localise the {feature}.",
    "Circle the {feature} in this scene.",
    "Pinpoint {singular} for me.",
    "Put a box on every {singular}.",
    "Whereabouts is {singular}?",
    "I want the pixel positions of the {feature}.",
    "Box the {feature}.",
    "Which part of the image has the {feature}?",
]

_LANDCOVER = [
    "Classify the {landcover} in this image.",
    "Produce a {landcover} map.",
    "What {landcover} are present in this scene?",
    "Generate a {landcover} classification.",
    "Break this scene down by {landcover}.",
    "Map the {landcover} for this image.",
    "Which {landcover} categories occur here?",
    "Give me a {landcover} breakdown.",
    "Segment this image by {landcover}.",
    "Label the {landcover} across the scene.",
    "Provide a per-class {landcover} summary.",
    "Categorise the surface into {landcover}.",
    "Run a {landcover} classification on this image.",
    "What fraction of each {landcover} class is present?",
    "Produce a thematic {landcover} map.",
]

_XMODAL = [
    "Combine the optical and radar images to identify {feature}.",
    "Using both the optical and SAR data, find {feature}.",
    "What does the SAR image add to the optical view?",
    "Fuse the optical and radar imagery and describe {feature}.",
    "Compare what the optical and SAR sensors show for {feature}.",
    "Use the radar and optical images together to assess {feature}.",
    "Where do the optical and SAR images disagree about {feature}?",
    "Jointly analyse the optical and SAR scenes.",
    "Extract {feature} using both sensors.",
    "Cross-check the optical result against the SAR data.",
    "What can radar reveal here that optical cannot?",
    "Merge the two modalities and report on {feature}.",
    "Use optical and SAR jointly to map {feature}.",
    "Exploit both sensors to detect {feature}.",
    "Combine radar backscatter with optical reflectance for {feature}.",
]

_CHANGE_DESC = [
    "Describe what changed between the two images.",
    "What is different between these two dates?",
    "Summarise the changes between the two acquisitions.",
    "Explain how the scene changed over time.",
    "Tell me what has changed here.",
    "Describe the {change} visible between the images.",
    "Give me a narrative of the changes.",
    "What developments occurred between the two images?",
    "Characterise the differences between the two scenes.",
    "Report on how this area evolved.",
    "Describe the transformation between the two dates.",
    "In words, what changed here?",
    "Give an account of the {change}.",
    "Summarise the temporal differences.",
    "What happened between the first and second image?",
    "Whats moved since last time?",
    "Write up how the area shifted.",
    "Talk me through the differences across the two passes.",
    "Describe how things developed since the earlier image.",

    # Compound "what AND where" queries (added 2026-08-30, limitation L13).
    #
    # PS representative query 3 is "What changed between these two dates, and
    # where did the change occur?" - it asks for both a description and a
    # location. It was routing to TEMPORAL_CHANGE_MAP, because _CHANGE_MAP
    # owns every "where" phrasing ("Show me where the changes occurred"), and
    # MAP's plan is index_engine -> change_mask: it returns the mask and the
    # answer "Produced a change mask; see the exported raster artifact."
    # That answers *where* and not *what*.
    #
    # TEMPORAL_CHANGE_DESC's plan is index_engine -> change_mask ->
    # change_caption, so it already produces **both** the georeferenced mask
    # and the prose. The compound shape belongs here.
    #
    # A plain "show me where" still belongs to MAP and is left there: the
    # distinction is whether the user asked for a description as well.
    #
    # None of these is the PS string verbatim. That is deliberate - the PS
    # query is an acceptance test in tests/test_golden_traces.py, and a
    # template equal to the test string would prove memorisation rather than
    # that the router generalises to the shape.
    "Tell me what changed and where it happened.",
    "Which features changed, and in which parts of the scene?",
    "Describe the changes and point out where they are.",
    "What is different between the images, and whereabouts?",
    "Identify what changed and its location.",
    "Explain the changes and where they occurred in the scene.",
    "What has altered here, and in which areas?",
    "Report what changed and the areas affected.",

    # Transition phrasings: "which areas BECAME X". The bank had the
    # quantitative shape ("How many new {feature} appeared?" in _CHANGE_VQA)
    # and CDVQA's "What have the areas of X mainly changed to?", but not the
    # plain conversational transition, so "Which areas became built up?" on a
    # bitemporal pair scored 0.326 for change against 0.273 for SINGLE_VQA -
    # under the confidence bar, so the router fell back to the single-image
    # default and rs_vqa_v1 answered "only one image was provided" about a
    # two-image input. These belong with DESC rather than MAP because they
    # ask *which areas*, and DESC's plan is index -> change_mask ->
    # change_caption, which returns the georeferenced mask AND the prose.
    "Which areas became {feature}?",
    "Which areas turned into {feature}?",
    "Which parts of the scene became {feature}?",
    "What became {feature} between the two dates?",
    "Which regions changed to {feature}?",
    "Which areas were cleared?",
    "What has been replaced by {feature}?",
    "Which areas have become {feature} since the earlier image?",
]

_CHANGE_VQA = [
    "How much did the {feature} area change between the two images?",
    "Did the {feature} increase or decrease?",
    "By what percentage did {feature} change?",
    "How many new {feature} appeared?",
    "Was there any {change} between the two dates?",
    "Did {change} occur here?",
    "How much {feature} was lost?",
    "What is the net change in {feature}?",
    "Is the {feature} extent larger in the second image?",
    "Quantify the change in {feature}.",
    "How many {feature} were removed?",
    "What is the change in area of {feature} in hectares?",
    "Did the amount of {feature} grow?",
    "How significant was the {change}?",
    "Count the {feature} that changed.",
    "How many more {feature} are there now than before?",
    "Put a number on the {change}.",
    "Did the {feature} shrink since the earlier image?",
    "How much {feature} was there before compared to now?",
    "Is there more {feature} now than previously?",
]

_CHANGE_MAP = [
    "Produce a change mask for these two images.",
    "Generate a change detection map.",
    "Show me where the changes occurred.",
    "Map the areas that changed between the two dates.",
    "Create a binary change raster.",
    "Output a change mask highlighting {change}.",
    "Give me a georeferenced change layer.",
    "Render the changed pixels as a mask.",
    "Produce a change map I can open in QGIS.",
    "Segment the changed regions.",
    "Export the change detection result as a raster.",
    "Where did {change} happen? Give me a mask.",
    "Delineate the changed areas.",
    "Generate a per-pixel change map.",
    "Produce a spatial map of {change}.",
    "I want a raster showing altered pixels.",
    "Output a difference layer for GIS.",
    "Draw the footprint of what moved.",
    "Give me a mask of the altered pixels.",
    "Export a difference raster between the two dates.",
]

# Content-free requests: filler that names no visual subject, or that refers
# to context the system does not have ("the other one", "last time").
#
# This class is *not* where a question with a visual subject belongs, however
# vaguely it is phrased. It kept "Is this an urban or rural scene?" out of the
# VQA model for exactly that reason - see the note on _GENERAL_VISUAL - and
# the fix was to give the general shapes their own templates rather than to
# thin this list out, so the filler below still abstains.
_CLARIFY = [
    "Hello.",
    "Hi there.",
    "Hey.",
    "What do you think?",
    "Tell me about it.",
    "Can you help?",
    "Do the thing.",
    "Have a look and let me know.",
    "Just take a look.",
    "OK.",
    "Thoughts?",
    "Analyse.",
    "Go ahead.",
    "What should I do next?",
    "Is it good?",
    "Compare them.",
    "Which one is better?",
    "Show me the difference.",
    "What about the other one?",
    "Explain.",
    "And?",
    "Continue.",
    "Sort it out.",
]

# Real CDVQA question phrasings, for the routing gap measured on 2026-08-30.
#
# The synthetic templates above were written by us, and the router trained on
# them sent only 67.4% of CDVQA's questions to TEMPORAL_CHANGE_VQA - the tool
# that can actually answer them. `change_to_what` scored **0.000**: not one of
# its phrasings ("What have the areas of X mainly changed to?") reached the
# change-VQA task, they went to the change map or the change captioner. That
# routing loss, not the segmenter, was the larger half of the end-to-end
# CDVQA deficit.
#
# **Only half the templates are here, and that is deliberate.** CDVQA ships
# 300 distinct question phrasings and its train and test splits use *exactly
# the same 300* - zero novel phrasings at test time. Training on all of them
# would produce perfect routing that measures memorisation, which is the same
# trap the in-template split of this very bank falls into (see
# docs/phase1-status.md: 100%, "near-meaningless"). So the templates are split
# by a stable hash and only the training half is used; the other 151 are never
# seen, and routing on them is the number that means something.
#
# The split rule is `sha1(template)[0] % 2 == 0`, reproduced by
# `evaluation/cdvqa_routing.py` so the held-out half can be identified from
# the data on disk without shipping it here.
_CDVQA_TRAINED_TEMPLATES = [
    'Did the areas of buildings change in the pre-change image?',
    'Did the areas of buildings decrease?',
    'Did the areas of low vegetation change in the post-change image?',
    'Did the areas of low vegetation decrease?',
    'Did the areas of low vegetation increase?',
    'Did the areas of non-vegetated ground surface change in the pre-change image?',
    'Did the areas of playgrounds change in the post-change image?',
    'Did the areas of playgrounds change in the pre-change image?',
    'Did the areas of trees decrease?',
    'Did the areas of water decrease?',
    'Did the regions of buildings change in the post-event image?',
    'Did the regions of buildings change in the pre-event image?',
    'Did the regions of low vegetation change in the post-event image?',
    'Did the regions of low vegetation change?',
    'Did the regions of low vegetation decrease?',
    'Did the regions of non-vegetated ground surface change?',
    'Did the regions of non-vegetated ground surface decrease?',
    'Did the regions of non-vegetated ground surface increase?',
    'Did the regions of playgrounds change in the post-event image?',
    'Did the regions of playgrounds change in the pre-event image?',
    'Did the regions of playgrounds decrease?',
    'Did the regions of playgrounds increase?',
    'Did the regions of trees change in the post-event image?',
    'Did the regions of water change in the pre-event image?',
    'Did the regions of water decrease?',
    'Did the regions of water increase?',
    'Have the areas of buildings changed in the second image?',
    'Have the areas of buildings decreased?',
    'Have the areas of low vegetation changed in the second image?',
    'Have the areas of low vegetation changed?',
    'Have the areas of low vegetation decreased?',
    'Have the areas of low vegetation increased?',
    'Have the areas of non-vegetated ground surface changed in the first image?',
    'Have the areas of non-vegetated ground surface decreased?',
    'Have the areas of playgrounds changed in the first image?',
    'Have the areas of playgrounds changed?',
    'Have the areas of trees changed?',
    'Have the areas of water changed in the second image?',
    'Have the areas of water decreased?',
    'Have the areas of water increased?',
    'Have the regions of buildings changed in the first image?',
    'Have the regions of buildings changed in the second image?',
    'Have the regions of low vegetation changed in the first image?',
    'Have the regions of low vegetation changed in the second image?',
    'Have the regions of low vegetation decreased?',
    'Have the regions of non-vegetated ground surface changed in the first image?',
    'Have the regions of non-vegetated ground surface changed in the second image?',
    'Have the regions of non-vegetated ground surface decreased?',
    'Have the regions of non-vegetated ground surface increased?',
    'Have the regions of playgrounds changed in the first image?',
    'Have the regions of playgrounds changed in the second image?',
    'Have the regions of playgrounds changed?',
    'Have the regions of playgrounds decreased?',
    'Have the regions of playgrounds increased?',
    'Have the regions of trees changed in the first image?',
    'Have the regions of trees changed in the second image?',
    'Have the regions of trees increased?',
    'Have the regions of water changed in the first image?',
    'Have the regions of water changed?',
    'Have the regions of water decreased?',
    'Have the regions of water increased?',
    'How much area of low vegetation has changed in the pre-change image?',
    'How much area of non-vegetated ground surface has changed in the first image?',
    'How much area of non-vegetated ground surface has changed in the pre-change image?',
    'How much area of playgrounds has changed in the first image?',
    'How much area of playgrounds has changed in the pre-change image?',
    'How much area of playgrounds has changed in the second image?',
    'How much area of trees has changed in the first image?',
    'How much area of trees has changed in the post-change image?',
    'How much area of trees has changed in the pre-change image?',
    'How much area of water has changed in the first image?',
    'How much area of water has changed in the second image?',
    'How much of the area has changed?',
    'How much of the area has not changed?',
    'What have the areas of buildings in the pre-event image mainly changed to?',
    'What have the areas of low vegetation in the first image mainly changed to?',
    'What have the areas of non-vegetated ground surface in the first image mainly changed to?',
    'What have the areas of non-vegetated ground surface in the pre-event image mainly changed to?',
    'What have the areas of trees in the pre-event image mainly changed to?',
    'What have the areas of water in the first image mainly changed to?',
    'What have the areas of water in the pre-event image mainly changed to?',
    'What have the regions of buildings in the first image mainly changed to?',
    'What have the regions of buildings in the pre-change image mainly changed to?',
    'What have the regions of low vegetation in the first image mainly changed to?',
    'What have the regions of low vegetation in the pre-change image mainly changed to?',
    'What have the regions of non-vegetated ground surface in the first image mainly changed to?',
    'What have the regions of non-vegetated ground surface in the pre-change image mainly changed to?',
    'What have the regions of playgrounds in the pre-event image mainly changed to?',
    'What have the regions of water in the first image mainly changed to?',
    'What is the change percentage of buildings in the post-change image?',
    'What is the change percentage of buildings in the second image?',
    'What is the change percentage of low vegetation in the post-change image?',
    'What is the change percentage of low vegetation in the pre-change image?',
    'What is the change percentage of low vegetation in the second image?',
    'What is the change percentage of non-vegetated ground surface in the first image?',
    'What is the change percentage of non-vegetated ground surface in the post-change image?',
    'What is the change percentage of non-vegetated ground surface in the pre-change image?',
    'What is the change percentage of playgrounds in the second image?',
    'What is the change percentage of trees in the first image?',
    'What is the change percentage of trees in the post-change image?',
    'What is the change percentage of trees in the pre-change image?',
    'What is the change percentage of water in the post-change image?',
    'What is the change percentage of water in the pre-change image?',
    'What is the change proportion of buildings in the first image?',
    'What is the change proportion of buildings in the second image?',
    'What is the change proportion of low vegetation in the post-event image?',
    'What is the change proportion of non-vegetated ground surface in the first image?',
    'What is the change proportion of non-vegetated ground surface in the pre-event image?',
    'What is the change proportion of non-vegetated ground surface in the second image?',
    'What is the change proportion of playgrounds in the post-event image?',
    'What is the change proportion of playgrounds in the pre-event image?',
    'What is the change proportion of playgrounds in the second image?',
    'What is the change proportion of trees in the post-event image?',
    'What is the change proportion of trees in the pre-event image?',
    'What is the change proportion of trees in the second image?',
    'What is the change ratio of buildings in the post-event image?',
    'What is the change ratio of buildings in the pre-change image?',
    'What is the change ratio of buildings in the pre-event image?',
    'What is the change ratio of buildings in the second image?',
    'What is the change ratio of low vegetation in the first image?',
    'What is the change ratio of low vegetation in the pre-change image?',
    'What is the change ratio of low vegetation in the pre-event image?',
    'What is the change ratio of low vegetation in the second image?',
    'What is the change ratio of non-vegetated ground surface in the second image?',
    'What is the change ratio of playgrounds in the post-event image?',
    'What is the change ratio of playgrounds in the second image?',
    'What is the change ratio of trees in the post-event image?',
    'What is the change ratio of trees in the pre-change image?',
    'What is the change ratio of trees in the pre-event image?',
    'What is the change ratio of trees in the second image?',
    'What is the change ratio of water in the post-change image?',
    'What is the change ratio of water in the second image?',
    'What is the largest change in the first image?',
    'What is the largest change in the post-change image?',
    'What is the percentage of changed areas?',
    'What is the percentage of changed regions?',
    'What is the percentage of non-change regions?',
    'What is the smallest change in the first image?',
    'What is the smallest change in the post-change image?',
    'What is the smallest change?',
    'What type of change is the largest in the first image?',
    'What type of change is the largest in the post-event image?',
    'What type of change is the largest in the pre-event image?',
    'What type of change is the largest?',
    'What type of change is the smallest in the post-change image?',
    'What type of change is the smallest in the post-event image?',
    'What type of change is the smallest in the pre-change image?',
    'What type of change is the smallest in the pre-event image?',
    'What type of change is the smallest?',
]


TEMPLATES: dict[TaskID, list[str]] = {
    "SINGLE_VQA": _VQA + _GENERAL_VISUAL,
    "SINGLE_CAPTION": _CAPTION,
    "SINGLE_GROUND": _GROUND,
    "SINGLE_LANDCOVER": _LANDCOVER,
    "XMODAL_JOINT_EXTRACT": _XMODAL,
    "TEMPORAL_CHANGE_DESC": _CHANGE_DESC,
    "TEMPORAL_CHANGE_VQA": _CHANGE_VQA + _CDVQA_TRAINED_TEMPLATES,
    "TEMPORAL_CHANGE_MAP": _CHANGE_MAP,
    "CLARIFY_OR_ABSTAIN": _CLARIFY,
}

# Prefixes and suffixes that add natural variation without changing intent.
_PREFIXES = ["", "", "", "Please ", "Could you ", "I need you to ", "Can you "]
_SUFFIXES = ["", "", "", "", " for me", " please", " in this scene"]


@dataclass(frozen=True)
class QueryExample:
    text: str
    task: TaskID


def _fill(template: str, rng: random.Random) -> str:
    feature = rng.choice(FEATURES)
    feature2 = rng.choice([f for f in FEATURES if f != feature])
    scene = rng.choice(SCENE_TYPES)
    scene2 = rng.choice([s for s in SCENE_TYPES if s != scene])
    return (
        template.replace("{feature2}", feature2)
        .replace("{feature}", feature)
        .replace("{singular}", rng.choice(SINGULAR))
        .replace("{landcover}", rng.choice(LANDCOVER_CLASSES))
        .replace("{change}", rng.choice(CHANGE_NOUNS))
        .replace("{scene2}", scene2)
        .replace("{scene}", scene)
    )


def _decorate(text: str, rng: random.Random) -> str:
    prefix = rng.choice(_PREFIXES)
    suffix = rng.choice(_SUFFIXES)
    if prefix and text:
        # Lower-case the first letter so "Please Describe" reads naturally.
        text = text[0].lower() + text[1:]
    if suffix and text.endswith((".", "?")):
        text = text[:-1] + suffix + text[-1]
    return prefix + text


def generate(
    per_task: int = 400, seed: int = 20260829, decorate: bool = True
) -> list[QueryExample]:
    """Generate a balanced, deterministic query bank."""
    rng = random.Random(seed)
    examples: list[QueryExample] = []
    for task, templates in TEMPLATES.items():
        seen: set[str] = set()
        attempts = 0
        while len(seen) < per_task and attempts < per_task * 60:
            attempts += 1
            text = _fill(rng.choice(templates), rng)
            if decorate:
                text = _decorate(text, rng)
            if text not in seen:
                seen.add(text)
        examples.extend(QueryExample(text=t, task=task) for t in sorted(seen))
    rng.shuffle(examples)
    return examples


def train_test_split(
    examples: list[QueryExample], test_fraction: float = 0.2, seed: int = 7
) -> tuple[list[QueryExample], list[QueryExample]]:
    """Stratified split so every task is represented in both halves."""
    rng = random.Random(seed)
    by_task: dict[str, list[QueryExample]] = {}
    for ex in examples:
        by_task.setdefault(ex.task, []).append(ex)

    train: list[QueryExample] = []
    test: list[QueryExample] = []
    for task_examples in by_task.values():
        shuffled = list(task_examples)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * test_fraction)
        test.extend(shuffled[:cut])
        train.extend(shuffled[cut:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test
