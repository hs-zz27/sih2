"""JSON-safe coercion, shared by the trace serialiser and the report layer.

Non-finite floats are legitimate results here - an index over an all-nodata
band produces NaN, and average precision for a class with no positive examples
is undefined - but they are not valid JSON. Left alone they break the SSE
stream and any endpoint that serialises them.

They become `None`, which is what they mean: a missing value, distinguishable
from a measured zero. This lives here rather than inside the executor because
`satquery/report/registry.py` reads training `metrics.json` files that contain
exactly the same NaNs, and two copies of this rule would eventually disagree.
"""

from __future__ import annotations

import math
from typing import Any


def json_safe(value: Any) -> Any:
    """Recursively replace non-finite floats with None."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value
