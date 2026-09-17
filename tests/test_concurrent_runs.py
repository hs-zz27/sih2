"""Two runs at once must not contaminate each other's trace.

The API builds one `Controller` for the process and `/runs/stream` serves
each request from its own thread, so every piece of per-run state that lived
on the router was shared by every request in flight. The router used to
publish two of them - `last_prediction` and `last_config_excluded` - and the
controller read them back after calling `route()`. Between the write and the
read another thread could route a different query.

The visible failure is not a crash. It is one user's answer carrying another
user's "this input configuration cannot support TEMPORAL_CHANGE_MAP" notice,
with nothing in the trace to show where it came from.

**It was measured before it was fixed**, by replaying the pre-fix router and
the pre-fix read pattern under contention:

| replay | contamination |
|---|---|
| router level, 4+4 threads x 200 rounds | 97 / 800 |
| controller level, 4+4 threads x 40 rounds | 2 / 160 |
| controller level, 2+2 threads x 10 rounds, 3 ms classifier | 6 / 20 |

The first version of this test used two threads and no delay and measured
**zero** - it would have passed against the defect it was written for. The
delay in `_SlowClassifier` is what makes the window observable in a second
rather than in a minute of brute force.
"""

from __future__ import annotations

import threading
import time

from satquery.controller.matrix_loader import load_matrix
from satquery.controller.pipeline import Controller
from satquery.controller.router import Router
from satquery.ingest import ingest

# A query whose best task is excluded by a SINGLE image, and one that is not.
CHANGE_QUERY = "Produce a change mask for these images."
DESCRIBE_QUERY = "Describe this image."


class _SlowClassifier:
    """The real classifier with a few milliseconds added to each prediction.

    Widens the interleaving window without changing a single decision: every
    prediction is the wrapped classifier's own. A test for a race has to make
    the race likely, and the alternative - thousands of rounds - costs half a
    minute of CI for the same information.
    """

    def __init__(self, inner, delay: float = 0.003):
        self.inner = inner
        self.delay = delay

    def predict(self, *args, **kwargs):
        time.sleep(self.delay)
        return self.inner.predict(*args, **kwargs)


class TestRouteDecisionIsCarried:
    def test_decide_returns_the_exclusion_instead_of_only_storing_it(self, msi_6band):
        router = Router(load_matrix("configs/capability_matrix.yaml"))

        decision = router.decide(CHANGE_QUERY, ingest([msi_6band]))

        assert decision.config_excluded == "TEMPORAL_CHANGE_MAP"
        assert decision.plan.tasks[0] != "TEMPORAL_CHANGE_MAP"

    def test_a_blocked_input_reports_no_prediction(self, tiny_raster):
        """The classifier is not consulted, so there is no score to report."""
        router = Router(load_matrix("configs/capability_matrix.yaml"))

        decision = router.decide(DESCRIBE_QUERY, ingest([tiny_raster]))

        assert decision.prediction is None
        assert decision.plan.tasks[0] == "CLARIFY_OR_ABSTAIN"

    def test_a_later_route_cannot_leave_a_stale_exclusion_behind(self, msi_6band):
        """Both attributes are rewritten on every call, including abstentions.

        They are kept for the callers that read them - the routing tests do -
        and they are still only safe single-threaded, which is exactly why
        the controller uses the returned decision instead.
        """
        router = Router(load_matrix("configs/capability_matrix.yaml"))
        manifest = ingest([msi_6band])

        router.decide(CHANGE_QUERY, manifest)
        assert router.last_config_excluded == "TEMPORAL_CHANGE_MAP"
        router.decide(DESCRIBE_QUERY, manifest)
        assert router.last_config_excluded is None


class TestConcurrentControllerRuns:
    """One controller, four threads, two different queries."""

    ROUNDS = 10
    THREADS = 2

    def test_a_describe_run_never_inherits_another_runs_exclusion(
        self, msi_6band, msi_4band
    ):
        controller = Controller()
        controller.router.classifier = _SlowClassifier(controller.router.classifier)
        manifest_a = ingest([msi_6band])
        manifest_b = ingest([msi_4band])
        results: dict[str, list] = {"describe": [], "change": []}
        lock = threading.Lock()
        errors: list[BaseException] = []

        def drive(key: str, manifest, query: str) -> None:
            try:
                for _ in range(self.ROUNDS):
                    trace = controller.run_on_manifest(manifest, query)
                    with lock:
                        results[key].append(trace)
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors.append(exc)

        threads = [
            threading.Thread(target=drive, args=("describe", manifest_a, DESCRIBE_QUERY))
            for _ in range(self.THREADS)
        ] + [
            threading.Thread(target=drive, args=("change", manifest_b, CHANGE_QUERY))
            for _ in range(self.THREADS)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if errors:
            raise errors[0]

        # The describe runs asked for nothing their input cannot support, so
        # not one of them may carry an exclusion notice.
        for trace in results["describe"]:
            assert trace.routing.config_excluded_task is None
            assert "cannot support" not in trace.answer

        # And the change runs must all keep theirs: a fix that simply stopped
        # reporting exclusions would pass the assertion above.
        for trace in results["change"]:
            assert trace.routing.config_excluded_task == "TEMPORAL_CHANGE_MAP"
            assert "cannot support" in trace.answer
