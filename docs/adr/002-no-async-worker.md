# ADR 002: No asynchronous worker

**Date:** 2026-08-30
**Status:** Accepted
**Decision:** Remove the `worker` service. Do not replace it.

## Context

`docker-compose.yml` declared three services: `api`, `worker` and `web`. The
worker built the entire Python stack from `docker/worker.Dockerfile` and ran
`satquery.controller.worker`, whose complete body was:

```python
def main() -> None:
    print("SatQuery AI worker started (stub — no task queue wired up yet).")
```

There was no broker, no queue, and nothing that enqueued to it. `depends_on:
api` placed it in the dependency graph as though it were on the request path.
Started with `docker compose up`, the container printed one line and exited,
and its exit was indistinguishable from a crash.

Phase 0 created it as a placeholder. It was never wired up, and it was still
in the compose file at the Phase-4 freeze.

## Decision

Delete `satquery/controller/worker.py`, `docker/worker.Dockerfile`, and the
`worker` service. Do not introduce Celery, RQ, Redis, or a task table.

## Why not implement one instead

1. **Nothing asks for it.** The problem statement (`docs/ps-26167.md`)
   specifies an interactive GUI or web application that accepts an image and
   a query and returns an evidence-grounded result. It says nothing about
   background jobs, and the evaluation is over the observable execution trace.

2. **The work is already asynchronous where it matters.** `/runs/stream`
   runs the pipeline on a thread and streams ingest, routing and each tool
   step as server-sent events, which is the path the frontend uses. The user
   watches the run rather than waiting on an opaque job id.

3. **The latency profile does not need one.** Seven of the nine demo beats
   finish in under 3 seconds; the two full-Cartosat beats take ≈56 s
   (`docs/assets/rehearsal/*.json`) and stream progress throughout. A queue
   would add a broker, a result store, a polling endpoint and a second
   failure mode to a request the client is already watching.

4. **This project's own design document said so.**
   `docs/01-Solution-Architecture-and-System-Design.md`: *FastAPI plus an
   in-process queue is sufficient at demo scale. Add Redis + RQ only if you
   genuinely need multiple workers.* Removing the stub implements that
   decision; keeping it advertised a component that did not exist.

5. **A stub in the deployment topology is worse than an absence.** Anyone
   reading the compose file - a judge included - would reasonably conclude
   that submitted queries are processed by a worker. They are not.

## What would reverse this

Any one of:

* concurrent users beyond what one process can serve, where the binding
  constraint is GPU residency rather than request handling;
* a batch mode that accepts many scenes and returns later - `satquery eval`
  is that today, and it is a CLI rather than a service;
* runs that must survive an API restart, which needs durable job state.

None of these is true at the freeze, and the first is measurable rather than
speculative: the soak test (`docs/assets/soak/soak.json`) is the place to
find it.

## Consequences

* `docker compose up` now starts two services, both of which stay up, and
  both of which have healthchecks.
* `satquery.controller.worker` no longer exists. Nothing imported it.
* `tests/test_packaging.py` asserts the compose file declares no service
  whose entrypoint is a stub, so this cannot quietly return.
