# Rehearsal record

**Plan task 4.2: "Rehearse the 7-minute demo ten times, including on the
actual venue laptop with networking off."**

That item has two halves and only one of them is automatable. This file
records what was measured, and states plainly what was not.

## Venue checklist - do this before the judges arrive

**The whole of it: bring the stack up, put ONE cross-modal query through the
API, wait for it to come back, then start.** Everything below is why, and the
variants.

### If you have one minute

```bash
docker compose up -d
# wait until `docker compose ps` shows satquery_ai-api-1 as (healthy)

curl -s -X POST http://localhost:8000/runs   -F "query=Use the optical and SAR images together to identify built-up and water-covered regions."   -F "images=@data/demo_bundle/synthetic/optical_t1.tif"   -F "images=@data/demo_bundle/synthetic/sar_dualpol.tif" > /dev/null
```

**That curl takes about 70-80 seconds** (79 s measured 2026-09-07). **Wait
for it to return.** It is slow
*because* it is the warm-up - it is paying the model-load cost so the demo
does not. The next cross-modal query costs **3.6-3.8 s**, measured three times
in a row.

> **Do not start the demo 2-3 seconds after firing it.** The wait is for the
> command to *finish*, not a pause before it works. Starting early means the
> first live beat is still the slow one, or queues behind the warm-up.

### If you have three minutes

Run the one-minute sequence, then warm the real-product path as well:

```bash
curl -s -X POST http://localhost:8000/runs   -F "query=Describe the land-cover and major objects visible in this image."   -F "images=@data/bhoonidhi/cartosat2s_mx_5132611/5132611/BAND1.tif"   -F "images=@data/bhoonidhi/cartosat2s_mx_5132611/5132611/BAND2.tif"   -F "images=@data/bhoonidhi/cartosat2s_mx_5132611/5132611/BAND3.tif"   -F "images=@data/bhoonidhi/cartosat2s_mx_5132611/5132611/BAND4.tif" > /dev/null
```

**About 40 s** (39 s measured, run second - it is quicker than the first
because the stack is already up). It buys less than the cross-modal warm-up -
the Cartosat beat is mostly raster I/O, so it goes from ~73 s to ~58 s rather
than collapsing - but that beat is the one that overruns its slot in **every**
run, so the 15 seconds are worth having.

Both commands were run end to end on 2026-09-07 and returned
`XMODAL_JOINT_EXTRACT` and `SINGLE_CAPTION` respectively. The four BAND files
arrive as **one** logical image - the API groups them (limitation L17), and the
trace confirms `n_images: 1`.

### Two ways to lose the warm-up

1. **Restarting anything.** The warm state lives in the running `uvicorn`
   process. `docker compose restart`, a crash, or a laptop sleep/resume that
   kills the container puts you back to cold. Warm again if that happens.
2. **Warming from the CLI.** `docker compose exec ... satquery ask` warms a
   process that then exits. **Measured: it does not help** - a full pass after
   a CLI warm-up still took 219.9 s, versus 131 s genuinely warm. It has to go
   through the API.

### If you are on the CPU fallback

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
```

**Do not bother warming it** - the learned tools are stubs, nothing loads, and
a full pass is already 104 s.

**One change to the script there:** `clouded_optical` **answers instead of
abstaining** on this config, so the 4:50 abstention beat does not demonstrate
abstention. Close the live portion on `incompatible_pair` instead - it abstains
in both configurations. See `docs/00` **L36** and the note on deck Slide 5.

### Sanity check, if there is time

```bash
docker compose exec -T api python scripts/make_demo_bundle.py --verify
```

Expect **9/9 beats behave as scripted**. Takes about 4-5 minutes, so this is a
morning-of check rather than a five-minutes-before one.

---

## What was measured — on the HOST, 2026-08-30

> **Still broadly valid.** An earlier revision of this file claimed the
> container was much slower and told you to plan against container numbers
> instead. With n=4 that is **half right**: a *cold* container pass is slow
> (224.6 s median), but a *warm* one is **131.4 s** - close to this table's
> 118.5 s median and well inside its 111.0-268.2 s range. The difference is
> cold start, not the container. See "In-container timings" below for the
> per-beat picture and the one beat that genuinely overruns.

`scripts/rehearse.py` executes every beat of the `docs/04` §10 script through
the real controller, in the scripted order, and checks that each beat produces
what the script says it produces. Twenty rehearsals: ten online, ten with the
socket layer blocked.

| | online | offline |
|---|---|---|
| Rehearsals | 10 | 10 |
| **All beats behaved, every run** | **yes** | **yes** |
| Median total system time | 118.5 s | **116.6 s** |
| Fastest / slowest run | 111.0 s / 268.2 s | 106.0 s / 135.9 s |
| First (cold) run | 126.5 s | 135.9 s |
| Runs 2–10, median | 116.6 s | 116.3 s |

Artifacts: `docs/assets/rehearsal/online.json`, `.../offline.json`.

**Offline is not slower.** 116.6 s against 118.5 s median, and the offline
spread is *tighter* (max 135.9 s against 268.2 s). The online outlier is the
network being attempted; with sockets blocked there is nothing to wait for.
The system does not need the internet, and that is now measured rather than
claimed.

## Per-beat timings on the host, and the problem they found

| beat | median | slot | |
|---|---|---|---|
| 0:30 rejection — incompatible pair | 0.03 s | 40 s | ✅ |
| 0:30 rejection — PNG in operational mode | 0.01 s | 40 s | ✅ |
| 1:10 cross-modal flagship | 0.29 s | 70 s | ✅ |
| **2:20 single optical, real Cartosat** | **56.59 s** | 50 s | ❌ **over** |
| 2:20 single SAR, real EOS-04 | 2.32 s | 50 s | ✅ |
| 3:10 bi-temporal — what changed and where | 0.23 s | 60 s | ✅ |
| 3:10 bi-temporal — increased or decreased | 0.24 s | 60 s | ✅ |
| 4:50 abstention — clouded optical | 0.28 s | 50 s | ✅ |
| **5:40 the large scene, real Cartosat** | **55.36 s** | 60 s | ⚠ marginal |

**The finding: the two real-Cartosat beats cost about 56 seconds each.**
*(Holds up in-container at n=4: single optical **63.0 s** median, large scene
**57.4 s** median. An n=1 revision of this file briefly claimed a third slow
beat; four runs showed that was cold start. See below.)* That
is the full 7687×7640, four-band product going through ingest, tiling and the
index engine — it is honest work, not a bug, and it is roughly the entire slot
those beats have in a seven-minute script. Together they are **112 of the
118 seconds** of system time in a rehearsal; every other beat finishes in
under three seconds.

### What to do about it, in preference order

1. **Pre-warm both Cartosat runs before the demo starts** and show the stored
   `/runs/{id}` permalinks. The GUI renders a stored run identically to a live
   one — verified in the browser — so nothing about the demo looks different.
2. **Narrate over it.** 56 seconds is enough to explain Axiom 2 (no SWIR on
   Cartosat, so NDBI is unavailable and the SAR-primary path fires) while the
   trace fills. This is the option that shows real work happening.
3. **Do not** substitute a synthetic scene for the Cartosat beat. Real
   target-sensor imagery is the most convincing thing in the demo, and the
   whole point of holding those products out of training.

The other seven beats total under 2 seconds, so the script has slack
everywhere except here.

## In-container timings - 2026-09-07, **n=4**

Same `scripts/rehearse.py`, unmodified, four passes inside the running GPU
container: one standalone pass, then a `--runs 3` invocation. The CPU column
is a single pass, kept for the fallback comparison.

**Read the cold/warm split before the numbers.** `--runs 3` executes all
three passes in one process, so its first pass is cold and the other two are
warm. Combined with the standalone pass, the four runs are **2 cold + 2
warm** - and the two groups are not close to each other.

| | total system time |
|---|---|
| Cold passes (2) | 239.8 s, 209.5 s - **median 224.6 s** |
| **Warm passes (2)** | 132.7 s, 130.2 s - **median 131.4 s** |
| **All four** | **median 171.1 s**, range **130.2-239.8 s** |
| Host, for comparison (10 runs) | median 118.5 s, range 111.0-268.2 s |

**A warm container is close to the host** - 131.4 s against 118.5 s - and
every one of the four runs sits inside the host's own observed range. The
slowness is **cold start**, not the container.

### Per-beat, four runs

| beat | slot | cold | cold | warm | warm | **median** | over slot |
|---|---|---|---|---|---|---|---|
| 0:30 rejection - incompatible pair | 40 s | 0.29 | 0.20 | 0.17 | 0.11 | **0.19** | 0/4 |
| 0:30 rejection - PNG in operational mode | 40 s | 2.26 | 1.94 | 0.19 | 0.17 | **1.06** | 0/4 |
| 1:10 cross-modal flagship | 70 s | 83.34 | 69.48 | 4.25 | 4.14 | **36.87** | **1/4** |
| **2:20 single optical, real Cartosat** | 50 s | 73.16 | 67.68 | 58.33 | 55.13 | **63.01** | **4/4** ❌ |
| 2:20 single SAR, real EOS-04 | 50 s | 11.59 | 11.21 | 12.61 | 11.18 | **11.40** | 0/4 |
| 3:10 bi-temporal - what changed and where | 60 s | 0.88 | 0.67 | 0.33 | 0.39 | **0.53** | 0/4 |
| 3:10 bi-temporal - increased or decreased | 60 s | 0.91 | 0.88 | 0.42 | 0.33 | **0.65** | 0/4 |
| 4:50 abstention - clouded optical | 50 s | 0.89 | 0.67 | 0.91 | 0.66 | **0.78** | 0/4 |
| 5:40 the large scene | 60 s | 66.47 | 56.75 | 55.52 | 58.02 | **57.39** | **1/4** |

All nine beats produced their scripted result in all four runs.

### Exactly one beat overruns consistently

* **2:20 single optical, real Cartosat - over in 4 of 4**, median 63.0 s
  against a 50 s slot. This is the real one. Plan for it every time.
* **1:10 cross-modal flagship - over in 1 of 4.** Cold it costs 69-83 s;
  warm it costs **4.2 s**, a 17x swing. Nearly all of that beat's apparent
  cost was first-call model initialisation, not the fusion work.
* **5:40 the large scene - over in 1 of 4**, and its median of 57.4 s is
  *inside* its 60 s slot. Marginal, as the host record already said.

### Warming works - but only through the API, and this was measured

**The single highest-value thing you can do is warm the API before the demo.**
Both halves of that sentence were verified rather than inferred, because the
first version of this note was inferred and was wrong.

| what was tested | result |
|---|---|
| Throwaway CLI query, then `rehearse.py` in a **new** process | **219.9 s - a cold run.** Cross-modal came back at 81.3 s, no benefit at all |
| **Cross-modal beat POSTed to `/runs` three times, same container** | **70.0 s, then 3.8 s, then 3.6 s** |

**Why the difference:** the tool model handles are per-process singletons, and
`docker compose exec` starts a fresh process every time - so a CLI warm-up
warms a process that then exits. The API is one long-lived `uvicorn` process
serving every request, so warming it warms the thing that actually serves the
demo. **The demo drives the API, so warming applies.**

**Warm with the cross-modal input specifically.** A warm-up only warms the
tools it touches: a caption query does not load `optsar_fusion_v1`, and the
cross-modal beat is the one with the 18x swing. One POST of the cross-modal
pair before the judges walk in is the whole mitigation.

Warmed, the total drops to roughly **131 s** (the two warm `rehearse.py`
passes, same mechanism, in-process) from ~225 s cold, and takes two of the
three "slow" beats off the board.

### Filler cues

**Needed every run:**

* **2:20 single optical, real Cartosat (63 s median, 50 s slot).**
  *While this runs, say:* "This is a real Cartosat-2S product, 7687 by 7640
  pixels, four bands, going through ingest, tiling and the index engine. Most
  of this wait is reading the scene off disk, not the model - we measured it
  at 48 seconds with every learned tool switched off. It is the price of not
  substituting a synthetic image for the sensor you actually care about."

**Cold-start contingency only** - if the stack was not warmed, or the venue
machine restarts it between beats:

* **1:10 cross-modal flagship (4 s warm, up to 83 s cold).**
  *While this runs, say:* "It is computing the triad - optical alone, SAR
  alone, and fused - as three separate passes, because that is the only way
  to measure whether fusion actually helps. Ours says it does not: optical
  0.7778, fused 0.7714, a gain of minus 0.0064. We report all three rather
  than one fused number that hides it."
* **5:40 the large scene (57 s median, 60 s slot - usually fits).**
  *While this runs, say:* "Same product path, and while it works: every
  neural claim in the answer is checked against NDVI, NDWI, NDBI and SAR
  backscatter computed deterministically from the pixels. The verifier has no
  opinion and no training - it is arithmetic - which is what makes a
  confident wrong answer catchable rather than merely unlikely."

**Still the better option where it is available:** pre-warm the two
real-product runs before the demo and show the stored `/runs/{id}`
permalinks, as the host section recommends. The cues are for when a judge
asks to see it run live.

### The slot math

| | |
|---|---|
| Demo slot | **420 s** (7 minutes) |
| System time, warm container | **131.4 s** - 31% of the slot |
| System time, n=4 median | **171.1 s** - 41% |
| System time, cold container | 224.6 s - 53% |
| **Left for narration, warm** | **289 s (4:49)** across nine beats |

**Warmed, the seven-minute script fits comfortably** - 131.4 s of 420 s, with
4:49 for narration, which is close to the 5:02 the host figures implied. The
one structural problem that survives four runs is the single-optical beat,
**13 s over its slot at the median**; the narration cue above is sized to
cover it. Cold, the script still fits end to end at 224.6 s but the two
cold-only beats push everything after 1:10 late.

### CPU fallback, single pass

104.2 s total, zero beats over slot, all nine correct - the stubs skip the
model work, so the fallback is *faster* than the real system. Worth knowing
before a judge notices. `clouded_optical` **answers rather than abstains**
there; see `docs/00` **L36** and the do-not-present note on deck Slide 5.

## What was NOT measured, and is not done

**This is not ten rehearsals in the sense the plan means.** A rehearsal is a
person driving the demo and speaking to a clock. What ran here is the system's
half: the beats execute, in order, repeatably, within budget except where
noted. Reporting this as "task 4.2 complete" would be a false claim.

Specifically not covered:

* **Narration and timing against the spoken script.** The 7-minute budget is
  dominated by speech, not compute, and nothing here measures whether the
  words fit.
* **The actual venue laptop.** These ran on the development machine. The plan
  names the venue laptop because that is where surprises happen — different
  GPU or none, different screen, a locked-down network.
* **Recovery from interruption.** A judge asking a question mid-beat is the
  most likely live failure and cannot be simulated.
* **The GUI path end to end.** Beats were driven through the controller; a
  separate browser pass verified the run view, three-component confidence, map
  overlays and PDF link, but not the upload-and-watch-the-trace-stream flow
  ten times over.

**What the team must still do:** ten timed run-throughs with narration, at
least one on the venue laptop with its network off, and one recorded (task
4.6's backup video, also open).

## How to re-run

```bash
python scripts/make_demo_bundle.py --out data/demo_bundle --verify
python scripts/rehearse.py --runs 10 --out docs/assets/rehearsal/online.json
python scripts/rehearse.py --runs 10 --offline --out docs/assets/rehearsal/offline.json
```

The script exits non-zero if any beat misbehaves or exceeds its slot, so it is
usable as a pre-demo check on the venue machine — which is the cheapest way to
find out that the venue laptop is slower than this one.
