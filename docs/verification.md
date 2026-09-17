# Week-0 Verification Gate

Tracks resolution of the 12-item verification gate defined in
`docs/03-Models-and-Datasets.md` §6. Plan item 0.1 requires every row below to
carry a written answer before GPU time is spent against an unverified
assumption.

| # | Claim | Status | Answer / Evidence | Owner | Resolved |
|---|---|---|---|---|---|
| 1 | BigEarthNet.txt dataset contents + licence | **Resolved** | Paper `2603.29630v2` read: 464,044 pairs, 9.6M annotations, captions + VQA + referring. HF card confirms licence **CDLA-Permissive-1.0** (permissive). Format: 467 MB Parquet, 9,553,962 rows (1 per annotation), text only — S1/S2 imagery is a separate reBEN download. | ML lead | 2026-08-27 |
| 2 | BigEarthNet.txt headline figures | **Resolved** | `2603.29630v2` verified — 464,044 pairs / 9.6M annotations / 1,082-pair benchmark all match. | ML lead | 2026-08-27 |
| 3 | CROMA / DOFA checkpoints downloadable, permissive licence | **Open** | Not yet checked. Fallback if false: torchgeo SSL weights (guaranteed). | ML #2 | Pending — Phase 1 |
| 4 | Change-Agent / LEVIR-MCI weights available | **Open** | Not yet checked. Fallback: TinyCD + separate caption head. | ML #2 | Pending — Phase 1 |
| 5 | Which RISAT sensor/mode ISRO/SAC will use (C-band vs X-band, look count) | **RESOLVED for EOS-04** | Read from a real Bhoonidhi product's `product.xml` (2026-08-29): **`radarCenterFrequency = 5.40e09 Hz` = 5.40 GHz = C-band.** Sentinel-1 is 5.405 GHz, a **0.09% difference**, so S1-trained backscatter behaviour transfers essentially directly. Look count answered too: MRS product reports `RangeLooks=2.0, AzimuthLooks=1.0`, `IncidenceAngle=37.8`, `NoOfPolarizations=2` (HH+HV), `OutputPixelSpacing=18.0 m`, 8 ScanSAR beams. FRS-1 sample is **quad-pol** (HH/HV/VH/VV). Residual risk: if ISRO instead uses RISAT-2B/2BR1 those are X-band; adaptive thresholds already cover that. **Residual risk narrowed by elimination 2026-08-29 — see §"Which RISAT" below; still needs one sentence from the team.** | Geo lead | 2026-08-29 |
| 6 | Cartosat-2S MX band composition - confirm 4-band VNIR, no SWIR | **RESOLVED** | Read from the real sample's `BAND_META.txt` (2026-08-29): **`NoOfBands=4`, `BandNumbers=1234`** - 4-band VNIR, **no SWIR**. The original assumption holds, so MNDWI/NDBI stay unavailable and the SWIR-free fallback paths are the operative ones. Also learned: `SatID=CARTOSAT-2E` (sample is 2E, same MX sensor family), `PixelSpacing=1.6 m` (matches the assumed GSD), **`BitsPerPixel=11` in a uint16 container**, `ORTHORECTIFIED`, UTM zone 45N / EPSG:32645, scene 7687x7640 px. | Geo lead | 2026-08-29 |
| 7 | Newer ≤4B VLM beats Qwen2.5-VL-3B on the §2.1 criteria | **Open** | InternVL3-1B (used by the BigEarthNet.txt authors) flagged as the candidate to evaluate first. Not yet benchmarked. | ML lead | Pending — Phase 1 |
| 8 | SpaceNet 6 / Umbra / Capella high-res SAR accessible and licensed | **RESOLVED — available, but the WRONG BAND for EOS-04** | **Accessible: yes, and confirmed by retrieval, not by reading a web page.** The Umbra bucket `s3://umbra-open-data-catalog` (us-west-2) serves anonymously — no AWS account, no credentials — and a real product's STAC metadata was pulled and read (2026-08-29, saved as `artifacts/verify/umbra.json`). **Licences are permissive:** Umbra and Capella open data are **CC BY 4.0**; SpaceNet 6 is **CC BY-SA 4.0** — note the **share-alike**, which matters because the PS lists model weights as a deliverable. **But all three are X-band, and item 5 established EOS-04 is C-band.** Measured from the Umbra product: `sar:frequency_band=X`, `sar:center_frequency=9.6924 GHz` (λ 3.09 cm) against EOS-04's 5.40 GHz (λ 5.55 cm) — **+79.5%, a 1.79× wavelength ratio**, where the Sentinel-1 match that made item 5 comfortable was 0.09%. Two further mismatches in the same file: `sar:polarizations=['HH']` (single-pol, so the VH/VV ratio and CoV the index engine computes are not derivable) and `view:incidence_angle=22.9°` against EOS-04's 37.8°. Resolution is genuinely superb (`sar:resolution_azimuth=0.125 m`, `range=0.50 m`, SPOTLIGHT). **Verdict: these sources trade a resolution gap for a frequency, polarisation and geometry gap.** Stage A3 therefore ran optical-only (task 3.2), which is the plan's own documented fallback — chosen on evidence rather than on unavailability. **If the team confirms RISAT-2B/2BR1 instead of EOS-04, those are X-band and this verdict inverts** — these become the right sources, so item 5's residual risk and this item must be re-read together. | Geo lead | 2026-08-29 |
| 9 | Prescribed benchmark test splits downloadable (VRSBench, RSVQA, CDVQA) | **Partly resolved** | VRSBench downloads freely from HF `xiang709/VRSBench` (6.1 GB, CC-BY-4.0) and includes the prescribed eval splits (`VRSBench_EVAL_vqa/Cap/referring.json`). **But it ships annotations only - zero imagery.** Its 142,390 train rows are LLaVA-style `conversations` referencing images that live in the separate DOTA and DIOR datasets, so it cannot train anything on its own. RSVQA-LR is available self-contained as `dmarsili/RSVQA-LR-2k` (174 MB, CC-BY-4.0, images embedded in the parquet) and was used for Track B v0. **CDVQA resolved 2026-08-29:** the official release (`github.com/YZHJessica/CDVQA`, **Apache-2.0**) is a plain git repo — Train/Val/Test/Test2 questions, answers and image indices download with `curl`, no Drive link, no form. Test = **39,686 questions over 968 image pairs**, 8 question types. Like VRSBench it ships **annotations only**; the pixels are the SECOND dataset's 512×512 bi-temporal tiles. Imagery obtained from the webdataset mirror `ljx620/CDVQA` (HF), whose per-sample JSON carries the official `question_id`, so **every sample is verified against the official annotation before use** (`training/prepare/cdvqa.py`) — 0 mismatches in the shards read so far. The mirror duplicates the image pair per question, so the full test split is ~32 GB for 968 pairs; the prepare script deduplicates and works from a partial download. **First-ever CDVQA measurement recorded in `docs/phase1-status.md`.** | Eval lead | 2026-08-29 |
| 10 | SIH 2026 timeline: internal deadline, grand finale dates, submission format | **Open** | Not yet confirmed with team/organizers. Fallback: compress the phase plan in doc `04` proportionally once the real deadline is known. | Team lead | Pending |
| 11 | Bhoonidhi registration approved; Cartosat-2S + RISAT products downloaded | **RESOLVED** | **Registration completed and products downloaded 2026-08-29.** On disk: Cartosat-2E MX (5132611), EOS-04 FRS-1 (226981731, quad-pol), EOS-04 MRS x2 (226981721, 247111021). Items 5 and 6 were both closed from this data. The products are held out as the cross-sensor generalisation set per docs/03 section 4.3 and are never trained on. | Geo lead | 2026-08-29 |
| 12 | GeoChat-7B / RS-LLaVA / LHRS-Bot downloadable for zero-shot baselines | **Open** | Not yet checked. Fallback: baseline against the un-finetuned base VLM only. | ML #2 | Pending — Phase 1 |

## Summary

- **Resolved: 6/12** - items 1 and 2 (dataset paper, 2026-08-27), items 5, 6 and
  11 (closed 2026-08-29 by reading real Bhoonidhi product metadata), and item 8
  (closed 2026-08-29 by pulling a real Umbra product's STAC metadata from the
  open bucket).
- **Open: 6/12** (items 3, 4, 7, 9, 10, 12). Every one has a documented, costed
  fallback per doc `03` section 6, and **none blocks Phase 2 or Phase 3**.
- **Item 9 is now down to one dataset.** CDVQA closed 2026-08-29 — official
  annotations are Apache-2.0 and `curl`-able, imagery obtained and verified
  against them, and the split is **measured for the first time**. Only
  VRSBench's missing DOTA imagery keeps the row from resolving outright.
- **The two plan-changing unknowns are gone.** Cartosat-2E MX is confirmed 4-band
  VNIR with no SWIR, so the SWIR-free fallback paths are the operative ones. EOS-04
  is confirmed C-band at 5.40 GHz, within 0.09% of Sentinel-1's 5.405 GHz, which
  means backscatter behaviour learned on S1 transfers to the target sensor almost
  directly. Both were assumptions; both now rest on primary evidence.
- **Item 8 resolved to a "no", which is more useful than the "yes" it was
  looking for.** High-resolution SAR is freely available and permissively
  licensed - the question as written - but every source is **X-band** while the
  target sensor is **C-band**, a 1.79x wavelength difference against the 0.09%
  match that made item 5 comfortable. The plan offered "SpaceNet 6 / Umbra, or
  optical-only with the limitation documented" as if the fallback were the
  weaker branch; the measurement says optical-only is the **correct** branch for
  EOS-04, and Stage A3 (task 3.2) took it on that basis rather than on
  unavailability.

  **This inverts if the sensor changes.** Item 5 records a residual risk that
  ISRO may use RISAT-2B/2BR1, which *are* X-band. Under that sensor these three
  sources become exactly right, and Stage A3 should be redone against them. The
  two items must be read together, and whoever confirms the sensor should come
  back to this row. **Update 2026-08-29:** that risk is now narrowed by
  elimination — of the four candidate RISATs, one is decommissioned, one failed
  at launch, and the two X-band ones are not publicly distributed, leaving
  EOS-04 as the only openly-served candidate. See §"Which RISAT" below. The
  inversion is less likely than it was, and it is still not ruled out.

## Which RISAT — item 5's residual risk, narrowed by elimination (2026-08-29)

The PS names the evaluation set only as "Cartosat-2S optical + **RISAT** SAR".
Item 5 read C-band off a real EOS-04 product, but that proves what *we*
downloaded, not what *SAC* will hand the judges. Four RISAT-family SAR
satellites could supply that half of the pair. Three can be struck or
de-prioritised on public record:

| candidate | band | status | can it supply an evaluation pair? |
|---|---|---|---|
| RISAT-1 | C | **decommissioned 2017** after its 5-year life | No — nothing new since 2017 |
| RISAT-1B / EOS-09 | C | **launch failed**, 2025-05-18 | No — never reached orbit |
| RISAT-2B, RISAT-2BR1 | **X** | operational (WMO OSCAR, checked 2026-02-02), but **"Data not ordinarily available to the public"**; neither appears in Bhoonidhi's civil catalogue, which lists RISAT-1 CRS/MRS and EOS-04 and stops there | Only from SAC's own restricted holdings |
| **EOS-04 / RISAT-1A** | **C, 5.40 GHz measured** | operational since 2022-02-14; **openly distributed on Bhoonidhi**; three products already on our disk | **Yes** |

**The balance of evidence favours EOS-04**, which keeps item 8's verdict intact:
every open high-res SAR source is X-band, the target is C-band, and Stage A3's
optical-only arm is the correct branch rather than the fallback branch.

**The counter-argument, stated fairly.** The PS pairs the SAR with Cartosat-2S
at 0.65–1.6 m. EOS-04's finest civil mode is FRS-1 at ~2.5 m; RISAT-2B reaches
0.35 m. A *resolution-matched* pair argues for the X-band satellites, and SAC
can draw on restricted data that the public catalogue does not show. That is
why this is **narrowed, not closed**.

**The one question to put to the team**, and what each answer costs:

> Is the SAR in the ISRO/SAC evaluation set **EOS-04 / RISAT-1A**, or
> **RISAT-2B / 2BR1**?

- **EOS-04, or any C-band RISAT** → nothing changes. Item 8 stands, Stage A3
  stays optical-only, and the σ⁰ behaviour learned on Sentinel-1 transfers at a
  0.09% frequency difference.
- **RISAT-2B / 2BR1** → **item 8 inverts.** Umbra, Capella and SpaceNet 6 stop
  being the wrong band and become exactly the right sources, and Stage A3
  should be redone against 0.25 m X-band SAR. That is roughly 2–4 GPU-h plus
  the download, and it is the only branch in the plan that a wrong answer here
  makes wasted work.

Sources: [WMO OSCAR RISAT-2B](https://space.oscar.wmo.int/satellites/view/risat_2b),
[WMO OSCAR RISAT-2BR1](https://space.oscar.wmo.int/satellites/view/risat_2br1),
[Gunter's Space Page, RISAT-1/1A/1B](https://space.skyrocket.de/doc_sdat/risat-1.htm),
[Bhoonidhi](https://bhoonidhi.nrsc.gov.in/bhoonidhi/home.html).

## NEW RISK — SECOND states no licence at all (discovered 2026-08-30)

Closing the CDVQA zero needs a semantic change head, and the only labels for
CDVQA's six change classes are the **SECOND** dataset's pixel annotations.
SECOND is freely downloadable — [captain-whu.github.io/SCD](https://captain-whu.github.io/SCD/)
serves a 2.4 GB `SECOND_train_set.rar` from Google Drive with no form and no
registration — but **the project page, the paper (arXiv 2010.05687) and the
archive state no licence, no terms of use and no citation requirement.** Not
a restrictive licence: *no* licence.

Why this matters here and not for every dataset: **the PS lists model weights
as a deliverable.** That is the same reason item 8 flagged SpaceNet 6's
share-alike. Weights trained on unlicensed data inherit an unresolved
question, and "everyone does it" is not an answer we should put in front of an
ISRO reviewer without having noticed it.

Why it is not blocking, and the work proceeds:

- **The PS itself prescribes CDVQA as an evaluation benchmark, and CDVQA is
  built on SECOND imagery.** Entrants cannot be expected to score on CDVQA
  without touching SECOND, so its use is implicitly sanctioned by the problem
  statement.
- CDVQA's own annotations — the questions, answers and splits — are
  **Apache-2.0**, and those are what we are graded against.
- Academic use of SECOND is universal in this literature; the risk is to
  redistribution of derived weights, not to evaluating on it.

Action: one email to the authors (`kunpingyang@whu.edu.cn`,
`guisong.xia@whu.edu.cn`) asking for explicit terms, sent alongside the team
questions. Until an answer arrives, **train on it and say so in the report**,
and keep the option of publishing the head's code and metrics without the
weights if the answer is restrictive. Flag to the team lead with the Cartosat
pricing risk below.

## NEW RISK (discovered 2026-08-29, not in the original 12 items)

**Cartosat-2S full-scene data is priced, not open, for our likely user class.**
Under the Indian Space Policy 2023 implementation at Bhoonidhi, data finer than
5 m resolution is **open only to Indian Government Entities** and is **priced for
Non-Government Entities** (Cartosat-2S ordering is handled by NSIL). A student
competition team is most likely an NGE, so routine bulk Cartosat-2S acquisition
may cost money or require institutional sponsorship.

Why this is not currently blocking:

- The **Cartosat-2S MX sample product is a free direct download**, which is
  sufficient to answer item 6 (band composition) — the only thing we actually
  need Cartosat data for at this stage.
- Doc `03` §5 already specifies that Bhoonidhi products are a **small curated
  qualitative/OOD set that is never trained on**. A handful of scenes is enough;
  we were never going to need bulk Cartosat data.
- SAR is better off: **RISAT-1 MRS/CRS are open to all**, and **EOS-04
  (RISAT-1A, C-band)** samples are freely downloadable.

Action: if bulk high-res optical is ever needed beyond the samples, route the
request through the team's academic institution (which may qualify differently)
or budget for NSIL pricing. Flag to the team lead alongside item 10.

## Tooling

`scripts/inspect_product.py` reads a downloaded product and prints exactly what
items 5 and 6 need — band count, band descriptions, CRS, ground sample distance,
subdatasets, and any sensor/frequency/polarization metadata tags, plus a list of
sidecar XML/metadata files (where RISAT frequency and look count usually live,
rather than in the raster header):

```
python scripts/inspect_product.py <path-to-product-or-directory>
```

## Recommended next actions

1. ~~Download the free Cartosat-2S MX and EOS-04 sample products~~ — **done
   2026-08-29**, closing items 5, 6 and 11.
2. ~~Check SpaceNet 6 / Umbra / Capella~~ — **done 2026-08-29**, closing item 8.
   Reproduce with:
   `curl -s "https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/?list-type=2&max-keys=8&prefix=sar-data/tasks/"`
   then fetch any `*.stac.v2.json` and read `sar:frequency_band` and
   `sar:center_frequency`. No AWS account needed.
3. **Confirm which SAR sensor ISRO/SAC will actually use** (item 5's residual
   risk). Narrowed 2026-08-29 by elimination — §"Which RISAT" above shows
   EOS-04 is the only RISAT-family SAR that is both operational and publicly
   distributed — but **not closed**: the team still has to answer one sentence,
   because a resolution-matched pair with Cartosat-2S argues for the restricted
   X-band satellites and SAC can reach data the public catalogue cannot. The
   answer decides whether Stage A3 stays optical-only or is redone against
   0.25 m SAR.
4. Confirm the SIH 2026 submission deadline and format (item 10) with the
   team/organizers — this affects how the W0–W14 phase plan in
   `docs/04-Implementation-Plan.md` should be paced.
5. Items 3, 4, 7, 9, 12 are model/dataset-availability checks with zero-cost
   fallbacks already specified — verify opportunistically rather than blocking
   on them.
