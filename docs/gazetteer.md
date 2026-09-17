# Gazetteer: naming the region a scene falls in

`ImageMeta.lonlat_bounds` says where a scene is in degrees. The gazetteer
(`satquery/geo/gazetteer.py`) turns those degrees into words, so a description
can end "The scene lies in Lithuania, Koppen climate zone Dfb (cold, no dry
season, warm summer)." rather than stopping at the coordinate.

It is **off by default and the data is not in this repository.** Without it,
answers report the coordinate and say nothing about the region. That is the
correct behaviour, not a degraded one.

## Why the data is not vendored

The layers are third-party, they are large, and they carry their own licences
and attribution requirements. Vendoring them would put a licence obligation
into every clone of this repository and would commit the project to
redistributing data it does not own. The operator installs them instead, the
same way model weights are installed.

## Activation

```bash
export SATQUERY_GAZETTEER=/path/to/gazetteer
```

The directory holds up to two layers. Either may be absent; a directory with
only `climate.tif` reports climate and stays silent about the country.

```
gazetteer/
  country.tif     # categorical raster, EPSG:4326
  country.json    # legend
  climate.tif
  climate.json
```

## Raster format

* **EPSG:4326**, whole-world coverage, any resolution.
* **Single band, integer**, one code per region.
* **`nodata` set** in the header. Unmapped cells - ocean, unclaimed area -
  must be nodata, not `0` masquerading as a region. `0` is also treated as
  blank, but relying on that is worse than setting the header correctly.

## Legend format

```json
{
  "labels":       {"1": "Lithuania", "2": "Latvia"},
  "descriptions": {"1": "cold, no dry season, warm summer"},
  "attribution":  "Beck et al. (2018) Koppen-Geiger maps, CC BY 4.0"
}
```

* `labels` maps the raster's integer codes to names. **A code with no label is
  not reported** - printing the raw integer would be worse than silence.
* `descriptions` is optional and used for the climate layer, to expand a
  terse code like `Dfb` into readable words.
* `attribution` travels into `Trace.data_sources` on every answer that used
  the layer. Fill it in. A CC BY layer requires the credit to travel with the
  output, and an answer that names a country without saying whose boundaries
  it used is not reproducible either.

A missing, unreadable or corrupt legend degrades that layer to "no names"
rather than failing the query.

## Candidate datasets

These are the shapes that fit; **verify licence and provenance yourself
before installing** - this file names them, it does not vouch for them.

| Layer | Candidate | Note |
|---|---|---|
| `climate` | Beck et al. (2018) Koppen-Geiger present-day maps | Published as GeoTIFF, CC BY 4.0. Drops in with only a legend to write. |
| `country` | Natural Earth admin-0 | Public domain, but published as **vector**. Needs rasterising to EPSG:4326 first, which needs GDAL - do it once, offline, outside this process. |

There is deliberately no `scripts/fetch_gazetteer.py`. The existing fetchers
(`fetch_models.py`, `fetch_datasets.py`) pin publisher digests that were
verified when they were written; a fetcher here would carry URLs and
checksums that nobody has verified, which is worse than no fetcher. Install
the files, then check them:

```bash
python scripts/check_gazetteer.py --dest /path/to/gazetteer
```

## Borders

A categorical raster puts a hard edge where the world has a soft one. A scene
3 km inside one country and a scene 3 km inside its neighbour can land in the
same cell.

So a lookup reads a **3x3 window, not a pixel**. The centre cell supplies the
label; the window decides whether that label is trustworthy. When the window
disagrees the label is flagged ambiguous and the answer hedges - "close to a
national boundary in the reference data - the nearest match is Lithuania" -
rather than asserting a country the data does not establish. This is the same
principle as `landcover_v1` abstaining between its thresholds.

Resolution therefore matters for honesty, not only for accuracy: a coarse
raster will hedge more often, which is the correct response to being coarse.

## What this does not do

* **It does not name cities, regions or landmarks.** Only what a layer
  supplies.
* **It does not reach the network.** Ever - the offline guarantee in task 3.9
  covers this module, and `tests/test_offline.py` blocks the socket layer.
* **It does not help an ungeoreferenced upload.** A PNG or JPEG carries no
  CRS, so there is no coordinate to look up and nothing here can run. The
  answer says so instead.
