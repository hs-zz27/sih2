/**
 * The union of every ingested scene's lon/lat bounds.
 *
 * `ingest.images[]` carries `lonlat_bounds` as `[west, south, east, north]`
 * for any scene that is georeferenced, and `null` for one that is not. Two
 * scenes in a bi-temporal pair overlap but are rarely identical, so the map
 * should open on the box that contains both rather than on whichever happened
 * to be listed first.
 *
 * Returns null when no scene carries a CRS — which the map reports as a
 * different situation from "the run wrote no overlays".
 */
export type Bounds = [number, number, number, number];

export function sceneFootprint(images: any[] | undefined | null): Bounds | null {
  const boxes: Bounds[] = (images ?? [])
    .map((image) => image?.lonlat_bounds)
    .filter(
      (b: any): b is Bounds =>
        Array.isArray(b) && b.length === 4 && b.every((n) => Number.isFinite(n)),
    );
  if (boxes.length === 0) return null;

  return boxes.reduce<Bounds>(
    (acc, b) => [
      Math.min(acc[0], b[0]),
      Math.min(acc[1], b[1]),
      Math.max(acc[2], b[2]),
      Math.max(acc[3], b[3]),
    ],
    boxes[0],
  );
}


/**
 * Whether any ingested scene claims a coordinate reference system.
 *
 * Needed because a missing `lonlat_bounds` has two very different causes, and
 * saying the wrong one is worse than saying nothing:
 *
 * * the scene genuinely has no CRS — a PNG, or a GeoTIFF written without one;
 * * the API build in front of us does not report bounds at all.
 *
 * The second is real: `lonlat_bounds` arrived in `pre-import:geo-answering` on the
 * phase-0-closeout branch and is absent from builds made before it. Reading
 * its absence as "this scene carries no CRS" tells the user something false
 * about their file. `georeferenced` and `crs` are present either way, so they
 * are what separates the two.
 */
export function hasGeoreference(images: any[] | undefined | null): boolean {
  return (images ?? []).some(
    (image) =>
      image?.georeferenced === true ||
      (typeof image?.crs === 'string' &&
        image.crs.trim() !== '' &&
        image.crs.toUpperCase() !== 'UNKNOWN'),
  );
}
