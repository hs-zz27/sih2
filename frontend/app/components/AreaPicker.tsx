'use client';

/**
 * Draw an area of interest over the attached scene's footprint.
 *
 * Sits inline under the composer, not in a modal: the scene appears as soon as
 * it is attached, so choosing an area is part of setting up the run rather
 * than a dialog you have to know to open.
 *
 * The scene itself is drawn on the map, not just its outline. A rectangle over
 * a street map tells you where the imagery is and nothing about what is in it,
 * and you cannot choose an area of something you cannot see. The preview comes
 * from `to_rgb_preview`, the same path the model is fed, so the pixels being
 * selected are the pixels that will be read.
 *
 * Separate from MapView on purpose. MapView renders a *finished* run — its
 * overlays, its basemap probe, its layer switching — and is mounted inside the
 * results deck. Sharing one component between the two would mean a pile of
 * mode flags in a file that is already the most intricate on the frontend.
 *
 * The footprint comes from `POST /probe`, which opens the file's header and
 * throws the pixels away. The browser could parse a GeoTIFF header itself, but
 * that means shipping a parser to read something the server already knows, and
 * the server is the authority on whether a file is georeferenced — it is the
 * one that will refuse the run.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import XYZ from 'ol/source/XYZ';
import ImageLayer from 'ol/layer/Image';
import Static from 'ol/source/ImageStatic';
import Feature from 'ol/Feature';
import Polygon, { fromExtent } from 'ol/geom/Polygon';
import Draw, { createBox } from 'ol/interaction/Draw';
import { defaults as defaultInteractions } from 'ol/interaction';
import { Graticule } from 'ol/layer';
import { getIntersection, containsExtent, isEmpty } from 'ol/extent';
import { transformExtent } from 'ol/proj';
import { Fill, Stroke, Style } from 'ol/style';
import CircleStyle from 'ol/style/Circle';
import 'ol/ol.css';

import type { Bounds } from '../lib/footprint';

const BASEMAP_URL =
  process.env.NEXT_PUBLIC_BASEMAP_URL ??
  'https://tile.openstreetmap.org/{z}/{x}/{y}.png';

/** Mirrors MIN_AOI_PX in satquery/api/main.py. */
const MIN_AOI_PX = 64;

export type ProbedScene = {
  name: string;
  crs?: string;
  georeferenced: boolean;
  lonlat_bounds?: Bounds | null;
  width?: number;
  height?: number;
  multi_file?: boolean;
  /** A small PNG of the scene as a data URL, rendered by the API. */
  preview?: string | null;
  /**
   * Where to draw that PNG, in EPSG:3857.
   *
   * Sent by the server rather than derived here from `lonlat_bounds`: the
   * preview is reprojected, so its extent is the warped one. Deriving it from
   * the lon/lat envelope would put the imagery back where it was wrong.
   */
  preview_extent?: [number, number, number, number] | null;
};

/** Metres per degree of longitude at a given latitude, near enough for a label. */
function boxSizeKm(box: Bounds): [number, number] {
  const [west, south, east, north] = box;
  const midLat = ((south + north) / 2) * (Math.PI / 180);
  const kmPerDegLat = 110.574;
  const kmPerDegLon = 111.32 * Math.cos(midLat);
  return [(east - west) * kmPerDegLon, (north - south) * kmPerDegLat];
}

export default function AreaPicker({
  scenes,
  value,
  onChange,
}: {
  scenes: ProbedScene[];
  value: Bounds | null;
  onChange: (box: Bounds | null) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const mapRef = useRef<Map | null>(null);
  const drawnRef = useRef<VectorSource | null>(null);
  const [box, setBox] = useState<Bounds | null>(value);
  const [tooSmall, setTooSmall] = useState(false);
  const [basemapFailed, setBasemapFailed] = useState(false);

  const located = scenes.filter((s) => s.georeferenced && s.lonlat_bounds);

  /* The union of every located scene: with a bi-temporal pair the box has to
     sit inside both, so the map opens on the ground they share. */
  const footprint: Bounds | null = located.length
    ? (located.reduce<Bounds>(
        (acc, s) => {
          const b = s.lonlat_bounds!;
          return [
            Math.min(acc[0], b[0]),
            Math.min(acc[1], b[1]),
            Math.max(acc[2], b[2]),
            Math.max(acc[3], b[3]),
          ];
        },
        located[0].lonlat_bounds!,
      ) as Bounds)
    : null;

  /* The smallest box worth allowing, from the coarsest scene's resolution:
     the API refuses anything under MIN_AOI_PX, so the picker should not let
     you draw one and find out afterwards. */
  const minSpanDeg = located.reduce((worst, s) => {
    const b = s.lonlat_bounds!;
    const px = Math.min(s.width ?? 0, s.height ?? 0);
    if (!px) return worst;
    const degPerPx = Math.min(b[2] - b[0], b[3] - b[1]) / px;
    return Math.max(worst, degPerPx * MIN_AOI_PX);
  }, 0);

  useEffect(() => {
    if (!container.current || mapRef.current || !footprint) return;

    const extent = transformExtent(footprint, 'EPSG:4326', 'EPSG:3857');

    const outline = new VectorLayer({
      source: new VectorSource({ features: [new Feature(fromExtent(extent) as Polygon)] }),
      style: new Style({
        stroke: new Stroke({ color: 'rgba(232,195,158,0.85)', width: 1 }),
      }),
    });

    const drawn = new VectorSource();
    drawnRef.current = drawn;
    const drawnLayer = new VectorLayer({
      source: drawn,
      style: new Style({
        stroke: new Stroke({ color: '#E8C39E', width: 2 }),
        fill: new Fill({ color: 'rgba(232,195,158,0.14)' }),
      }),
    });

    // Each located scene painted at its own extent, over the basemap. Two
    // scenes in a bi-temporal pair overlap, and seeing that overlap is the
    // point — it is the ground the crop has to sit inside.
    const sceneLayers = located
      .filter((s) => s.preview && s.preview_extent)
      .map(
        (s) =>
          new ImageLayer({
            source: new Static({
              url: s.preview!,
              // The server's warped extent, used as given. Recomputing it from
              // `lonlat_bounds` would undo the reprojection.
              imageExtent: s.preview_extent!,
              projection: 'EPSG:3857',
            }),
            opacity: 0.92,
          }),
      );

    /* The basemap is optional context; the scene is the point.
       MapView probes for a reachable tile before deciding. Probing here would
       put a network round trip in front of a panel that opens as soon as a
       file is attached, so this reacts instead: if tiles start failing — no
       network, a captive portal, a blocked host — the layer is dropped and a
       graticule takes its place, so the scene is never floating in a void
       waiting for a basemap that is not coming. */
    const graticule = new Graticule({
      strokeStyle: new Stroke({ color: 'rgba(154,146,158,0.28)', width: 1 }),
      showLabels: true,
      wrapX: false,
    });
    graticule.setVisible(!BASEMAP_URL);

    const tiles = BASEMAP_URL
      ? new TileLayer({ source: new XYZ({ url: BASEMAP_URL, crossOrigin: 'anonymous' }) })
      : null;
    if (tiles) {
      let fellBack = false;
      tiles.getSource()!.on('tileloaderror', () => {
        if (fellBack) return;
        fellBack = true;
        tiles.setVisible(false);
        graticule.setVisible(true);
        setBasemapFailed(true);
      });
    }

    const map = new Map({
      target: container.current,
      layers: [
        ...(tiles ? [tiles] : []),
        graticule,
        ...sceneLayers,
        outline,
        drawnLayer,
      ],
      view: new View({ center: [0, 0], zoom: 2 }),
      // Same reasoning as MapView: this sits in a page, the wheel is the
      // page's. Pinch and the ± buttons still zoom.
      interactions: defaultInteractions({ mouseWheelZoom: false }),
    });
    map.getView().fit(extent, { padding: [32, 32, 32, 32], maxZoom: 16 });
    mapRef.current = map;

    const draw = new Draw({
      source: drawn,
      type: 'Circle',
      geometryFunction: createBox(),
      // Drag, not click-move-click. OpenLayers defaults a box to two separate
      // clicks, which is a fine convention but not the one the instruction on
      // screen gives — and the instruction is what people follow.
      freehand: true,
      // The interaction needs its own style, not just the layer the finished
      // box lands in. Without this OpenLayers draws its stock editing style —
      // a blue dot parked at the cursor over the imagery, which reads as a
      // marker on the scene rather than as a cursor.
      style: new Style({
        stroke: new Stroke({ color: '#E8C39E', width: 1.5, lineDash: [4, 3] }),
        fill: new Fill({ color: 'rgba(232,195,158,0.10)' }),
        image: new CircleStyle({
          radius: 2.5,
          fill: new Fill({ color: 'rgba(232,195,158,0.9)' }),
        }),
      }),
    });

    // One box at a time: drawing again replaces it rather than stacking
    // rectangles nobody can tell apart.
    draw.on('drawstart', () => drawn.clear());
    draw.on('drawend', (event) => {
      const drawnExtent = event.feature.getGeometry()!.getExtent();
      const clipped = getIntersection(drawnExtent, extent);
      if (isEmpty(clipped)) {
        drawn.clear();
        setBox(null);
        return;
      }
      // Clamped to the footprint, because the API will refuse the part that
      // hangs off the scene and there is no reason to let you draw it.
      event.feature.setGeometry(fromExtent(clipped) as Polygon);
      const lonlat = transformExtent(clipped, 'EPSG:3857', 'EPSG:4326') as Bounds;
      setBox(lonlat);
      const small =
        minSpanDeg > 0 &&
        (lonlat[2] - lonlat[0] < minSpanDeg || lonlat[3] - lonlat[1] < minSpanDeg);
      // Reported up as soon as it is drawn — there is no confirm step in an
      // inline panel, and a box the API would refuse is not sent.
      onChange(small ? null : lonlat);
      setTooSmall(
        minSpanDeg > 0 &&
          (lonlat[2] - lonlat[0] < minSpanDeg || lonlat[3] - lonlat[1] < minSpanDeg),
      );
    });
    map.addInteraction(draw);

    if (value) {
      const restored = transformExtent(value, 'EPSG:4326', 'EPSG:3857');
      if (containsExtent(extent, restored)) {
        drawn.addFeature(new Feature(fromExtent(restored) as Polygon));
      }
    }

    return () => {
      map.setTarget(undefined);
      mapRef.current = null;
    };
  }, [footprint, minSpanDeg]);
  /* eslint-disable-line react-hooks/exhaustive-deps */

  const clear = useCallback(() => {
    drawnRef.current?.clear();
    setBox(null);
    setTooSmall(false);
    onChange(null);
  }, [onChange]);

  const size = box ? boxSizeKm(box) : null;
  const unlocated = scenes.filter((s) => !s.georeferenced);

  return (
    <section className="scenebox" aria-label="Attached scene">
      <div className="scenebox-head">
        <span className="label">Attached scene</span>
        <span className="spacer" />
        <span className="meta">
          {located.length} located · drag on the image to crop
        </span>
      </div>

      {footprint ? (
        <>
          <div ref={container} className="scenebox-map" />
          <div className="scenebox-foot">
            <span className="composer-hint">
              {box && size
                ? `area · ${size[0].toFixed(1)} × ${size[1].toFixed(1)} km · ${box
                    .map((v) => v.toFixed(4))
                    .join(', ')}`
                : 'the whole scene will be used unless you drag a box on it'}
            </span>
            <span className="spacer" />
            {box && (
              <button type="button" className="composer-attach" onClick={clear}>
                <svg className="icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M6 6l12 12M18 6L6 18" />
                </svg>
                Use whole scene
              </button>
            )}
          </div>
          {tooSmall && (
            <p className="composer-error" role="alert">
              That area is under {MIN_AOI_PX} pixels on a side at this scene&apos;s
              resolution — the run would have nothing to look at. Drag a larger one.
            </p>
          )}
          {basemapFailed && (
            <p className="cap">
              No basemap — the tile server is unreachable. The scene and its
              coordinates are unaffected; only the map behind them is missing.
            </p>
          )}
          {unlocated.length > 0 && (
            <p className="cap">
              {unlocated.map((s) => s.name).join(', ')} carries no CRS and will be sent
              whole.
            </p>
          )}
        </>
      ) : (
        <p className="cap">
          No attached scene carries a coordinate reference system, so there is nothing to
          place on a map. The files will be sent whole.
        </p>
      )}
    </section>
  );
}
