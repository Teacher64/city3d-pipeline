#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_osm_srtm.py — скачивает OSM и SRTM для района LAS.

Использование:
    python download_osm_srtm.py <input.las> [output_dir]

Определяет bbox из LAS, скачивает:
  • OSM buildings  (полигоны зданий)
  • OSM roads      (линейные объекты)
  • OSM waterways  (реки/каналы)
  • SRTM 30m DEM   (высотная модель)
  • SRTM 30m hillshade

Сохраняет в <output_dir> как GeoJSON + GeoTIFF.
"""
import sys
import json
from pathlib import Path
from datetime import datetime

def get_las_bbox_and_crs(las_path):
    """Возвращает (bbox_utm, bbox_wgs84, epsg)."""
    import laspy
    from pyproj import CRS, Transformer

    las = laspy.read(las_path)
    # Определяем CRS из VLR
    crs = None
    for vlr in las.header.vlrs:
        if vlr.record_id == 2112:
            try:
                wkt = vlr.record_data.decode("utf-8", errors="ignore")
                crs = CRS.from_wkt(wkt)
                break
            except Exception:
                pass

    if crs is None:
        # Fallback: UTM 39N по умолчанию (из вашей задачи)
        crs = CRS.from_epsg(32639)
        print("[WARN] CRS не найден в LAS, использую EPSG:32639")

    xmin = float(las.header.x_min); xmax = float(las.header.x_max)
    ymin = float(las.header.y_min); ymax = float(las.header.y_max)
    bbox_utm = (xmin, ymin, xmax, ymax)

    # Переводим в WGS84
    if crs.to_epsg() == 4326:
        bbox_wgs84 = bbox_utm
        epsg = 4326
    else:
        t = Transformer.from_crs(crs, 4326, always_xy=True)
        lon_min, lat_min = t.transform(xmin, ymin)
        lon_max, lat_max = t.transform(xmax, ymax)
        bbox_wgs84 = (lon_min, lat_min, lon_max, lat_max)
        epsg = crs.to_epsg()

    return bbox_utm, bbox_wgs84, epsg


def download_osm(bbox_wgs84, out_dir):
    """Скачивает OSM через Overpass API (buildings, roads, waterways)."""
    import urllib.request, urllib.parse

    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    bbox_str = f"{lat_min},{lon_min},{lat_max},{lon_max}"
    print(f"[OSM] bbox: {bbox_str}")

    overpass_query = f"""
[out:json][timeout:120];
(
  way["building"]({bbox_str});
  relation["building"]({bbox_str});
  way["highway"]({bbox_str});
  way["waterway"]({bbox_str});
  way["natural"="water"]({bbox_str});
);
out geom;
"""

    results = {}
    for layer, query in [
        ("buildings", f'[out:json][timeout:120];(way["building"]({bbox_str});relation["building"]({bbox_str}););out geom;'),
        ("roads", f'[out:json][timeout:120];(way["highway"]({bbox_str}););out geom;'),
        ("waterways", f'[out:json][timeout:120];(way["waterway"]({bbox_str});way["natural"="water"]({bbox_str}););out geom;'),
    ]:
        print(f"[OSM] скачиваю {layer}…")
        data = urllib.parse.urlencode({"data": query}).encode("utf-8")
        req = urllib.request.Request(
            "https://overpass.kumi.systems/api/interpreter",
            data=data,
            headers={"User-Agent": "City3D/1.0 (contact: michel@local)"})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            # Конвертируем в GeoJSON
            features = []
            for el in payload.get("elements", []):
                if el.get("type") == "way" and "geometry" in el:
                    coords = [[p["lon"], p["lat"]] for p in el["geometry"]]
                    if len(coords) < 2:
                        continue
                    # Полигон или линия
                    is_area = layer in ("buildings", "waterways") and \
                              len(coords) >= 4 and coords[0] == coords[-1]
                    geom = {
                        "type": "Polygon" if is_area else "LineString",
                        "coordinates": [coords] if is_area else coords,
                    }
                    features.append({
                        "type": "Feature",
                        "properties": {
                            "osm_id": el.get("id"),
                            "tags": el.get("tags", {}),
                        },
                        "geometry": geom,
                    })
            geojson = {"type": "FeatureCollection", "features": features}
            out_path = out_dir / f"osm_{layer}.geojson"
            out_path.write_text(
                json.dumps(geojson, ensure_ascii=False, indent=2),
                encoding="utf-8")
            results[layer] = out_path
            print(f"[OSM] {layer}: {len(features)} объектов → {out_path.name}")
        except Exception as e:
            print(f"[OSM] {layer}: ОШИБКА — {e}")
            results[layer] = None

    return results


def download_srtm(bbox_wgs84, out_dir):
    """Скачивает SRTM 30m DEM через elevation API или USGS."""
    lon_min, lat_min, lon_max, lat_max = bbox_wgs84
    print(f"[SRTM] bbox: {lat_min:.4f},{lon_min:.4f} → {lat_max:.4f},{lon_max:.4f}")

    srtm_path = out_dir / "srtm_dem.tif"

    # Способ 1: пакет elevation (простой)
    try:
        import elevation
        print("[SRTM] использую пакет 'elevation'…")
        elevation.clip(
            bounds=(lon_min, lat_min, lon_max, lat_max),
            output=str(srtm_path))
        elevation.clean()
        if srtm_path.exists():
            print(f"[SRTM] готово → {srtm_path}")
            return srtm_path
    except ImportError:
        print("[SRTM] пакет 'elevation' не установлен, "
              "пробую USGS напрямую")
    except Exception as e:
        print(f"[SRTM] elevation.clip: {e}")

    # Способ 2: opentopography API
    try:
        import requests
        import os
        api_key = os.environ.get("OPENTOPO_API_KEY", "")
        if not api_key:
            print("[SRTM] нет OPENTOPO_API_KEY — пропускаю")
            return None
        url = "https://portal.opentopography.org/API/globaldem"
        params = {
            "demtype": "SRTMGL1",
            "south": lat_min, "north": lat_max,
            "west": lon_min, "east": lon_max,
            "outputFormat": "GTiff",
            "API_Key": api_key,
        }
        r = requests.get(url, params=params, timeout=300)
        if r.status_code == 200:
            srtm_path.write_bytes(r.content)
            print(f"[SRTM] OpenTopography OK → {srtm_path}")
            return srtm_path
        else:
            print(f"[SRTM] OpenTopography HTTP {r.status_code}")
    except Exception as e:
        print(f"[SRTM] opentopography: {e}")

    return None


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    las_path = Path(sys.argv[1]).expanduser()
    if not las_path.exists():
        print(f"[ERROR] файл не найден: {las_path}"); sys.exit(1)

    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else \
              Path.home() / "projects" / "city3d_models" / "references"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] выходная папка: {out_dir}")

    bbox_utm, bbox_wgs84, epsg = get_las_bbox_and_crs(las_path)
    print(f"[LAS] EPSG:{epsg}")
    print(f"[LAS] UTM bbox : {bbox_utm}")
    print(f"[LAS] WGS84 bbox: {bbox_wgs84}")

    # Сохраняем метаданные для дальнейшего использования
    meta = {
        "source_las": str(las_path),
        "epsg": epsg,
        "bbox_utm": list(bbox_utm),
        "bbox_wgs84": list(bbox_wgs84),
        "downloaded_at": datetime.now().isoformat(),
    }
    (out_dir / "reference_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== OSM ===")
    osm_files = download_osm(bbox_wgs84, out_dir)

    print("\n=== SRTM ===")
    srtm_file = download_srtm(bbox_wgs84, out_dir)

    print("\n" + "=" * 60)
    print("ИТОГ:")
    for k, v in osm_files.items():
        print(f"  OSM {k:12s}: {v if v else '—'}")
    print(f"  SRTM DEM    : {srtm_file if srtm_file else '—'}")
    print(f"  meta        : {out_dir / 'reference_meta.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
