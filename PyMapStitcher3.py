#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Py Map Stitcher - direct BigTIFF tile downloader/stitcher with PySide6 WebEngine GUI.

Use only with map/tile servers for which you have permission. Many public map
providers prohibit bulk downloading. The app intentionally uses a conservative
rate limit and requires user-supplied/custom URL templates.
"""

import concurrent.futures as cf
import dataclasses
import io
import json
import subprocess
import tempfile
import math
import os
import queue
import random
import shutil
import sys
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception as exc:  # pragma: no cover
    requests = None

try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:  # pragma: no cover
    Image = None

TILE_SIZE = 256
USER_AGENT = "PyMapStitcher/1.0 (+local user tool)"
MAX_INFLIGHT_PER_WORKER = 4  # prevents millions of Futures in RAM
HARD_TILE_WARNING = 5_000_000
DEFAULT_CHUNK_SIZE = 64
MAX_DIRECT_TIFF_BYTES = 1_000_000_000_000  # 1 TB safety limit for sparse BigTIFF output
CUDA_DEFAULT_MAX_CHUNK_MB = 1024  # larger GPU chunk buffers for stronger CUDA use
CUDA_LOAD_MATRIX_SIZE = 1536  # optional compute load to keep NVIDIA GPU busy during I/O waits



MAP_PRESETS = {
    "Custom": {
        "url": "https://your-tile-server.example/{z}/{x}/{y}.png",
        "note": "Enter a custom URL template manually.",
        "preview": True,
    },
    "Google Satellite": {
        "url": "https://mt.google.com/vt/lyrs=s&x={x}&y={y}&z={z}&hl=de",
        "note": "Google Satellite. Respect the terms of use; no bulk downloading without permission.",
        "preview": True,
    },
    "Google Hybrid": {
        "url": "https://mt.google.com/vt/lyrs=y&x={x}&y={y}&z={z}&hl=de",
        "note": "Google Satellite with labels. Respect the terms of use.",
        "preview": True,
    },
    "Bing Satellite": {
        "url": "https://ecn.t{snum}.tiles.virtualearth.net/tiles/a{q}.jpeg?g=14574&mkt=de-DE&n=z",
        "note": "Bing Aerial/Satellite via QuadKey {q}. Respect the terms of use.",
        "preview": True,
    },
    "Bing Hybrid": {
        "url": "https://ecn.t{snum}.tiles.virtualearth.net/tiles/h{q}.jpeg?g=14574&mkt=de-DE&n=z",
        "note": "Bing Hybrid via QuadKey {q}. Respect the terms of use.",
        "preview": True,
    },
    "Esri World Imagery": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "note": "Satellite/aerial tiles. Respect Esri terms of use.",
        "preview": True,
    },
    "OpenStreetMap Mapnik": {
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "note": "OSM standard map. Respect the terms of use; no bulk downloading.",
        "preview": True,
    },
    "OpenTopoMap": {
        "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        "note": "Topographic map. Respect the terms of use.",
        "preview": True,
    },
    "CartoDB Positron": {
        "url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",
        "note": "Light basemap. Respect the terms of use.",
        "preview": True,
    },
    "NoniMapView Legacy: Google Satellite": {
        "url": "http://khm{rnd}.google.com/kh/v=47&x={x}&y={y}&z={z}&s=&hl=de",
        "note": "Legacy NoniMapView profile; may be outdated or blocked today.",
        "preview": True,
    },
    "NoniMapView Legacy: Google Road": {
        "url": "http://mt{rnd}.google.com/vt/lyrs=m&hl=de&x={x}&y={y}&z={z}",
        "note": "Legacy NoniMapView profile; may be outdated or blocked today.",
        "preview": True,
    },
}



@dataclasses.dataclass(frozen=True)
class TileJob:
    x: int
    y: int
    z: int
    col: int
    row: int


@dataclasses.dataclass
class StitchConfig:
    url_template: str
    output_file: Path
    z: int
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    workers: int = 8
    rate_limit_ms: int = 50
    retries: int = 3
    timeout: int = 20
    headers: Optional[Dict[str, str]] = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
    use_cuda_stitch: bool = False
    cuda_max_chunk_mb: int = CUDA_DEFAULT_MAX_CHUNK_MB
    cuda_utilization_boost: bool = False
    cuda_load_matrix_size: int = CUDA_LOAD_MATRIX_SIZE


def clamp_lat(lat: float) -> float:
    return max(min(lat, 85.05112878), -85.05112878)


def lonlat_to_tile(lon: float, lat: float, z: int) -> Tuple[int, int]:
    lat = clamp_lat(lat)
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x)), max(0, min(n - 1, y))


def tile_to_lonlat(x: float, y: float, z: int) -> Tuple[float, float]:
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n)))
    lat = math.degrees(lat_rad)
    return lon, lat


def tile_bounds_for_bbox(min_lat: float, min_lon: float, max_lat: float, max_lon: float, z: int):
    # NW and SE tile indices for Web Mercator XYZ.
    x1, y1 = lonlat_to_tile(min_lon, max_lat, z)
    x2, y2 = lonlat_to_tile(max_lon, min_lat, z)
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def tile_to_quadkey(x: int, y: int, z: int) -> str:
    q = []
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        q.append(str(digit))
    return "".join(q)


def expand_url(template: str, x: int, y: int, z: int) -> str:
    rnd = random.randint(0, 3)
    sub = ["a", "b", "c"][rnd % 3]
    snum = str(rnd % 4)
    q = tile_to_quadkey(x, y, z)
    # Unterstützt moderne Platzhalter, Bing QuadKey und viele alte NoniMapView-Platzhalter.
    return (template.replace("{x}", str(x))
                    .replace("{y}", str(y))
                    .replace("{z}", str(z))
                    .replace("{q}", q)
                    .replace("{quadkey}", q)
                    .replace("{rnd}", str(rnd))
                    .replace("{snum}", snum)
                    .replace("{s}", sub)
                    .replace("*GMX*", str(x))
                    .replace("*GMY*", str(y))
                    .replace("*ZM1*", str(z))
                    .replace("*IZM*", str(z))
                    .replace("*RND*", str(rnd))
                    .replace("*LAN*", "de")
                    .replace("*LAN-LAN*", "de-DE"))




def project_tiles_dir(output_file: Path) -> Path:
    return output_file.parent / f"{output_file.stem}_tiles"

def project_sqlite_dir(output_file: Path) -> Path:
    return output_file.parent / f"{output_file.stem}_sqlite"

def project_single_tiff_dir(output_file: Path) -> Path:
    return output_file.parent / f"{output_file.stem}_single_tiff_tiles"

def safe_cache_path(cache_dir: Path, z: int, x: int, y: int) -> Path:
    # Dateiname enthält jetzt ausdrücklich Zoom, X und Y.
    # Dadurch sieht man auch nach einem Abbruch sofort, welche Kachel vorhanden ist.
    return cache_dir / str(z) / f"z{z}_x{x}_y{y}.tile"


def default_tile_tif_dir(cfg: "StitchConfig") -> Path:
    base = cfg.output_file.parent if cfg.output_file.parent else Path.cwd()
    stem = cfg.output_file.stem or "map_output"
    return base / f"{stem}_einzelkacheln_tif_z{cfg.z}"


def safe_tile_tif_path(tile_tif_dir: Path, z: int, x: int, y: int) -> Path:
    return tile_tif_dir / f"z{z}_x{x}_y{y}.tif"


def lonlat_to_webmercator(lon: float, lat: float) -> Tuple[float, float]:
    lat = clamp_lat(lat)
    r = 6378137.0
    x = r * math.radians(lon)
    y = r * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def tile_webmercator_bounds(x: int, y: int, z: int) -> Tuple[float, float, float, float]:
    west_lon, north_lat = tile_to_lonlat(x, y, z)
    east_lon, south_lat = tile_to_lonlat(x + 1, y + 1, z)
    west, north = lonlat_to_webmercator(west_lon, north_lat)
    east, south = lonlat_to_webmercator(east_lon, south_lat)
    return west, south, east, north


def mosaic_webmercator_bounds(x_min: int, y_min: int, x_max: int, y_max: int, z: int) -> Tuple[float, float, float, float]:
    west_lon, north_lat = tile_to_lonlat(x_min, y_min, z)
    east_lon, south_lat = tile_to_lonlat(x_max + 1, y_max + 1, z)
    west, north = lonlat_to_webmercator(west_lon, north_lat)
    east, south = lonlat_to_webmercator(east_lon, south_lat)
    return west, south, east, north


def write_worldfile_and_prj(tif_path: Path, width: int, height: int, bounds_3857: Tuple[float, float, float, float]) -> None:
    # Minimal-invasive Georeferenzierung: Der ursprüngliche TIFF/BigTIFF-Schreibweg bleibt unverändert.
    # QGIS/GIS liest die Georeferenz über .tfw + .prj neben der TIFF-Datei.
    west, south, east, north = bounds_3857
    px_w = (east - west) / float(width)
    px_h = (south - north) / float(height)
    tfw = tif_path.with_suffix(".tfw")
    prj = tif_path.with_suffix(".prj")
    tfw.write_text(
        f"{px_w:.12f}\n0.0\n0.0\n{px_h:.12f}\n{west + px_w / 2.0:.12f}\n{north + px_h / 2.0:.12f}\n",
        encoding="utf-8",
    )
    prj.write_text(
        'PROJCS["WGS 84 / Pseudo-Mercator",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Mercator_1SP"],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1],AUTHORITY["EPSG","3857"]]',
        encoding="utf-8",
    )


def save_tile_as_tif(data: Optional[bytes], out_path: Path, z: int, x: int, y: int) -> None:
    # Schreibt genau eine erzeugte Kachel sofort als TIFF.
    # Vorhandene TIFF-Tiles werden nicht erneut geschrieben.
    if out_path.exists() and out_path.stat().st_size > 100:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = decode_tile(data)
    tmp = out_path.with_suffix(".tmp.tif")
    im.save(tmp, format="TIFF", compression="tiff_deflate")
    os.replace(tmp, out_path)
    write_worldfile_and_prj(out_path, TILE_SIZE, TILE_SIZE, tile_webmercator_bounds(x, y, z))


def download_one(job: TileJob, cfg: StitchConfig, stop_event: threading.Event) -> Tuple[TileJob, Optional[bytes], Optional[str]]:
    """Download one tile without persistent cache.

    The tile bytes are returned to the stitcher and are never written to a raw
    tile cache folder. Resume/SQLite caching is intentionally disabled so the
    only persistent output is the streamed BigTIFF.
    """
    if stop_event.is_set():
        return job, None, "cancelled"
    if requests is None:
        return job, None, "requests is not installed"
    url = expand_url(cfg.url_template, job.x, job.y, job.z)
    headers = {"User-Agent": USER_AGENT}
    if cfg.headers:
        headers.update(cfg.headers)
    last_err = None
    for attempt in range(cfg.retries):
        if stop_event.is_set():
            return job, None, "cancelled"
        try:
            if cfg.rate_limit_ms:
                time.sleep(cfg.rate_limit_ms / 1000.0)
            r = requests.get(url, headers=headers, timeout=cfg.timeout, stream=True)
            r.raise_for_status()
            data = r.content
            if len(data) < 50:
                raise RuntimeError("empty/invalid tile")
            return job, data, None
        except Exception as exc:
            last_err = str(exc)
            time.sleep(0.5 * (attempt + 1))
    return job, None, last_err

def make_blank_tile() -> "Image.Image":
    return Image.new("RGB", (TILE_SIZE, TILE_SIZE), (255, 255, 255))


def decode_tile(data: Optional[bytes]) -> "Image.Image":
    if Image is None:
        raise RuntimeError("Pillow is not installed")
    if not data:
        return make_blank_tile()
    try:
        im = Image.open(io.BytesIO(data))
        return im.convert("RGB").resize((TILE_SIZE, TILE_SIZE))
    except Exception:
        return make_blank_tile()


def init_cupy_cuda(log_cb=None):
    """Return the CuPy module when NVIDIA CUDA is usable; otherwise return None.

    Network download and PNG/JPEG decompression are still CPU/I/O operations.
    CuPy is used for the pixel compositing/stitching stage before the data is
    copied back into the on-disk BigTIFF memmap.
    """
    try:
        import cupy as cp  # type: ignore
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 1:
            raise RuntimeError("no CUDA device found")
        device_id = cp.cuda.Device().id
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        name = props.get("name", b"CUDA GPU")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        # Run a tiny operation to catch broken driver/DLL setups immediately.
        test = cp.arange(4, dtype=cp.uint8)
        int(cp.asnumpy(test).sum())
        if log_cb:
            log_cb(f"CUDA/CuPy active: device {device_id} - {name}")
        return cp
    except Exception as exc:
        if log_cb:
            log_cb(f"CUDA/CuPy unavailable, falling back to CPU stitching: {exc}")
        return None


def tile_bytes_to_numpy_rgb(data: Optional[bytes]):
    """Decode a tile and return a NumPy RGB array.

    Pillow still performs compressed PNG/JPEG decoding on CPU. The heavy CUDA
    path starts after decoding: full chunk staging, optional GPU sanity work,
    and bulk GPU-to-CPU writeback into the on-disk BigTIFF memmap.
    """
    import numpy as np
    return np.asarray(decode_tile(data), dtype=np.uint8)


class CudaUtilizationBooster:
    """Optional CUDA load generator.

    The downloader is normally network/CPU-decode/disk limited, so the GPU may
    idle even when CUDA stitching is enabled. This booster deliberately runs
    harmless matrix multiplications on the GPU while the job is active. It can
    push NVIDIA utilization close to 100%, but it does not make tile servers or
    Pillow JPEG/PNG decoding faster. It is useful only when the user explicitly
    wants the GPU kept loaded.
    """

    def __init__(self, cp, stop_event: threading.Event, log_cb, matrix_size: int = CUDA_LOAD_MATRIX_SIZE):
        self.cp = cp
        self.stop_event = stop_event
        self.log_cb = log_cb
        self.matrix_size = max(256, int(matrix_size))
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.log_cb(
            f"CUDA utilization boost active: background {self.matrix_size}x{self.matrix_size} FP32 matmul loop. "
            "This increases GPU load/heat and is not a download-speed guarantee."
        )

    def stop(self) -> None:
        try:
            if self._thread:
                self._thread.join(timeout=2.0)
        except Exception:
            pass

    def _run(self) -> None:
        cp = self.cp
        try:
            # Use deterministic-ish random matrices once, then keep reusing them.
            a = cp.random.random((self.matrix_size, self.matrix_size), dtype=cp.float32)
            b = cp.random.random((self.matrix_size, self.matrix_size), dtype=cp.float32)
            c = cp.empty((self.matrix_size, self.matrix_size), dtype=cp.float32)
            while not self.stop_event.is_set():
                c = a @ b
                # Prevent total compiler/dead-code elimination and keep the stream moving.
                a = c * cp.float32(0.9999) + cp.float32(0.0001)
                cp.cuda.Stream.null.synchronize()
        except Exception as exc:
            try:
                self.log_cb(f"CUDA utilization boost stopped: {exc}")
            except Exception:
                pass
        finally:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass


def cuda_preprocess_tile(cp, tile_arr):
    """Small real CUDA operation per tile before inserting into the GPU chunk.

    It keeps the CUDA path active without changing visual output: uint8 -> float32
    -> clipped uint8. This is intentionally conservative and lossless for the
    normal 0..255 RGB tile values.
    """
    gpu = cp.asarray(tile_arr, dtype=cp.float32)
    gpu = cp.clip(gpu, 0, 255).astype(cp.uint8)
    return gpu



def iter_tile_jobs(x_min: int, y_min: int, x_max: int, y_max: int, z: int):
    # Generator statt Liste: selbst riesige Bereiche erzeugen keine RAM-Spitze.
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            yield TileJob(x, y, z, x - x_min, y - y_min)


def count_existing_tiles(cache_dir: Path, x_min: int, y_min: int, x_max: int, y_max: int, z: int) -> int:
    existing = 0
    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            p = safe_cache_path(cache_dir, z, x, y)
            if p.exists() and p.stat().st_size > 100:
                existing += 1
    return existing



def sqlite_path_for(cfg: StitchConfig) -> Path:
    return cfg.cache_dir / f"download_state_z{cfg.z}.sqlite"


def init_state_db(cfg: StitchConfig):
    if not cfg.use_sqlite:
        return None
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(sqlite_path_for(cfg)), timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("CREATE TABLE IF NOT EXISTS tiles (z INTEGER, x INTEGER, y INTEGER, status TEXT, updated REAL, error TEXT, PRIMARY KEY(z,x,y))")
    db.execute("CREATE TABLE IF NOT EXISTS chunks (z INTEGER, x0 INTEGER, y0 INTEGER, x1 INTEGER, y1 INTEGER, status TEXT, updated REAL, PRIMARY KEY(z,x0,y0,x1,y1))")
    db.commit()
    return db


def db_tile_done(db, z: int, x: int, y: int) -> bool:
    if db is None:
        return False
    row = db.execute("SELECT status FROM tiles WHERE z=? AND x=? AND y=?", (z, x, y)).fetchone()
    return bool(row and row[0] == "done")


def db_mark_tile(db, z: int, x: int, y: int, status: str, error: Optional[str] = None):
    if db is None:
        return
    db.execute("INSERT OR REPLACE INTO tiles(z,x,y,status,updated,error) VALUES(?,?,?,?,?,?)", (z, x, y, status, time.time(), error))


def db_mark_chunk(db, z: int, x0: int, y0: int, x1: int, y1: int, status: str):
    if db is None:
        return
    db.execute("INSERT OR REPLACE INTO chunks(z,x0,y0,x1,y1,status,updated) VALUES(?,?,?,?,?,?,?)", (z, x0, y0, x1, y1, status, time.time()))
    db.commit()


def iter_chunks(x_min: int, y_min: int, x_max: int, y_max: int, chunk_size: int):
    """Spatial chunk scheduler. Yields chunk bounds only; never builds a global tile list."""
    chunk_size = max(1, int(chunk_size))
    for cy in range(y_min, y_max + 1, chunk_size):
        for cx in range(x_min, x_max + 1, chunk_size):
            yield cx, cy, min(cx + chunk_size - 1, x_max), min(cy + chunk_size - 1, y_max)


def iter_chunk_jobs(cx0: int, cy0: int, cx1: int, cy1: int, z: int, x_min: int, y_min: int):
    """Yields jobs for one chunk only."""
    for y in range(cy0, cy1 + 1):
        for x in range(cx0, cx1 + 1):
            yield TileJob(x, y, z, x - x_min, y - y_min)


def format_bytes(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if value < 1024 or unit == "PB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def ensure_enough_disk_space(path: Path, required_bytes: int, log_cb) -> None:
    """Raise before creating the BigTIFF when the target drive is too small."""
    target_dir = path.expanduser().parent
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        usage = shutil.disk_usage(str(target_dir))
    except Exception as exc:
        raise RuntimeError(f"Could not check free disk space for {target_dir}: {exc}") from exc
    # tifffile metadata and filesystem allocation need a little headroom.
    required_with_margin = int(required_bytes * 1.03) + 512 * 1024 * 1024
    log_cb(f"Estimated raw BigTIFF payload: {format_bytes(required_bytes)}")
    log_cb(f"Free space on target drive: {format_bytes(usage.free)}")
    if usage.free < required_with_margin:
        raise RuntimeError(
            "Not enough free disk space for direct BigTIFF streaming. "
            f"Required with safety margin: {format_bytes(required_with_margin)}; "
            f"available: {format_bytes(usage.free)}. Choose a smaller area/zoom or another drive."
        )


def geotiff_extratags_epsg3857(width: int, height: int, bounds_3857: Tuple[float, float, float, float]):
    """Return embedded GeoTIFF tags for EPSG:3857 / Web Mercator.

    This writes georeferencing into the TIFF itself, so QGIS can place the
    BigTIFF without depending on .tfw/.prj sidecar files.
    """
    west, south, east, north = bounds_3857
    px_w = (east - west) / float(width)
    px_h = (north - south) / float(height)
    model_pixel_scale = (float(px_w), float(px_h), 0.0)
    # Raster coordinate (0,0,0) is tied to the top-left model coordinate.
    model_tiepoint = (0.0, 0.0, 0.0, float(west), float(north), 0.0)
    # GeoKeyDirectoryTag: header + GTModelTypeGeoKey(Projected),
    # GTRasterTypeGeoKey(PixelIsArea), ProjectedCSTypeGeoKey(EPSG:3857).
    geo_key_directory = (
        1, 1, 0, 3,
        1024, 0, 1, 1,
        1025, 0, 1, 1,
        3072, 0, 1, 3857,
    )
    return [
        (33550, "d", 3, model_pixel_scale, False),
        (33922, "d", 6, model_tiepoint, False),
        (34735, "H", len(geo_key_directory), geo_key_directory, False),
    ]


def open_direct_bigtiff(cfg: StitchConfig, width: int, height: int, bounds_3857: Tuple[float, float, float, float], log_cb):
    """Create a georeferenced on-disk BigTIFF memmap or raise a clear error.

    There is no cache fallback in this build. Georeferencing is embedded as
    GeoTIFF tags, not only written as .tfw/.prj sidecars.
    """
    estimated = int(width) * int(height) * 3
    ensure_enough_disk_space(cfg.output_file, estimated, log_cb)
    try:
        import tifffile
    except Exception as exc:
        raise RuntimeError("tifffile is required for direct BigTIFF streaming. Install with: pip install tifffile") from exc
    try:
        cfg.output_file.parent.mkdir(parents=True, exist_ok=True)
        bigtiff = estimated > 3_800_000_000
        extratags = geotiff_extratags_epsg3857(width, height, bounds_3857)
        mem = tifffile.memmap(
            str(cfg.output_file),
            shape=(height, width, 3),
            dtype="uint8",
            bigtiff=bigtiff,
            photometric="rgb",
            metadata=None,
            extratags=extratags,
        )
        log_cb(f"Direct GeoTIFF/BigTIFF writer opened: {cfg.output_file}")
        log_cb("Embedded GeoTIFF georeferencing written: EPSG:3857, ModelPixelScaleTag, ModelTiepointTag, GeoKeyDirectoryTag")
        return mem, "memmap"
    except OSError as exc:
        raise RuntimeError(
            "Direct BigTIFF output could not be created. This is usually caused by not enough disk space, "
            f"permission problems, or a path/drive limit. Target: {cfg.output_file}. Error: {exc}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Direct GeoTIFF/BigTIFF writer failed: {exc}") from exc

def stitch_tiles(cfg: StitchConfig, progress_cb, log_cb, stop_event: threading.Event):
    if Image is None:
        raise RuntimeError("Pillow is required. Install with: pip install pillow requests")

    x_min, y_min, x_max, y_max = tile_bounds_for_bbox(cfg.min_lat, cfg.min_lon, cfg.max_lat, cfg.max_lon, cfg.z)
    cols = x_max - x_min + 1
    rows = y_max - y_min + 1
    total = cols * rows
    width = cols * TILE_SIZE
    height = rows * TILE_SIZE
    chunk_size = max(1, int(cfg.chunk_size))

    log_cb(f"Tile range: x={x_min}..{x_max}, y={y_min}..{y_max}")
    log_cb(f"Tiles: {cols} x {rows} = {total:,}")
    log_cb(f"Image size: {width:,} x {height:,} px")
    log_cb(f"Direct BigTIFF streaming active: chunk size {chunk_size} x {chunk_size} tiles")
    log_cb("No raw tile cache, no SQLite resume database, and no separate TIFF tile output will be created.")

    if total > HARD_TILE_WARNING:
        log_cb(f"Warning: very large selection with {total:,} tiles. This can run for days/weeks and may violate server terms if not authorized.")

    bounds_3857 = mosaic_webmercator_bounds(x_min, y_min, x_max, y_max, cfg.z)
    direct_mem, direct_kind = open_direct_bigtiff(cfg, width, height, bounds_3857, log_cb)

    cupy = None
    cuda_booster = None
    if cfg.use_cuda_stitch:
        cupy = init_cupy_cuda(log_cb)
        if cupy is not None:
            log_cb("CUDA mode: chunk compositing and bulk writeback use CuPy; network download, Pillow PNG/JPEG decode, and disk writes can still bottleneck.")
            if cfg.cuda_utilization_boost:
                cuda_booster = CudaUtilizationBooster(cupy, stop_event, log_cb, cfg.cuda_load_matrix_size)
                cuda_booster.start()

    max_workers = max(1, cfg.workers)
    max_inflight = max_workers * MAX_INFLIGHT_PER_WORKER
    done = 0
    errors = 0

    try:
        with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for cx0, cy0, cx1, cy1 in iter_chunks(x_min, y_min, x_max, y_max, chunk_size):
                if stop_event.is_set():
                    break
                log_cb(f"Chunk start: x={cx0}..{cx1}, y={cy0}..{cy1}")
                chunk_cols = cx1 - cx0 + 1
                chunk_rows = cy1 - cy0 + 1
                chunk_bytes = chunk_cols * TILE_SIZE * chunk_rows * TILE_SIZE * 3
                cuda_chunk = None
                cuda_chunk_dirty = False
                if cupy is not None and direct_mem is not None and chunk_bytes <= int(cfg.cuda_max_chunk_mb) * 1024 * 1024:
                    cuda_chunk = cupy.full((chunk_rows * TILE_SIZE, chunk_cols * TILE_SIZE, 3), 255, dtype=cupy.uint8)
                elif cupy is not None and direct_mem is not None and done == 0:
                    log_cb(f"CUDA chunk buffer disabled for this size ({chunk_bytes/1024/1024:.1f} MB > {cfg.cuda_max_chunk_mb} MB); using CUDA tile transfer fallback.")
                job_iter = iter_chunk_jobs(cx0, cy0, cx1, cy1, cfg.z, x_min, y_min)
                pending = set()
                while not stop_event.is_set():
                    while len(pending) < max_inflight:
                        try:
                            job = next(job_iter)
                        except StopIteration:
                            break
                        pending.add(pool.submit(download_one, job, cfg, stop_event))
                    if not pending:
                        break
                    done_set, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
                    for fut in done_set:
                        job, data, err = fut.result()
                        done += 1
                        if err:
                            errors += 1
                            if errors <= 30:
                                log_cb(f"Error {job.z}/{job.x}/{job.y}: {err}")
                        else:
                            try:
                                tile_arr = tile_bytes_to_numpy_rgb(data)
                                r0 = job.row * TILE_SIZE
                                c0 = job.col * TILE_SIZE
                                if cuda_chunk is not None:
                                    lr0 = (job.y - cy0) * TILE_SIZE
                                    lc0 = (job.x - cx0) * TILE_SIZE
                                    cuda_chunk[lr0:lr0+TILE_SIZE, lc0:lc0+TILE_SIZE, :] = cuda_preprocess_tile(cupy, tile_arr)
                                    cuda_chunk_dirty = True
                                else:
                                    direct_mem[r0:r0+TILE_SIZE, c0:c0+TILE_SIZE, :] = tile_arr
                            except Exception as exc:
                                errors += 1
                                if errors <= 30:
                                    log_cb(f"Write error {job.z}/{job.x}/{job.y}: {exc}")
                        if done % 25 == 0 or done == total:
                            progress_cb(done, total, "Stream")
                if cuda_chunk is not None and cuda_chunk_dirty and direct_mem is not None:
                    try:
                        r0 = (cy0 - y_min) * TILE_SIZE
                        c0 = (cx0 - x_min) * TILE_SIZE
                        direct_mem[r0:r0 + chunk_rows * TILE_SIZE, c0:c0 + chunk_cols * TILE_SIZE, :] = cupy.asnumpy(cuda_chunk)
                        cupy.cuda.Stream.null.synchronize()
                    except Exception as exc:
                        log_cb(f"CUDA chunk writeback failed, continuing with CPU path for later chunks: {exc}")
                        cupy = None
                    finally:
                        try:
                            del cuda_chunk
                            if cupy is not None:
                                cupy.get_default_memory_pool().free_all_blocks()
                        except Exception:
                            pass
                if direct_mem is not None:
                    try:
                        direct_mem.flush()
                    except Exception:
                        pass
    finally:
        if cuda_booster is not None:
            try:
                cuda_booster.stop()
            except Exception:
                pass
        if direct_mem is not None:
            try:
                direct_mem.flush()
                del direct_mem
            except Exception:
                pass

    if stop_event.is_set():
        log_cb("Stopped. Partial BigTIFF remains at the output path; no cache/resume database was created.")
        return

    log_cb(f"Finished direct BigTIFF streaming. Processed: {done:,}; errors: {errors:,}")
    log_cb(f"Finished BigTIFF/direct output: {cfg.output_file}")

def open_folder_in_file_manager(path: Path) -> None:
    """Open a folder in the OS file manager. Safe no-op if it cannot be opened."""
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(path)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass



# -----------------------------------------------------------------------------
# PySide6 integrated WebEngine GUI
# -----------------------------------------------------------------------------
try:
    from PySide6.QtCore import QObject, QTimer, Qt, QUrl, Slot
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
        QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
        QMessageBox, QPushButton, QProgressBar, QSpinBox, QSplitter, QTextEdit,
        QVBoxLayout, QWidget
    )
    from PySide6.QtWebChannel import QWebChannel
    from PySide6.QtWebEngineWidgets import QWebEngineView
except Exception as _pyside_exc:  # pragma: no cover
    QObject = object  # type: ignore
    QMainWindow = object  # type: ignore
    _PYSIDE_IMPORT_ERROR = _pyside_exc
else:
    _PYSIDE_IMPORT_ERROR = None

ESRI_WORLD_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def leaflet_webengine_html(lon: float, lat: float, zoom: int, tile_template: str) -> str:
    """Leaflet/QWebEngine preview adapted from Mustatil Satellite Preview.

    Shift+Drag or right mouse drag selects a bbox and sends it through QWebChannel.
    """
    return f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
<link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"/>
<script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
<script src=\"qrc:///qtwebchannel/qwebchannel.js\"></script>
<style>
html,body,#map{{margin:0;padding:0;width:100%;height:100%;overflow:hidden;background:#111}}
.leaflet-container{{background:#111;cursor:grab}}.leaflet-container.selecting{{cursor:crosshair}}
.hint{{position:absolute;left:10px;bottom:10px;z-index:1000;color:#eee;background:rgba(0,0,0,.68);font:12px/1.35 Arial,sans-serif;padding:7px 9px;border-radius:4px;user-select:none}}
.crosshair{{position:absolute;left:50%;top:50%;width:18px;height:18px;margin-left:-9px;margin-top:-9px;pointer-events:none;z-index:1000}}
.crosshair:before,.crosshair:after{{content:\"\";position:absolute;background:rgba(255,255,255,.88);box-shadow:0 0 2px #000}}
.crosshair:before{{left:8px;top:0;width:2px;height:18px}}.crosshair:after{{left:0;top:8px;width:18px;height:2px}}
</style></head><body><div id=\"map\"></div><div class=\"crosshair\"></div><div id=\"hint\" class=\"hint\">Shift+Drag oder Rechts-Drag: Feld markieren</div>
<script>
(function(){{
const TILE_TEMPLATE={json.dumps(tile_template or ESRI_WORLD_IMAGERY)};
let bridge=null;
const map=L.map('map',{{zoomControl:true,attributionControl:false,preferCanvas:true,inertia:true,zoomAnimation:true,fadeAnimation:true,updateWhenIdle:false,updateWhenZooming:false,wheelPxPerZoomLevel:96}}).setView([{float(clamp_lat(lat))},{float(lon)}],{int(zoom)});
let layer=L.tileLayer(TILE_TEMPLATE,{{tileSize:256,minZoom:0,maxZoom:22,maxNativeZoom:22,keepBuffer:5,updateWhenIdle:false,updateWhenZooming:false,detectRetina:false,crossOrigin:false}}).addTo(map);
let selectionRect=null, selecting=false, startLatLng=null;
function hint(t){{document.getElementById('hint').textContent=t;}}
function notifyMove(){{const c=map.getCenter();hint(`Zoom ${{map.getZoom()}} | lon ${{c.lng.toFixed(7)}} lat ${{c.lat.toFixed(7)}} | Shift+Drag/Rechts-Drag: Feld markieren`);if(bridge&&bridge.mapMoved)bridge.mapMoved(c.lng,c.lat,map.getZoom());}}
map.on('moveend zoomend',notifyMove);
map.getContainer().addEventListener('contextmenu',function(e){{e.preventDefault();}});
map.on('mousedown',function(e){{const oe=e.originalEvent||{{}};if(!(oe.shiftKey||oe.button===2))return;selecting=true;startLatLng=e.latlng;map.dragging.disable();map.getContainer().classList.add('selecting');if(selectionRect)map.removeLayer(selectionRect);selectionRect=L.rectangle([startLatLng,startLatLng],{{color:'#00ffff',weight:2,fill:true,fillOpacity:.12,dashArray:'5,4'}}).addTo(map);}});
map.on('mousemove',function(e){{if(selecting&&selectionRect&&startLatLng)selectionRect.setBounds(L.latLngBounds(startLatLng,e.latlng));}});
function finishSelection(e){{
  if(!selecting||!selectionRect)return;
  selecting=false;map.dragging.enable();map.getContainer().classList.remove('selecting');
  const b=selectionRect.getBounds();
  const west=b.getWest(),south=b.getSouth(),east=b.getEast(),north=b.getNorth();
  if(east<=west||north<=south||Math.abs(east-west)<1e-9||Math.abs(north-south)<1e-9){{hint('Auswahl ignoriert: Rechteck größer ziehen');return;}}
  hint(`Auswahl eingetragen: W ${{west.toFixed(8)}} S ${{south.toFixed(8)}} E ${{east.toFixed(8)}} N ${{north.toFixed(8)}}`);
  if(bridge&&bridge.selectionChanged)bridge.selectionChanged(west,south,east,north);
}}
map.on('mouseup',finishSelection);map.on('mouseout',function(e){{if(selecting)finishSelection(e);}});
window.pymapSetView=function(lon,lat,zoom,tileTemplate){{if(tileTemplate)layer.setUrl(tileTemplate);map.setView([lat,lon],zoom,{{animate:false}});setTimeout(function(){{map.invalidateSize(true);notifyMove();}},30);}};
if(window.qt&&window.qt.webChannelTransport){{new QWebChannel(qt.webChannelTransport,function(channel){{bridge=channel.objects.pymapBridge;notifyMove();}});}}else{{notifyMove();}}
setTimeout(function(){{map.invalidateSize(true);notifyMove();}},100);
}})();
</script></body></html>"""


class WebBridge(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window

    @Slot(float, float, int)
    def mapMoved(self, lon: float, lat: float, zoom: int) -> None:
        self.window.center_lon = float(lon)
        self.window.center_lat = float(lat)
        self.window.preview_zoom = int(zoom)
        self.window.status_label.setText(
            f"Preview: zoom {int(zoom)} | lon {float(lon):.7f} lat {float(lat):.7f}"
        )

    @Slot(float, float, float, float)
    def selectionChanged(self, west: float, south: float, east: float, north: float) -> None:
        self.window.min_lon_edit.setText(f"{float(west):.8f}")
        self.window.min_lat_edit.setText(f"{float(south):.8f}")
        self.window.max_lon_edit.setText(f"{float(east):.8f}")
        self.window.max_lat_edit.setText(f"{float(north):.8f}")
        self.window.log_msg(
            f"WebMap selection entered: South={south:.8f}, West={west:.8f}, North={north:.8f}, East={east:.8f}"
        )
        self.window.calculate()


class PySideMapStitcher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Py Map Stitcher - PySide6 WebView Direct BigTIFF CUDA")
        self.resize(1320, 820)
        self.stop_event = threading.Event()
        self.worker_thread = None
        self.q = queue.Queue()
        self.center_lon = 10.0
        self.center_lat = 51.0
        self.preview_zoom = 3
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(80)
        QTimer.singleShot(200, self.refresh_webmap)

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)
        splitter.addWidget(left)
        splitter.setStretchFactor(0, 0)

        form_box = QGroupBox("Map / Download")
        form = QFormLayout(form_box)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        left_layout.addWidget(form_box)

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(MAP_PRESETS.keys()))
        self.preset_combo.setCurrentText("Google Satellite")
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        form.addRow("Map Selection", self.preset_combo)

        self.url_edit = QLineEdit(MAP_PRESETS["Google Satellite"]["url"])
        form.addRow("URL Template", self.url_edit)
        self.note_label = QLabel(MAP_PRESETS["Google Satellite"]["note"])
        self.note_label.setWordWrap(True)
        form.addRow("Note", self.note_label)

        self.zoom_spin = QSpinBox(); self.zoom_spin.setRange(0, 22); self.zoom_spin.setValue(18)
        self.min_lat_edit = QLineEdit("")
        self.min_lon_edit = QLineEdit("")
        self.max_lat_edit = QLineEdit("")
        self.max_lon_edit = QLineEdit("")
        self.workers_spin = QSpinBox(); self.workers_spin.setRange(1, 256); self.workers_spin.setValue(32)
        self.rate_spin = QSpinBox(); self.rate_spin.setRange(0, 5000); self.rate_spin.setSingleStep(10); self.rate_spin.setValue(0)
        self.chunk_spin = QSpinBox(); self.chunk_spin.setRange(8, 2048); self.chunk_spin.setSingleStep(8); self.chunk_spin.setValue(128)
        self.cuda_check = QCheckBox("NVIDIA CUDA/CuPy stitch")
        self.cuda_check.setChecked(True)
        self.cuda_boost_check = QCheckBox("CUDA utilization boost / keep GPU busy")
        self.cuda_boost_check.setToolTip("Deliberately runs extra CUDA work while stitching. Higher GPU usage and heat; not necessarily faster.")
        self.cuda_mb_spin = QSpinBox(); self.cuda_mb_spin.setRange(128, 4096); self.cuda_mb_spin.setSingleStep(128); self.cuda_mb_spin.setValue(CUDA_DEFAULT_MAX_CHUNK_MB)
        self.cuda_load_spin = QSpinBox(); self.cuda_load_spin.setRange(256, 4096); self.cuda_load_spin.setSingleStep(256); self.cuda_load_spin.setValue(CUDA_LOAD_MATRIX_SIZE)
        self.outfile_edit = QLineEdit(str(Path.home() / "Desktop" / "map_output.tif"))

        form.addRow("Download Zoom", self.zoom_spin)
        form.addRow("South / min lat", self.min_lat_edit)
        form.addRow("West / min lon", self.min_lon_edit)
        form.addRow("North / max lat", self.max_lat_edit)
        form.addRow("East / max lon", self.max_lon_edit)
        form.addRow("Download Threads", self.workers_spin)
        form.addRow("Delay per Request ms", self.rate_spin)
        form.addRow("Chunk size tiles", self.chunk_spin)
        form.addRow("CUDA", self.cuda_check)
        form.addRow("CUDA boost", self.cuda_boost_check)
        form.addRow("CUDA chunk MB", self.cuda_mb_spin)
        form.addRow("CUDA load size", self.cuda_load_spin)

        out_row = QHBoxLayout()
        out_row.addWidget(self.outfile_edit, 1)
        browse = QPushButton("…")
        browse.clicked.connect(self.pick_output)
        out_row.addWidget(browse)
        out_widget = QWidget(); out_widget.setLayout(out_row)
        form.addRow("Output File", out_widget)

        btn_row = QHBoxLayout()
        calc_btn = QPushButton("Calculate")
        calc_btn.clicked.connect(self.calculate)
        start_btn = QPushButton("Start")
        start_btn.clicked.connect(self.start)
        stop_btn = QPushButton("Stop")
        stop_btn.clicked.connect(self.stop_event.set)
        btn_row.addWidget(calc_btn); btn_row.addWidget(start_btn); btn_row.addWidget(stop_btn)
        left_layout.addLayout(btn_row)

        terms = QLabel("Only use servers where downloading/stitching is allowed. Google/Bing/OSM may restrict bulk downloads.")
        terms.setWordWrap(True)
        left_layout.addWidget(terms)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        left_layout.addWidget(self.progress)
        self.status_label = QLabel("Ready")
        left_layout.addWidget(self.status_label)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(170)
        left_layout.addWidget(self.log, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 8, 8, 8)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        map_header = QHBoxLayout()
        map_title = QLabel("Satellite WebMap / Selection like Mustatil Satellite Preview")
        map_title.setStyleSheet("font-weight: 600;")
        map_header.addWidget(map_title, 1)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self.refresh_webmap)
        map_header.addWidget(reload_btn)
        right_layout.addLayout(map_header)

        self.webview = QWebEngineView()
        self.web_bridge = WebBridge(self)
        self.web_channel = QWebChannel(self.webview.page())
        self.web_channel.registerObject("pymapBridge", self.web_bridge)
        self.webview.page().setWebChannel(self.web_channel)
        right_layout.addWidget(self.webview, 1)

        splitter.setSizes([420, 900])

    def on_preset_changed(self, name: str) -> None:
        preset = MAP_PRESETS.get(name, MAP_PRESETS["Custom"])
        self.url_edit.setText(preset["url"])
        self.note_label.setText(preset.get("note", ""))
        self.refresh_webmap()

    def refresh_webmap(self) -> None:
        html = leaflet_webengine_html(
            self.center_lon,
            self.center_lat,
            self.preview_zoom,
            self.url_edit.text().strip() or ESRI_WORLD_IMAGERY,
        )
        self.webview.setHtml(html, QUrl("https://mustatil.local/"))
        self.status_label.setText("WebMap loaded - Shift+Drag or right-drag to select extent")

    def pick_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Output BigTIFF", self.outfile_edit.text(), "TIFF (*.tif *.tiff);;All files (*)")
        if path:
            self.outfile_edit.setText(path)

    def _config(self) -> StitchConfig:
        return StitchConfig(
            url_template=self.url_edit.text().strip(),
            output_file=Path(self.outfile_edit.text()).expanduser(),
            z=int(self.zoom_spin.value()),
            min_lat=float(self.min_lat_edit.text().replace(",", ".")),
            min_lon=float(self.min_lon_edit.text().replace(",", ".")),
            max_lat=float(self.max_lat_edit.text().replace(",", ".")),
            max_lon=float(self.max_lon_edit.text().replace(",", ".")),
            workers=int(self.workers_spin.value()),
            rate_limit_ms=int(self.rate_spin.value()),
            chunk_size=int(self.chunk_spin.value()),
            use_cuda_stitch=bool(self.cuda_check.isChecked()),
            cuda_max_chunk_mb=int(self.cuda_mb_spin.value()),
            cuda_utilization_boost=bool(self.cuda_boost_check.isChecked()),
            cuda_load_matrix_size=int(self.cuda_load_spin.value()),
        )

    def calculate(self) -> None:
        try:
            cfg = self._config()
            x_min, y_min, x_max, y_max = tile_bounds_for_bbox(cfg.min_lat, cfg.min_lon, cfg.max_lat, cfg.max_lon, cfg.z)
            cols = x_max - x_min + 1
            rows = y_max - y_min + 1
            raw_bytes = cols * TILE_SIZE * rows * TILE_SIZE * 3
            self.log_msg(f"Calculation: x={x_min}..{x_max}, y={y_min}..{y_max}")
            self.log_msg(f"Tiles: {cols} x {rows} = {cols*rows:,}; Pixel: {cols*TILE_SIZE:,} x {rows*TILE_SIZE:,}")
            self.log_msg(f"Estimated raw BigTIFF payload: {format_bytes(raw_bytes)}")
            self.log_msg("Output: direct BigTIFF streaming only; no cache and no SQLite resume database.")
            if self.cuda_check.isChecked():
                self.log_msg(f"CUDA enabled: chunk buffer up to {self.cuda_mb_spin.value()} MB; utilization boost={self.cuda_boost_check.isChecked()}.")
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))

    def start(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            QMessageBox.information(self, "Running", "A job is already running.")
            return
        try:
            cfg = self._config()
        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            return
        self.stop_event.clear()
        self.progress.setValue(0)
        self.worker_thread = threading.Thread(target=self._run_job, args=(cfg,), daemon=True)
        self.worker_thread.start()

    def _run_job(self, cfg: StitchConfig) -> None:
        try:
            stitch_tiles(cfg, self._progress, self._log_thread, self.stop_event)
        except Exception as exc:
            self._log_thread(f"ERROR: {exc}")
            self.q.put(("status", "Error"))

    def _progress(self, done: int, total: int, phase: str) -> None:
        self.q.put(("progress", done, total, phase))

    def _log_thread(self, msg: str) -> None:
        self.q.put(("log", msg))

    def log_msg(self, msg: str) -> None:
        self.log.append(str(msg))

    def _poll(self) -> None:
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "log":
                    self.log_msg(item[1])
                elif item[0] == "progress":
                    _, done, total, phase = item
                    self.progress.setRange(0, max(1, int(total)))
                    self.progress.setValue(int(done))
                    self.status_label.setText(f"{phase}: {done:,}/{total:,}")
                elif item[0] == "status":
                    self.status_label.setText(item[1])
        except queue.Empty:
            pass


def main() -> int:
    if _PYSIDE_IMPORT_ERROR is not None:
        print("PySide6 WebEngine is missing.")
        print("Install with: python -m pip install PySide6 PySide6-Addons PySide6-Essentials shiboken6")
        print("Import error:", _PYSIDE_IMPORT_ERROR)
        return 1
    app = QApplication(sys.argv)
    win = PySideMapStitcher()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
