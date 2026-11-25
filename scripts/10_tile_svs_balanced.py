from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import random
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import openslide
import pandas as pd
from PIL import Image


BALANCED_DIR = Path("processed") / "balanced_imaging_2y"
SLIDE_TABLE = BALANCED_DIR / "balanced_diagnostic_slide_files.csv"
SVS_DIR = BALANCED_DIR / "svs"
DEFAULT_TILE_DIR = BALANCED_DIR / "tiles"
DEFAULT_SUMMARY = BALANCED_DIR / "tile_summary.csv"


def find_slide_path(file_id: str, file_name: str) -> Path | None:
    expected = SVS_DIR / file_id / file_name
    if expected.exists():
        return expected
    matches = list(SVS_DIR.rglob(file_name))
    return matches[0] if matches else None


def best_level_for_downsample(slide: openslide.OpenSlide, target_downsample: float) -> int:
    best_level = 0
    best_gap = float("inf")
    for level, downsample in enumerate(slide.level_downsamples):
        gap = abs(float(downsample) - target_downsample)
        if gap < best_gap:
            best_level = level
            best_gap = gap
    return best_level


def tissue_fraction(tile_rgb: np.ndarray) -> float:
    arr = tile_rgb.astype(np.uint8)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc = arr.max(axis=2).astype(np.int16)
    minc = arr.min(axis=2).astype(np.int16)
    saturation = maxc - minc
    brightness = arr.mean(axis=2)

    not_white = brightness < 235
    colored = saturation > 18
    not_dark = brightness > 30
    tissue = not_white & colored & not_dark
    return float(tissue.mean())


def tile_is_usable(tile: Image.Image, min_tissue_fraction: float, min_std: float) -> bool:
    arr = np.asarray(tile.convert("RGB"))
    return tissue_fraction(arr) >= min_tissue_fraction and float(arr.std()) >= min_std


def candidate_locations(
    slide: openslide.OpenSlide,
    level: int,
    tile_size: int,
    stride: int,
    min_tissue_fraction: float,
    seed: int,
    max_candidates: int,
) -> list[tuple[int, int]]:
    downsample = int(round(float(slide.level_downsamples[level])))
    level_w, level_h = slide.level_dimensions[level]
    locations = []

    for y_level in range(0, max(1, level_h - tile_size + 1), stride):
        for x_level in range(0, max(1, level_w - tile_size + 1), stride):
            x0 = int(x_level * downsample)
            y0 = int(y_level * downsample)
            thumb_tile = slide.read_region((x0, y0), level, (tile_size, tile_size)).convert("RGB")
            if tissue_fraction(np.asarray(thumb_tile)) >= min_tissue_fraction:
                locations.append((x0, y0))

    rng = random.Random(seed)
    rng.shuffle(locations)
    return locations[:max_candidates]


def tile_slide(record: dict[str, str], args: ArgumentParser) -> dict[str, object]:
    patient_id = record["patient_barcode"][:12]
    file_id = record["file_id"]
    file_name = record["file_name"]
    slide_path = find_slide_path(file_id, file_name)
    result = {
        "PatientID": patient_id,
        "file_id": file_id,
        "file_name": file_name,
        "slide_path": str(slide_path) if slide_path else "",
        "tiles_saved": 0,
        "candidate_tiles": 0,
        "tile_level": "",
        "tile_downsample": "",
        "error": "",
    }

    if slide_path is None:
        result["error"] = "slide_not_found"
        return result

    try:
        slide = openslide.OpenSlide(str(slide_path))
        level = best_level_for_downsample(slide, args.target_downsample)
        downsample = int(round(float(slide.level_downsamples[level])))
        result["tile_level"] = level
        result["tile_downsample"] = downsample

        locations = candidate_locations(
            slide=slide,
            level=level,
            tile_size=args.tile_size,
            stride=args.stride,
            min_tissue_fraction=args.min_tissue_fraction,
            seed=args.seed,
            max_candidates=args.max_candidates_per_slide,
        )
        result["candidate_tiles"] = len(locations)

        patient_dir = args.output_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        saved = 0

        for x0, y0 in locations:
            if saved >= args.max_tiles_per_slide:
                break
            tile = slide.read_region((x0, y0), level, (args.tile_size, args.tile_size)).convert("RGB")
            if not tile_is_usable(tile, args.min_tissue_fraction, args.min_std):
                continue
            out_name = f"{file_id}_x{x0}_y{y0}_l{level}_d{downsample}.jpg"
            tile.save(patient_dir / out_name, quality=args.jpeg_quality)
            saved += 1

        slide.close()
        result["tiles_saved"] = saved
    except Exception as exc:
        result["error"] = str(exc)

    return result


def main() -> None:
    parser = ArgumentParser(description="Tile balanced GDC SVS slides into patient-grouped tissue tiles.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_TILE_DIR)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--stride", type=int, default=224)
    parser.add_argument("--target-downsample", type=float, default=16.0)
    parser.add_argument("--max-tiles-per-slide", type=int, default=250)
    parser.add_argument("--max-candidates-per-slide", type=int, default=1500)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.25)
    parser.add_argument("--min-std", type=float, default=8.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-slides", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slides = pd.read_csv(SLIDE_TABLE, dtype=str)
    if args.limit_slides is not None:
        slides = slides.head(args.limit_slides)

    rows = []
    for index, record in enumerate(slides.to_dict("records"), start=1):
        print(f"[{index}/{len(slides)}] Tiling {record['patient_barcode']} {record['file_name']}", flush=True)
        rows.append(tile_slide(record, args))

    summary = pd.DataFrame(rows)
    summary.to_csv(args.summary_path, index=False)
    print()
    print(f"Slides processed: {len(summary)}")
    print(f"Total tiles saved: {int(summary['tiles_saved'].sum())}")
    print(f"Slides with errors: {int(summary['error'].astype(bool).sum())}")
    print(f"Wrote: {args.summary_path}")


if __name__ == "__main__":
    main()
