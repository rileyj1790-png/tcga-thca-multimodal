from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import os

# Keep pretrained weight cache inside the project, not the user profile.
os.environ.setdefault("TORCH_HOME", str((Path("models") / "torch_cache").resolve()))

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import models


BALANCED_DIR = Path("processed") / "balanced_imaging_2y"
DEFAULT_TILE_DIR = BALANCED_DIR / "tiles"
DEFAULT_OUTPUT = BALANCED_DIR / "features" / "image_features_resnet18.csv"
DEFAULT_PARQUET = Path("data") / "features" / "wsi" / "patient_resnet18_embeddings.parquet"
DEFAULT_METADATA = Path("outputs") / "reports" / "resnet18_embedding_metadata.json"
RANDOM_SEED = 42


def select_tiles(tile_paths: list[Path], max_tiles: int, patient_id: str) -> list[Path]:
    """Select a reproducible random sample instead of the first files alphabetically."""
    if len(tile_paths) <= max_tiles:
        return tile_paths
    patient_seed = RANDOM_SEED + sum(ord(char) for char in patient_id)
    rng = np.random.default_rng(patient_seed)
    selected = rng.choice(len(tile_paths), size=max_tiles, replace=False)
    return [tile_paths[index] for index in sorted(selected)]


def load_encoder(device: torch.device):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    transform = weights.transforms()
    return model, transform, weights


def embed_tiles(
    model,
    transform,
    tile_paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    chunks = []
    with torch.no_grad():
        for start in range(0, len(tile_paths), batch_size):
            batch_paths = tile_paths[start : start + batch_size]
            images = [transform(Image.open(path).convert("RGB")) for path in batch_paths]
            batch = torch.stack(images).to(device)
            embeddings = model(batch).cpu().numpy()
            chunks.append(embeddings)
    return np.vstack(chunks)


def main() -> None:
    parser = ArgumentParser(description="Extract frozen pretrained ResNet18 patient embeddings from WSI tiles.")
    parser.add_argument("--tile-dir", type=Path, default=DEFAULT_TILE_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--max-tiles-per-patient", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    model, transform, weights = load_encoder(device)
    patient_dirs = sorted(path for path in args.tile_dir.iterdir() if path.is_dir())
    rows = []

    for index, patient_dir in enumerate(patient_dirs, start=1):
        all_tile_paths = sorted(patient_dir.glob("*.jpg"))
        tile_paths = select_tiles(
            all_tile_paths,
            max_tiles=args.max_tiles_per_patient,
            patient_id=patient_dir.name,
        )
        print(f"[{index}/{len(patient_dirs)}] {patient_dir.name}: {len(tile_paths)} tiles", flush=True)
        if not tile_paths:
            continue
        tile_embeddings = embed_tiles(
            model=model,
            transform=transform,
            tile_paths=tile_paths,
            batch_size=args.batch_size,
            device=device,
        )
        patient_embedding = tile_embeddings.mean(axis=0)
        row = {
            "PatientID": patient_dir.name,
            "n_tiles_used": len(tile_paths),
            "encoder_name": "torchvision_resnet18_imagenet1k_v1_frozen",
            "pooling": "mean",
        }
        row.update({f"img_feat_{i:03d}": float(value) for i, value in enumerate(patient_embedding)})
        rows.append(row)

        # Checkpoint after each patient; the full run can be resumed/restarted cheaply.
        checkpoint = pd.DataFrame(rows)
        checkpoint.to_csv(args.output_csv, index=False)
        checkpoint.to_parquet(args.output_parquet, index=False)

    features = pd.DataFrame(rows)
    features.to_csv(args.output_csv, index=False)
    features.to_parquet(args.output_parquet, index=False)

    metadata = {
        "encoder_name": "torchvision_resnet18_imagenet1k_v1_frozen",
        "weights_url": weights.url,
        "num_params": weights.meta.get("num_params"),
        "feature_dim": 512,
        "pooling": "mean",
        "max_tiles_per_patient": args.max_tiles_per_patient,
        "tile_selection": "reproducible_random_sample_without_replacement",
        "random_seed": RANDOM_SEED,
        "batch_size": args.batch_size,
        "device": str(device),
        "patients": int(len(features)),
        "total_tiles_used": int(features["n_tiles_used"].sum()) if len(features) else 0,
        "torch_home": os.environ.get("TORCH_HOME"),
    }
    args.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print(f"Patient rows: {len(features)}")
    print("Feature columns: 512")
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_parquet}")
    print(f"Wrote: {args.metadata_path}")


if __name__ == "__main__":
    main()
