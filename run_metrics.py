"""
Loop over prediction files and compute segmentation metrics against
the matching ground-truth labels.

Expected folder layout (relative to ROOT):

ROOT/
├── predictions/
│   └── <DATASET>/
│       └── <MODEL>/
│           └── <ImageName>.npz
└── datasets/
    └── <DATASET>/
        └── val/
            └── label/
                └── <ImageName>_1stHO.png

Example:
  predictions/CHASEDB/AGNet/Image_08L.npz
  datasets/CHASEDB/val/label/Image_08L_1stHO.png

Fill in `calculate_metrics()` with your real implementation. It should
accept a prediction array and a label array and return a dict of
metric_name -> value.
"""

import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import torch

from run_metrics_pipeline import run_metrics
from metrics_selection import Metrics

ROOT = Path("/run/media/ivancalvo/SSD drive/backup/")          # <-- change this
PRED_ROOT = ROOT / "predictions"
DATA_ROOT = ROOT / "restantes"

LABEL_SUFFIX = "_1stHO.png"           # how labels relate to the image stem


def calculate_metrics(pred: np.ndarray, label: np.ndarray, metrics_config: Metrics) -> dict:
    pred = binarize(pred, 0.5)
    metrics_results = run_metrics(pred, label, metrics_config)
    return metrics_results

def binarize(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """Threshold a tensor into a binary {0, 1} float tensor."""
    return (tensor > threshold).float()


def load_prediction(npz_path: Path) -> torch.Tensor:
    """Load the array stored in a .npz file (grabs the first array found)
    and convert it to a torch.Tensor of dtype float32.
 
    Predictions here are raw probability maps (e.g. sigmoid output,
    values in [0, 1]) — no thresholding is applied here, so
    calculate_metrics() can decide the threshold itself.
    """
    with np.load(npz_path) as data:
        keys = list(data.keys())
        if not keys:
            raise ValueError(f"No arrays found in {npz_path}")
        # If you know the exact key (e.g. "pred"), use data["pred"] instead.
        array = data[keys[0]]
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()
        return tensor.squeeze()  # drop any singleton batch/channel dims
 
 
def load_label(png_path: Path) -> torch.Tensor:
    """Load a label PNG (typically 0/255 uint8) as a torch.Tensor.
 
    Values stay in their original range (e.g. 0-255) — normalize to
    {0, 1} inside calculate_metrics() if needed.
    """
    array = np.array(Image.open(png_path).convert("L"))
    return torch.from_numpy(np.ascontiguousarray(array)).float()


def find_label_path(dataset: str, image_stem: str) -> Path:
    return DATA_ROOT / dataset / "val" / "label" / f"{image_stem}.png"

def to_python(value):
    """Convert torch tensors / numpy scalars into plain Python types."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.tolist()  # multi-element tensor -> list
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def flatten_metrics(d: dict, parent_key: str = "") -> dict:
    """
    Recursively flatten a (possibly nested) metrics dict into a single
    flat dict of scalar values, suitable for a DataFrame row.
 
    - Nested dicts -> keys joined with "_"  (e.g. distance_Hausdorff distance)
    - Tuples/lists  -> one column per element (e.g. confusionMatrix_0, _1, ...)
    - torch.Tensor / numpy scalars -> plain Python numbers
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}_{k}" if parent_key else str(k)
 
        if isinstance(v, dict):
            items.update(flatten_metrics(v, new_key))
        elif isinstance(v, (tuple, list)):
            for i, sub in enumerate(v):
                sub_key = f"{new_key}_{i}"
                if isinstance(sub, dict):
                    items.update(flatten_metrics(sub, sub_key))
                else:
                    items[sub_key] = to_python(sub)
        else:
            items[new_key] = to_python(v)
    return items
 


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main():
    results = []

    dataset_dirs = sorted(p for p in PRED_ROOT.iterdir() if p.is_dir())

    metrics_config = Metrics()
    metrics_config.disable("graphDistances")

    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.name

        model_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())

        for model_dir in model_dirs:
            model_name = model_dir.name

            npz_files = sorted(model_dir.glob("*.npz"))

            for npz_path in npz_files:
                image_stem = npz_path.stem  # e.g. "Image_08L"
                label_path = find_label_path(dataset_name, image_stem)

                print(f":: Calculating for {dataset_name} -{model_name} - {image_stem}")

                if not label_path.exists():
                    print(f"[WARN] Missing label for {npz_path} "
                          f"(expected {label_path}) — skipping.")
                    continue

                try:
                    pred = load_prediction(npz_path)
                    label = load_label(label_path)
                    metrics = calculate_metrics(pred, label, metrics_config)
                except Exception as e:
                    print(f"[ERROR] Failed on {npz_path}: {e}")
                    continue

                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "image": image_stem,
                }
                row.update(flatten_metrics(metrics))
                results.append(row)

    df = pd.DataFrame(results)
    out_path = ROOT / "metrics_results_stare_hrf.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    if not df.empty:
        print("\nAverage metrics per dataset/model:")
        numeric_cols = df.select_dtypes(include="number").columns
        print(df.groupby(["dataset", "model"])[numeric_cols].mean())


if __name__ == "__main__":
    main()