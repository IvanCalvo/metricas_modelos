"""
compare_models.py

Reads one or more per-image metric CSV files (columns like:
dataset, model, image, clDice, MCC, confusionMatrix_0..4, avgPrecision,
morph_betti_0/1, morph_correctness, morph_completeness, morph_quality,
morph_apls, distance_*, HCE, ROCAUC ...) and produces a summary table
that aggregates results per (dataset, model), so you can compare models.

Usage:
    python compare_models.py file1.csv file2.csv -o summary.csv

Notes:
- Non-numeric columns (dataset, model, image) are used as grouping keys
  / ignored; every other column is treated as a metric and averaged.
- HCE is treated as "lower is better" for ranking purposes; every other
  numeric metric is treated as "higher is better". Edit LOWER_IS_BETTER
  below if your metrics differ.
- Two tables are produced:
    1. per_dataset_summary: mean (+ std) of every metric, grouped by
       dataset and model.
    2. overall_summary: mean of every metric per model, averaged across
       all datasets (simple mean of the per-dataset means, so each
       dataset counts equally regardless of image count).
  A rank column is added to both, based on a composite score across the
  key metrics available in KEY_METRICS (only the ones actually present
  in your data are used).
"""

import argparse
import sys
import pandas as pd
import numpy as np

# Metrics where a LOWER value is better (everything else is treated as
# higher-is-better). Add/remove names to match your actual columns.
LOWER_IS_BETTER = {
    "HCE",
    "distance_Distance gt to pred",
    "distance_Distance pred to gt",
    "distance_Hausdorff distance",
}

# Metrics used to build the composite rank score (only ones present in
# the data will actually be used).
KEY_METRICS = ["clDice", "MCC", "avgPrecision", "ROCAUC", "HCE",
               "morph_quality", "morph_apls"]

NON_METRIC_COLS = {"dataset", "model", "image"}


def load_data(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = {"dataset", "model"} - set(df.columns)
        if missing:
            sys.exit(f"ERROR: {p} is missing required column(s): {missing}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def numeric_metric_columns(df):
    cols = []
    for c in df.columns:
        if c in NON_METRIC_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def add_rank(summary, metrics_available):
    """Add a composite rank (1 = best) based on z-scored key metrics."""
    keys = [m for m in KEY_METRICS if m in metrics_available]
    if not keys:
        return summary
    z = pd.DataFrame(index=summary.index)
    for m in keys:
        col = summary[m] if m in summary.columns else summary[(m, "mean")]
        std = col.std(ddof=0)
        zscore = (col - col.mean()) / std if std > 0 else 0
        if m in LOWER_IS_BETTER:
            zscore = -zscore
        z[m] = zscore
    summary = summary.copy()
    summary["composite_score"] = z.mean(axis=1)
    summary["rank"] = summary["composite_score"].rank(ascending=False, method="min").astype(int)
    return summary.sort_values("rank")


def build_per_dataset_summary(df, metrics):
    grouped = df.groupby(["dataset", "model"])[metrics]
    summary = grouped.mean()
    summary["n_images"] = grouped.size()
    summary = summary.reset_index()

    ranked_parts = []
    for dataset, part in summary.groupby("dataset"):
        part = part.set_index("model")
        part = add_rank(part, metrics)
        part = part.reset_index()
        part["dataset"] = dataset
        ranked_parts.append(part)
    result = pd.concat(ranked_parts, ignore_index=True)
    cols = ["dataset", "model", "n_images"] + metrics + ["composite_score", "rank"]
    return result[[c for c in cols if c in result.columns]]


def build_overall_summary(per_dataset_summary, metrics):
    # average of per-dataset means -> each dataset weighted equally
    overall = per_dataset_summary.groupby("model")[metrics].mean()
    n_images = per_dataset_summary.groupby("model")["n_images"].sum()
    overall["n_images_total"] = n_images
    overall = add_rank(overall, metrics)
    overall = overall.reset_index()
    cols = ["model", "n_images_total"] + metrics + ["composite_score", "rank"]
    return overall[[c for c in cols if c in overall.columns]]


def main():
    parser = argparse.ArgumentParser(description="Compare model results across CSV files.")
    parser.add_argument("csv_files", nargs="+", help="Paths to the input CSV files")
    parser.add_argument("-o", "--output-prefix", default="summary",
                         help="Prefix for output CSV files (default: 'summary')")
    args = parser.parse_args()

    df = load_data(args.csv_files)
    metrics = numeric_metric_columns(df)
    if not metrics:
        sys.exit("ERROR: no numeric metric columns found.")

    per_dataset = build_per_dataset_summary(df, metrics)
    overall = build_overall_summary(per_dataset, metrics)

    per_dataset_path = f"{args.output_prefix}_per_dataset.csv"
    overall_path = f"{args.output_prefix}_overall.csv"
    per_dataset.to_csv(per_dataset_path, index=False)
    overall.to_csv(overall_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("\n=== Per-dataset summary (mean per model, ranked within each dataset) ===")
    print(per_dataset.round(4).to_string(index=False))

    print("\n=== Overall summary (averaged across datasets, ranked overall) ===")
    print(overall.round(4).to_string(index=False))

    print(f"\nSaved: {per_dataset_path}")
    print(f"Saved: {overall_path}")


if __name__ == "__main__":
    main()