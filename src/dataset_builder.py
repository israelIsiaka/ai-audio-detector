import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from feature_extractor import extract_features
import warnings
warnings.filterwarnings("ignore")

SUPPORTED_FORMATS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")


def build_dataset(
    natural_dir: str,
    ai_dir: str,
    output_csv: str = "results/dataset.csv",
    max_files: int = None
) -> pd.DataFrame:
    """
    Process all audio files in both directories,
    extract features, label them, and save to CSV.

    Label: 0 = Natural, 1 = AI Generated
    """

    rows = []
    errors = []

    # ── Process Natural Audio ─────────────────────────────────────────
    print("\n" + "="*55)
    print("PROCESSING NATURAL AUDIO")
    print("="*55)

    natural_files = [
        f for f in os.listdir(natural_dir)
        if f.lower().endswith(SUPPORTED_FORMATS)
    ]

    if max_files:
        natural_files = natural_files[:max_files]

    print(f"Found {len(natural_files)} natural audio files\n")

    for fname in tqdm(natural_files, desc="Natural"):
        path = os.path.join(natural_dir, fname)
        try:
            feats = extract_features(path)
            feats["label"]      = 0
            feats["label_name"] = "natural"
            feats["source_dir"] = "natural"
            rows.append(feats)
        except Exception as e:
            print(f"  ❌ Failed: {fname} — {e}")
            errors.append({"file": fname, "error": str(e)})

    # ── Process AI Audio ──────────────────────────────────────────────
    print("\n" + "="*55)
    print("PROCESSING AI GENERATED AUDIO")
    print("="*55)

    ai_files = [
        f for f in os.listdir(ai_dir)
        if f.lower().endswith(SUPPORTED_FORMATS)
    ]

    if max_files:
        ai_files = ai_files[:max_files]

    print(f"Found {len(ai_files)} AI audio files\n")

    for fname in tqdm(ai_files, desc="AI"):
        path = os.path.join(ai_dir, fname)
        try:
            feats = extract_features(path)
            feats["label"]      = 1
            feats["label_name"] = "ai_generated"
            feats["source_dir"] = "ai_generated"
            rows.append(feats)
        except Exception as e:
            print(f"  ❌ Failed: {fname} — {e}")
            errors.append({"file": fname, "error": str(e)})

    # ── Build DataFrame ───────────────────────────────────────────────
    df = pd.DataFrame(rows)

    # Drop non-numeric columns for ML (keep for reference)
    df = df.replace({None: np.nan})

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    df.to_csv(output_csv, index=False)

    # ── Summary Report ────────────────────────────────────────────────
    print("\n" + "="*55)
    print("DATASET BUILD COMPLETE")
    print("="*55)
    print(f"  Total samples     : {len(df)}")
    print(f"  Natural samples   : {len(df[df['label'] == 0])}")
    print(f"  AI samples        : {len(df[df['label'] == 1])}")
    print(f"  Features per file : {len(df.columns) - 4}")
    print(f"  Saved to          : {output_csv}")

    if errors:
        print(f"\n  ⚠️  {len(errors)} files failed — check errors below:")
        for e in errors:
            print(f"     {e['file']}: {e['error']}")

    print("="*55)

    return df


def dataset_summary(csv_path: str):
    """
    Print a statistical summary of your dataset.
    Useful for your paper's data section.
    """
    df = pd.read_csv(csv_path)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "label"]

    print("\n" + "="*55)
    print("DATASET STATISTICAL SUMMARY")
    print("="*55)
    print(f"Total samples : {len(df)}")
    print(f"Natural       : {len(df[df['label']==0])}")
    print(f"AI Generated  : {len(df[df['label']==1])}")
    print(f"Features      : {len(numeric_cols)}")

    # Key features comparison
    key_features = [
        "f0_std", "f0_range", "jitter_local",
        "shimmer_local", "hnr_mean", "spectral_flatness_mean",
        "mfcc_delta_mean", "zcr_mean"
    ]

    print("\n── Mean values per class ──────────────────────────")
    print(f"{'Feature':<30} {'Natural':>10} {'AI':>10} {'Diff %':>10}")
    print("-"*55)

    nat = df[df["label"] == 0]
    ai  = df[df["label"] == 1]

    for feat in key_features:
        if feat not in df.columns:
            continue
        nat_mean = nat[feat].mean()
        ai_mean  = ai[feat].mean()
        if nat_mean and nat_mean != 0:
            diff_pct = ((ai_mean - nat_mean) / abs(nat_mean)) * 100
        else:
            diff_pct = 0
        print(f"{feat:<30} {nat_mean:>10.4f} {ai_mean:>10.4f} {diff_pct:>9.1f}%")

    print("="*55)
    return df


# ── Run ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # Default: use existing data folders
        df = build_dataset(
            natural_dir="data/natural",
            ai_dir="data/ai_generated",
            output_csv="results/dataset.csv"
        )
        dataset_summary("results/dataset.csv")

    elif sys.argv[1] == "summary":
        dataset_summary("results/dataset.csv")

    else:
        print("Usage:")
        print("  python src/dataset_builder.py           # build dataset")
        print("  python src/dataset_builder.py summary   # summary of existing dataset")