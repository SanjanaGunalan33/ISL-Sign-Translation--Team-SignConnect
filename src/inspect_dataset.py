"""Step 1: clean the iSign annotation metadata.

Reads the manifest (and optionally iSign_v1.1.csv to cross-check text), flags
problems, and writes data/clean_annotations.csv with columns:
    uid, text, video_path_in_archive, archive

Usage:
    python -m src.inspect_dataset
    python -m src.inspect_dataset --manifest path/to/isign_manifest.csv --annotations path/to/iSign_v1.1.csv
"""
import argparse
import re
import unicodedata

import pandas as pd

from src import config


def normalize_text(s: str) -> str:
    """Light, consistent normalization. Keep it identical at train and eval time."""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def find_col(df, *candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of {candidates} found. Columns are: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(config.MANIFEST_CSV))
    ap.add_argument("--annotations", default=str(config.ANNOTATIONS_CSV),
                    help="iSign_v1.1.csv; used to fill/cross-check text if present")
    ap.add_argument("--out", default=str(config.CLEAN_CSV))
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    print(f"Manifest rows: {len(man):,}")
    print(f"Columns: {list(man.columns)}")

    uid_c = find_col(man, "uid", "id")
    path_c = find_col(man, "video_path_in_archive", "video_path", "path")
    arch_c = find_col(man, "archive", "archive_name")
    df = pd.DataFrame({
        "uid": man[uid_c].astype(str).str.strip(),
        "video_path_in_archive": man[path_c].astype(str).str.strip(),
        "archive": man[arch_c].astype(str).str.strip(),
    })
    df["text"] = man["text"] if "text" in man.columns else pd.NA

    # Cross-check with the original annotation file if available
    try:
        ann = pd.read_csv(args.annotations)
        ann = ann.rename(columns={find_col(ann, "uid", "id"): "uid"})
        ann["uid"] = ann["uid"].astype(str).str.strip()
        ann = ann.drop_duplicates("uid")[["uid", "text"]].rename(columns={"text": "text_ann"})
        df = df.merge(ann, on="uid", how="left")
        df["text"] = df["text"].where(df["text"].notna(), df["text_ann"])
        both = df["text_ann"].notna() & df["text"].notna()
        mismatch = (df.loc[both, "text"].map(normalize_text) != df.loc[both, "text_ann"].map(normalize_text)).sum()
        print(f"Text mismatches between manifest and {args.annotations}: {mismatch:,}")
        df = df.drop(columns="text_ann")
    except FileNotFoundError:
        print("Annotation csv not found; using manifest text only.")

    report = {}
    df["text"] = df["text"].fillna("").map(normalize_text)
    report["missing_text"] = int((df["text"] == "").sum())
    bad_path = df["video_path_in_archive"].isin(["", "nan"]) | ~df["video_path_in_archive"].str.contains(r"\.\w{2,4}$", regex=True)
    report["malformed_path"] = int(bad_path.sum())
    report["duplicate_uid_rows"] = int(df["uid"].duplicated(keep=False).sum())
    report["duplicate_path_rows"] = int(df["video_path_in_archive"].duplicated(keep=False).sum())

    keep = (df["text"] != "") & ~bad_path
    keep &= ~df["uid"].duplicated(keep="first")
    keep &= ~df["video_path_in_archive"].duplicated(keep="first")
    clean = df[keep].reset_index(drop=True)

    # Useful stats for the report
    n_words = clean["text"].str.split().str.len()
    report["kept_rows"] = len(clean)
    report["distinct_sentences"] = int(clean["text"].nunique())
    report["rows_whose_sentence_repeats"] = int(clean["text"].duplicated(keep=False).sum())
    report["words_per_sentence_median"] = float(n_words.median())
    report["words_per_sentence_p95"] = float(n_words.quantile(0.95))
    report["archives"] = clean["archive"].value_counts().to_dict()

    print("\n=== Cleaning report ===")
    for k, v in report.items():
        print(f"{k:32s} {v}")

    clean[["uid", "text", "video_path_in_archive", "archive"]].to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
