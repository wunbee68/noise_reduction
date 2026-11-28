import csv

from .audio import read_wav


def manifest(root, split):
    with (root / split / "samples.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row["segment_id"] for row in rows}) != len(rows):
        raise ValueError(f"Empty manifest or duplicate segment IDs in {split}")
    if any(row["split"] != split for row in rows):
        raise ValueError(f"Manifest contains a file from another split: {split}")
    return rows


def read_pair(root, row):
    noisy, _ = read_wav(root / row["noisy_path"], challenge_format=True)
    clean, _ = read_wav(root / row["clean_path"], challenge_format=True)
    return noisy[:, 0], clean[:, 0]
