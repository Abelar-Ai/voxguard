import csv
import hashlib
from pathlib import Path


def read_manifest(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest is empty")
    for row in rows:
        row["label"] = int(row["label"])
        if row["label"] not in (0, 1) or row["split"] not in ("train", "dev", "eval"):
            raise ValueError("Invalid class or partition")
        audio_path = Path(row["path"])
        row["path"] = str((Path(path).parent / audio_path).resolve()) if not audio_path.is_absolute() else str(audio_path)
        if not Path(row["path"]).is_file():
            raise ValueError(f"Missing audio: {row['path']}")
        actual_hash = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
        if row.get("sha256") and row["sha256"] != actual_hash:
            raise ValueError(f"Audio hash mismatch for {row['utterance_id']}")
        row["sha256"] = actual_hash
    validate_partitions(rows)
    return rows


def validate_partitions(rows):
    ids, speakers, hashes = {}, {}, {}
    for row in rows:
        split = row["split"]
        if row["utterance_id"] in ids:
            raise ValueError("Duplicate utterance ID in manifest")
        ids[row["utterance_id"]] = split
        for value, seen, kind in ((row["speaker_id"], speakers, "speaker"),
                                   (row["sha256"], hashes, "audio hash")):
            if value in seen and seen[value] != split:
                raise ValueError(f"Cross-partition leakage through {kind}")
            seen[value] = split
    for split in ("train", "dev", "eval"):
        group = [r for r in rows if r["split"] == split]
        if {r["label"] for r in group} != {0, 1}:
            raise ValueError(f"Partition {split} must contain both classes")
