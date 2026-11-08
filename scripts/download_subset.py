"""Download a seeded ASVspoof 2019 LA pilot using HTTP byte ranges.

Only selected members of the official archive are downloaded. No dataset mirror
or relabeling is used. Official protocol partitions and speaker IDs are retained.
"""
import argparse
import csv
import hashlib
import io
import json
import random
import re
import struct
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

URL = "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
SOURCE = "https://doi.org/10.7488/ds/2555"
_local = threading.local()


def session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def fetch_range(start, end):
    for attempt in range(4):
        try:
            with session().get(URL, headers={"Range": f"bytes={start}-{end}",
                                           "Accept-Encoding": "identity"},
                               stream=True, timeout=45) as response:
                response.raise_for_status()
                header = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", header)
                if response.status_code != 206 or not match:
                    raise ValueError("Server must honor byte ranges; refusing full archive")
                actual_start, actual_end, size = map(int, match.groups())
                if start >= 0 and actual_start != start:
                    raise ValueError(f"Incorrect range returned: {header}")
                if start >= 0 and actual_end != min(end, size - 1):
                    raise ValueError(f"Incomplete range returned: {header}")
                data = response.raw.read(actual_end - actual_start + 2)
                if len(data) != actual_end - actual_start + 1:
                    raise ValueError("Truncated or oversized byte range")
                return data, size
        except (requests.RequestException, ValueError):
            if attempt == 3:
                raise
            time.sleep(0.5 * 2 ** attempt)


class RemoteArchive(io.RawIOBase):
    def __init__(self):
        _, self.size = fetch_range(0, 0)
        tail, _ = fetch_range(max(0, self.size - 65536), self.size - 1)
        self.tail = tail
        self.position = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.size) + offset
        if self.position < 0:
            raise ValueError("Negative file position")
        return self.position

    def tell(self):
        return self.position

    def read(self, n=-1):
        n = self.size - self.position if n < 0 else min(n, self.size - self.position)
        if n <= 0:
            return b""
        if n > 32 * 1024 * 1024:
            raise ValueError("Refusing a range larger than 32 MiB")
        if self.position >= self.size - len(self.tail):
            data = self.tail[self.position - (self.size - len(self.tail)):][:n]
        else:
            data, _ = fetch_range(self.position, self.position + n - 1)
        self.position += len(data)
        return data


def member_bytes(info):
    # Read the local header and payload together to avoid several HTTP round trips.
    end = info.header_offset + 30 + len(info.filename.encode()) + 256 + info.compress_size
    data, _ = fetch_range(info.header_offset, end)
    if data[:4] != b"PK\x03\x04":
        raise ValueError("Invalid ZIP local header")
    name_length, extra_length = struct.unpack_from("<HH", data, 26)
    offset = 30 + name_length + extra_length
    if offset + info.compress_size > len(data):
        data, _ = fetch_range(info.header_offset, info.header_offset + offset + info.compress_size - 1)
    compressed = data[offset:offset + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        content = compressed
    else:
        raise ValueError(f"Unsupported ZIP compression {info.compress_type}")
    if len(content) != info.file_size or zlib.crc32(content) & 0xffffffff != info.CRC:
        raise ValueError(f"ZIP integrity check failed for {info.filename}")
    return content


def choose(rows, count, seed):
    """Balance classes; distribute spoof samples equally across attack IDs."""
    if count < 26 or count % 2:
        raise ValueError("Subset sizes must be even and at least 26")
    rng = random.Random(seed)
    genuine = sorted((r for r in rows if r["label"] == 0), key=lambda r: r["utterance_id"])
    if len(genuine) < count // 2:
        raise ValueError("Not enough bona fide recordings")
    selected = rng.sample(genuine, count // 2)
    attacks = sorted({r["attack_id"] for r in rows if r["label"] == 1})
    base, extra = divmod(count // 2, len(attacks))
    for index, attack in enumerate(attacks):
        group = sorted((r for r in rows if r["attack_id"] == attack), key=lambda r: r["utterance_id"])
        selected.extend(rng.sample(group, base + (index < extra)))
    return sorted(selected, key=lambda r: r["utterance_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=1200)
    parser.add_argument("--dev", type=int, default=400)
    parser.add_argument("--eval", type=int, default=400)
    parser.add_argument("--seed", type=int, default=475)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--exclude-manifest", type=Path, help="Exclude previously inspected evaluation utterances")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(RemoteArchive())
    entries = {i.filename: i for i in archive.infolist()}
    by_name = {Path(name).name: info for name, info in entries.items() if name.endswith(".flac")}
    selected = []
    excluded = set()
    if args.exclude_manifest:
        with args.exclude_manifest.open(newline="") as handle:
            excluded = {r["utterance_id"] for r in csv.DictReader(handle) if r["split"] == "eval"}
    protocol_dir = args.out / "protocols"
    protocol_dir.mkdir(exist_ok=True)
    for split, count in (("train", args.train), ("dev", args.dev), ("eval", args.eval)):
        matches = [info for name, info in entries.items()
                   if "cm_protocols/" in name and f".cm.{split}." in name and name.endswith(".txt")]
        if len(matches) != 1:
            raise ValueError(f"Expected one CM protocol for {split}, found {[x.filename for x in matches]}")
        content = member_bytes(matches[0])
        (protocol_dir / Path(matches[0].filename).name).write_bytes(content)
        rows = []
        for line in content.decode().splitlines():
            speaker, utterance, placeholder, attack, key = line.split()
            if key not in {"bonafide", "spoof"}:
                raise ValueError(f"Unexpected protocol label {key}")
            rows.append({"split": split, "speaker_id": speaker, "utterance_id": utterance,
                         "attack_id": attack, "label": int(key == "spoof")})
        if split == "eval":
            rows = [r for r in rows if r["utterance_id"] not in excluded]
        group = choose(rows, count, args.seed + {"train": 0, "dev": 1, "eval": 2}[split])
        selected.extend(group)
        print(f"Selected {len(group)} {split} utterances from {len(rows)}", flush=True)

    def download(row):
        info = by_name[row["utterance_id"] + ".flac"]
        path = args.out / row["split"] / Path(info.filename).name
        path.parent.mkdir(exist_ok=True)
        if path.exists() and len(path.read_bytes()) == info.file_size and zlib.crc32(path.read_bytes()) & 0xffffffff == info.CRC:
            content = path.read_bytes()
        else:
            content = member_bytes(info)
            path.write_bytes(content)
        return dict(row, path=str(path.relative_to(args.out)), sha256=hashlib.sha256(content).hexdigest(),
                    source_member=info.filename)

    complete = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, row) for row in selected]
        for future in as_completed(futures):
            complete.append(future.result())
            if len(complete) % 100 == 0:
                print(f"Downloaded and verified {len(complete)}/{len(selected)} recordings", flush=True)
    complete.sort(key=lambda r: (r["split"], r["utterance_id"]))
    with (args.out / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(complete[0]))
        writer.writeheader()
        writer.writerows(complete)
    metadata = {"source": SOURCE, "archive_url": URL, "seed": args.seed,
                "selection": "equal class counts and approximately equal counts per spoof attack within each official partition",
                "counts": {"train": args.train, "dev": args.dev, "eval": args.eval},
                "excluded_prior_evaluation_utterances": len(excluded),
                "license": "Open Data Commons Attribution License (ODC-By)",
                "citation": "Todisco et al., ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection, Interspeech 2019, doi:10.21437/Interspeech.2019-2249"}
    (args.out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Manifest saved to {args.out / 'manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
