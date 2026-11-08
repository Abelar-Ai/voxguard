import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("download_subset", Path(__file__).parents[1] / "scripts" / "download_subset.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


def test_subset_sampling_is_seeded_balanced_and_covers_attacks():
    rows = [{"utterance_id": f"b-{i}", "label": 0, "attack_id": "-"} for i in range(50)]
    rows += [{"utterance_id": f"s-{attack}-{i}", "label": 1, "attack_id": attack}
             for attack in ("A01", "A02", "A03") for i in range(20)]
    selected = download.choose(rows, 30, seed=475)
    assert selected == download.choose(list(reversed(rows)), 30, seed=475)
    assert sum(r["label"] == 0 for r in selected) == 15
    assert sum(r["label"] == 1 for r in selected) == 15
    assert {r["attack_id"] for r in selected if r["label"] == 1} == {"A01", "A02", "A03"}
    assert len({r["utterance_id"] for r in selected}) == 30


def test_subset_sampling_rejects_odd_counts():
    with pytest.raises(ValueError, match="even"):
        download.choose([], 27, seed=475)
