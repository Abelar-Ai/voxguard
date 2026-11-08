"""Compare frozen models on fresh held-out audio under matched conditions."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .cli import predict_scores, write_json
from .data import read_manifest
from .features import FeatureConfig, features_from_audio, load_audio
from .metrics import compute_metrics
from .robust_features import add_white_noise, represent


def assert_holdout_safe(rows, metadata, prior_rows):
    if not rows:
        raise ValueError("Empty evaluation partition")
    for field, training_key in (("utterance_id", "training_ids"), ("speaker_id", "training_speakers"), ("sha256", "training_sha256")):
        if {r[field] for r in rows} & set(metadata[training_key]):
            raise ValueError(f"Evaluation overlaps training through {field}")
    prior_eval = [r for r in prior_rows if r["split"] == "eval"]
    if {r["utterance_id"] for r in rows} & {r["utterance_id"] for r in prior_eval}:
        raise ValueError("Previously inspected evaluation utterance detected")
    if {r["sha256"] for r in rows} & {r["sha256"] for r in prior_eval}:
        raise ValueError("Previously inspected evaluation audio hash detected")


def acoustic_scores(bundle, clips):
    if "representation" in bundle:
        values = np.stack([represent(clip, bundle["representation"]) for clip in clips])
        threshold = bundle["threshold"]
    else:
        values = np.stack([features_from_audio(clip, FeatureConfig(**bundle["config"])) for clip in clips])
        threshold = 0.5 if hasattr(bundle["model"], "predict_proba") else 0.0
    scores = predict_scores(bundle["model"], values)
    return scores, scores >= threshold


def benchmark(args):
    if (args.out / "results.json").exists():
        raise ValueError("Benchmark already exists; refusing to overwrite inspected results")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [row for row in read_manifest(args.manifest) if row["split"] == "eval"]
    with args.exclude_manifest.open(newline="") as handle:
        prior_rows = list(csv.DictReader(handle))
    predictors = {}
    for name, path in (("original_mfcc_svm", args.baseline), ("augmented_acoustic", args.robust)):
        if path is not None:
            if name == "augmented_acoustic" and not (path.parent / "frozen.json").exists():
                raise ValueError("Robust model must be frozen before evaluation")
            bundle = joblib.load(path)
            assert_holdout_safe(rows, bundle, prior_rows)
            predictors[name] = (path, bundle)
    if args.cnn:
        import torch
        if not (args.cnn.parent / "frozen.json").exists():
            raise ValueError("CNN model must be frozen before evaluation")
        metadata = torch.load(args.cnn, map_location="cpu", weights_only=True)["metadata"]
        assert_holdout_safe(rows, metadata, prior_rows)
        predictors["cnn"] = (args.cnn, metadata)
    if not predictors:
        raise ValueError("At least one frozen model is required")
    # Record the model hashes and evaluation IDs before computing any scores.
    write_json(args.out / "protocol.json", {
        "models": {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, (path, _) in predictors.items()},
        "evaluation_count": len(rows), "evaluation_ids": [r["utterance_id"] for r in rows],
        "prior_evaluation_excluded": True, "thresholds_fixed_before_evaluation": True,
        "seed": args.seed, "conditions": ["clean", "noise_20_db", "noise_10_db"]})
    audio = [load_audio(row["path"]) for row in rows]
    labels = np.array([row["label"] for row in rows])
    results = {name: {} for name in predictors}
    for condition, snr in (("clean", None), ("noise_20_db", 20), ("noise_10_db", 10)):
        rng = np.random.default_rng(args.seed + (snr or 0))
        clips = audio if snr is None else [add_white_noise(clip, snr, rng) for clip in audio]
        for name, (path, bundle) in predictors.items():
            if name == "cnn":
                from .neural import infer
                scores, _ = infer(path, clips, device=args.device)
                predictions = scores >= 0.5
            else:
                scores, predictions = acoustic_scores(bundle, clips)
            metrics = compute_metrics(labels, predictions, scores)
            metrics["per_attack_synthetic_recall"] = {}
            for attack in sorted({row["attack_id"] for row in rows if row["label"] == 1}):
                mask = np.array([row["attack_id"] == attack and row["label"] == 1 for row in rows])
                metrics["per_attack_synthetic_recall"][attack] = {"n": int(mask.sum()), "recall": float(np.mean(predictions[mask]))}
            results[name][condition] = metrics
            with (args.out / f"{name}_{condition}_predictions.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["utterance_id", "label", "prediction", "score"])
                writer.writerows((row["utterance_id"], row["label"], int(prediction), float(score))
                                 for row, prediction, score in zip(rows, predictions, scores, strict=True))
            print(f"{name} {condition}: accuracy={metrics['accuracy']:.3f}, AUC={metrics['roc_auc']:.3f}, EER={metrics['eer']:.3f}", flush=True)
    write_json(args.out / "results.json", results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, default=Path("runs/pilot/subset_manifest.csv"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--robust", type=Path)
    parser.add_argument("--cnn", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1475)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        benchmark(args)


if __name__ == "__main__":
    main()
