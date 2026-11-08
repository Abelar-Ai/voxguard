"""Tune augmented acoustic baselines using training/development data only."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits

from .cli import predict_scores, write_json
from .data import read_manifest
from .features import FeatureConfig, load_audio
from .metrics import compute_metrics
from .robust_features import add_white_noise, represent


def select_threshold(labels, scores, default):
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    balanced = (tpr + 1 - fpr) / 2
    candidates = np.flatnonzero(np.isfinite(thresholds) & np.isclose(balanced, balanced.max()))
    if not len(candidates):
        return float(default)
    index = min(candidates, key=lambda index: abs(thresholds[index] - default))
    return float(thresholds[index])


def candidates(seed):
    for c in (0.1, 1.0):
        yield f"logistic_C{c}", make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, random_state=seed)), 0.5
    for c in (1.0, 10.0):
        yield f"svm_C{c}", make_pipeline(StandardScaler(), SVC(C=c, gamma="scale", random_state=seed)), 0.0
    yield "forest_300", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=1, random_state=seed), 0.5


def train(args):
    if args.views < 1:
        raise ValueError("At least one augmented view is required")
    if (args.out / "frozen.json").exists():
        raise ValueError("Selection is already frozen; use a new run directory")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(args.manifest)
    training = [r for r in rows if r["split"] == "train"]
    development = [r for r in rows if r["split"] == "dev"]
    rng = np.random.default_rng(args.seed)
    train_audio = [load_audio(r["path"]) for r in training]
    dev_audio = [load_audio(r["path"]) for r in development]
    variants, labels = [], []
    for clip, row in zip(train_audio, training, strict=True):
        variants.append(clip)
        labels.append(row["label"])
        for _ in range(args.views):
            variants.append(add_white_noise(clip, float(rng.uniform(8, 35)), rng))
            labels.append(row["label"])
    dev_variants = {"clean": dev_audio}
    for snr in (20, 10):
        local_rng = np.random.default_rng(args.seed + snr)
        dev_variants[f"noise_{snr}_db"] = [add_white_noise(clip, snr, local_rng) for clip in dev_audio]
    y_train = np.array(labels)
    y_dev = np.array([r["label"] for r in development])
    all_results, best = [], None
    for representation in args.representations:
        print(f"Extracting {representation}: {len(variants)} training views", flush=True)
        x_train = np.stack([represent(clip, representation) for clip in variants])
        x_dev = {condition: np.stack([represent(clip, representation) for clip in clips])
                 for condition, clips in dev_variants.items()}
        for name, estimator, default_threshold in candidates(args.seed):
            estimator.fit(x_train, y_train)
            scores = {condition: predict_scores(estimator, matrix) for condition, matrix in x_dev.items()}
            threshold = select_threshold(np.tile(y_dev, len(scores)), np.concatenate(list(scores.values())), default_threshold)
            conditions = {condition: compute_metrics(y_dev, score >= threshold, score) for condition, score in scores.items()}
            robust_ba = float(np.mean([value["balanced_accuracy"] for value in conditions.values()]))
            result = {"representation": representation, "model": name, "threshold": threshold,
                      "mean_development_balanced_accuracy": robust_ba, "conditions": conditions}
            all_results.append(result)
            print(f"{representation} {name}: mean dev BA={robust_ba:.3f}, clean={conditions['clean']['accuracy']:.3f}, noise20={conditions['noise_20_db']['accuracy']:.3f}, noise10={conditions['noise_10_db']['accuracy']:.3f}", flush=True)
            key = (robust_ba, conditions["clean"]["balanced_accuracy"])
            if best is None or key > best[0]:
                best = (key, result, estimator)
    bundle = {"model": best[2], "name": best[1]["model"], "representation": best[1]["representation"],
              "threshold": best[1]["threshold"], "config": vars(FeatureConfig()), "seed": args.seed,
              "training_ids": [r["utterance_id"] for r in training],
              "training_speakers": sorted({r["speaker_id"] for r in training}),
              "training_sha256": [r["sha256"] for r in training]}
    joblib.dump(bundle, args.out / "selected.joblib")
    write_json(args.out / "development_results.json", all_results)
    write_json(args.out / "frozen.json", {"selected": best[1], "seed": args.seed,
               "train_recordings": len(training), "train_views": len(variants), "dev_recordings": len(development),
               "augmentation": {"views_per_recording": args.views, "noise": "Gaussian white", "snr_db_uniform": [8, 35]},
               "selection": "mean balanced accuracy across clean, 20 dB and 10 dB development conditions",
               "threshold": "chosen on pooled development scores only", "evaluation_used": False})
    print(f"Frozen: {best[1]['representation']} {best[1]['model']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=475)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--representations", nargs="+", choices=["mfcc", "lfcc", "combined"], default=["mfcc", "lfcc", "combined"])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        train(args)


if __name__ == "__main__":
    main()
