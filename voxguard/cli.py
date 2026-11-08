"""Train, evaluate, and use reproducible VoxGuard baselines."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .data import read_manifest
from .features import FeatureConfig, extract, features_from_audio, load_audio
from .metrics import compute_metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def candidates(seed):
    yield "majority", "most_frequent", DummyClassifier(strategy="most_frequent")
    for c in (0.1, 1.0, 10.0):
        yield "logistic_regression", f"C={c}", make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=seed))
    for c in (0.1, 1.0, 10.0):
        yield "svm_rbf", f"C={c},gamma=scale", make_pipeline(StandardScaler(), SVC(C=c, gamma="scale", random_state=seed))
    for depth in (8, None):
        yield "random_forest", f"trees=200,max_depth={depth}", RandomForestClassifier(n_estimators=200, max_depth=depth, random_state=seed, n_jobs=1)


def predict_scores(model, matrix):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(matrix)[:, list(model.classes_).index(1)]
    return model.decision_function(matrix)


def matrix(rows, config, cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_key = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:16]
    values = []
    for index, row in enumerate(rows):
        path = cache_dir / f"{row['sha256']}_{feature_key}.npy"
        if path.exists():
            value = np.load(path, allow_pickle=False)
            if value.shape != (4 * config.n_mfcc,) or not np.isfinite(value).all():
                raise ValueError("Invalid cached feature vector")
        else:
            value = extract(row["path"], config)
            np.save(path, value)
        values.append(value)
        if (index + 1) % 100 == 0:
            print(f"Features {index + 1}/{len(rows)}", flush=True)
    return np.stack(values), np.array([r["label"] for r in rows])


def save_predictions(path, rows, predictions, scores):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utterance_id", "speaker_id", "attack_id", "label", "prediction", "synthetic_score"])
        for row, prediction, score in zip(rows, predictions, scores, strict=True):
            writer.writerow([row["utterance_id"], row["speaker_id"], row["attack_id"], row["label"], int(prediction), float(score)])


def plot_confusion(path, metric, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ConfusionMatrixDisplay(np.array(metric["confusion_matrix"]), display_labels=["Genuine", "Synthetic"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train(args):
    rows = read_manifest(args.manifest)
    config = FeatureConfig()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "selection.json").exists():
        raise ValueError("Run already has a frozen selection; choose a new output directory")
    cache = args.cache if args.cache is not None else args.out / "feature_cache"
    training = [r for r in rows if r["split"] == "train"]
    development = [r for r in rows if r["split"] == "dev"]
    x_train, y_train = matrix(training, config, cache)
    x_dev, y_dev = matrix(development, config, cache)
    results, best = [], {}
    for name, settings, model in candidates(args.seed):
        start = time.perf_counter()
        model.fit(x_train, y_train)
        seconds = time.perf_counter() - start
        predictions, scores = model.predict(x_dev), predict_scores(model, x_dev)
        metric = compute_metrics(y_dev, predictions, scores)
        result = {"model": name, "settings": settings, "training_seconds": seconds, **metric}
        results.append(result)
        if name not in best or metric["balanced_accuracy"] > best[name]["metrics"]["balanced_accuracy"]:
            best[name] = {"model": model, "metrics": result}
        print(f"{name} {settings}: dev balanced accuracy={metric['balanced_accuracy']:.3f}, AUC={metric['roc_auc']:.3f}", flush=True)
    selection = max((name for name in best if name != "majority"), key=lambda name: best[name]["metrics"]["balanced_accuracy"])
    for name, value in best.items():
        bundle = {"model": value["model"], "name": name, "config": asdict(config),
                  "positive_class": "spoof", "seed": args.seed,
                  "training_ids": [r["utterance_id"] for r in training],
                  "training_speakers": sorted({r["speaker_id"] for r in training}),
                  "training_sha256": [r["sha256"] for r in training]}
        joblib.dump(bundle, args.out / f"{name}.joblib")
        predictions = value["model"].predict(x_dev)
        scores = predict_scores(value["model"], x_dev)
        save_predictions(args.out / f"dev_{name}_predictions.csv", development, predictions, scores)
    versions = {package: importlib.metadata.version(package) for package in ("numpy", "scipy", "scikit-learn", "librosa", "soundfile", "joblib")}
    write_json(args.out / "development_results.json", results)
    write_json(args.out / "selection.json", {
        "selection_metric": "development balanced accuracy", "selected_model": selection,
        "selected_hyperparameters": {name: value["metrics"]["settings"] for name, value in best.items()},
        "seed": args.seed, "features": asdict(config), "feature_dimensions": int(x_train.shape[1]),
        "train_count": len(training), "dev_count": len(development),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "python": platform.python_version(), "packages": versions,
        "threshold_policy": "default classifier thresholds; no evaluation-set threshold tuning",
        "evaluation_policy": "all family choices frozen on development data before evaluation"})
    print(f"Frozen model selection: {selection}", flush=True)


def evaluate(args):
    selection = json.loads((args.run / "selection.json").read_text())
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != selection["manifest_sha256"]:
        raise ValueError("Manifest differs from the training run")
    rows = [r for r in read_manifest(args.manifest) if r["split"] == "eval"]
    config = FeatureConfig(**selection["features"])
    x_eval, y_eval = matrix(rows, config, args.run / "feature_cache")
    all_results = {}
    for name in selection["selected_hyperparameters"]:
        bundle = joblib.load(args.run / f"{name}.joblib")
        start = time.perf_counter()
        predictions, scores = bundle["model"].predict(x_eval), predict_scores(bundle["model"], x_eval)
        inference_ms = 1000 * (time.perf_counter() - start) / len(rows)
        metric = compute_metrics(y_eval, predictions, scores)
        metric["classifier_prediction_and_score_ms_per_clip"] = inference_ms
        metric["selected_on_development"] = name == selection["selected_model"]
        metric["per_attack_synthetic_recall"] = {}
        for attack in sorted({r["attack_id"] for r in rows if r["label"] == 1}):
            mask = np.array([r["attack_id"] == attack and r["label"] == 1 for r in rows])
            metric["per_attack_synthetic_recall"][attack] = {"n": int(mask.sum()), "recall": float(np.mean(predictions[mask] == 1))}
        all_results[name] = metric
        save_predictions(args.run / f"eval_{name}_predictions.csv", rows, predictions, scores)
        plot_confusion(args.run / f"eval_{name}_confusion.png", metric, name.replace("_", " ").title())
        print(f"{name}: eval accuracy={metric['accuracy']:.3f}, AUC={metric['roc_auc']:.3f}, EER={metric['eer']:.3f}", flush=True)
    write_json(args.run / "evaluation_results.json", all_results)


def predict(args):
    bundle = joblib.load(args.model)
    start = time.perf_counter()
    if "representation" in bundle:
        from .robust_features import represent
        audio = load_audio(args.audio, FeatureConfig(**bundle["config"]))
        vector = represent(audio, bundle["representation"])[None, :]
        score = float(predict_scores(bundle["model"], vector)[0])
        prediction = int(score >= bundle["threshold"])
    else:
        vector = extract(args.audio, FeatureConfig(**bundle["config"]))[None, :]
        prediction = int(bundle["model"].predict(vector)[0])
        score = float(predict_scores(bundle["model"], vector)[0])
    print(json.dumps({"model": bundle["name"], "prediction": "synthetic" if prediction else "genuine",
                      "synthetic_score": score, "score_is_calibrated_confidence": False,
                      "elapsed_ms": 1000 * (time.perf_counter() - start),
                      "scope": "Research pilot; a prediction is not proof of authenticity"}, indent=2))


def robustness(args):
    selection = json.loads((args.run / "selection.json").read_text())
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != selection["manifest_sha256"]:
        raise ValueError("Manifest differs from the training run")
    rows = [r for r in read_manifest(args.manifest) if r["split"] == "eval"]
    name = selection["selected_model"]
    bundle = joblib.load(args.run / f"{name}.joblib")
    config = FeatureConfig(**bundle["config"])
    rng = np.random.default_rng(selection["seed"])
    audio = [load_audio(r["path"], config) for r in rows]
    labels = np.array([r["label"] for r in rows])
    results = {}
    for snr in (20, 10):
        values = []
        for clip in audio:
            noise = rng.normal(size=len(clip)).astype(np.float32)
            scale = np.sqrt(np.mean(clip ** 2) / (10 ** (snr / 10) * np.mean(noise ** 2)))
            values.append(features_from_audio(clip + noise * scale, config))
        matrix_ = np.stack(values)
        predictions = bundle["model"].predict(matrix_)
        scores = predict_scores(bundle["model"], matrix_)
        results[f"white_noise_{snr}_db"] = compute_metrics(labels, predictions, scores)
        print(f"{snr} dB noise: accuracy={results[f'white_noise_{snr}_db']['accuracy']:.3f}", flush=True)
    write_json(args.run / "robustness_results.json", {"model": name, "noise": "seeded Gaussian white noise, not recorded environmental noise", "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--manifest", type=Path, required=True)
    train_parser.add_argument("--out", type=Path, required=True)
    train_parser.add_argument("--seed", type=int, default=475)
    train_parser.add_argument("--cache", type=Path, help="Optional feature cache outside the run directory")
    for command in ("evaluate", "robustness"):
        child = subparsers.add_parser(command)
        child.add_argument("--manifest", type=Path, required=True)
        child.add_argument("--run", type=Path, required=True)
    prediction_parser = subparsers.add_parser("predict")
    prediction_parser.add_argument("--model", type=Path, required=True)
    prediction_parser.add_argument("--audio", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
    with threadpool_limits(limits=1):
        {"train": train, "evaluate": evaluate, "predict": predict, "robustness": robustness}[args.command](args)


if __name__ == "__main__":
    main()
