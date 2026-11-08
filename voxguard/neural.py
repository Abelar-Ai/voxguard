"""Small log-Mel CNN with training-only normalization and noise augmentation."""
import argparse
import hashlib
import json
from pathlib import Path

import librosa
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from threadpoolctl import threadpool_limits

from .cli import write_json
from .data import read_manifest
from .features import load_audio
from .metrics import compute_metrics
from .robust_features import add_white_noise

MAX_FRAMES = 400


def spectrogram(audio):
    power = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=512,
                                           win_length=400, hop_length=160,
                                           n_mels=64, center=False, power=2)
    value = librosa.power_to_db(power, ref=1.0, top_db=80).astype(np.float32)
    frames = min(value.shape[1], MAX_FRAMES)
    padded = np.full((64, MAX_FRAMES), -80.0, dtype=np.float32)
    padded[:, :frames] = value[:, :frames]
    return padded, frames


class SpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for inputs, outputs in ((1, 16), (16, 32), (32, 64)):
            blocks.extend([nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs),
                           nn.ReLU(), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, 1))

    def forward(self, inputs, lengths):
        values = self.features(inputs)
        valid = torch.clamp(lengths // 8, min=1, max=values.shape[-1])
        mask = torch.arange(values.shape[-1], device=values.device)[None, :] < valid[:, None]
        pooled = (values * mask[:, None, None, :]).sum(dim=(2, 3)) / (valid[:, None] * values.shape[2])
        return self.classifier(pooled).squeeze(1)


def prepare(rows, cache, seed, training, views):
    cache.mkdir(parents=True, exist_ok=True)
    entries, sums, squares, count = [], np.zeros(64), np.zeros(64), 0
    for index, row in enumerate(rows):
        key = f"spec-v1_{row['sha256']}_{seed}_{'train' if training else 'dev'}_{views}"
        path = cache / (hashlib.sha256(key.encode()).hexdigest() + ".npz")
        if not path.exists():
            clip = load_audio(row["path"])
            rng = np.random.default_rng(seed ^ int(row["sha256"][:8], 16))
            variants = [clip]
            if training:
                variants += [add_white_noise(clip, float(rng.uniform(8, 35)), rng) for _ in range(views)]
            else:
                variants += [add_white_noise(clip, snr, rng) for snr in (20, 10)]
            specs = []
            for variant in variants:
                # Match peak normalization in the acoustic-feature pipeline.
                value, frames = spectrogram(variant / np.max(np.abs(variant)))
                specs.append(value)
            np.savez_compressed(path, specs=np.stack(specs).astype(np.float16), frames=frames)
        with np.load(path, allow_pickle=False) as saved:
            specs = saved["specs"].astype(np.float32)
            frames = int(saved["frames"])
        for variant in range(len(specs)):
            entries.append((str(path), row["label"], variant))
        if training:
            valid = specs[:, :, :frames].astype(np.float64)
            sums += valid.sum(axis=(0, 2))
            squares += (valid ** 2).sum(axis=(0, 2))
            count += len(specs) * frames
        if (index + 1) % 100 == 0:
            print(f"Spectrogram cache: {index + 1}/{len(rows)}", flush=True)
    if training:
        mean = sums / count
        std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-4))
        return entries, mean.astype(np.float32), std.astype(np.float32)
    return entries


class SpecDataset(Dataset):
    def __init__(self, entries, mean, std, augment=False):
        self.entries, self.mean, self.std, self.augment = entries, mean[:, None], std[:, None], augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, label, variant = self.entries[index]
        with np.load(path, allow_pickle=False) as saved:
            value = saved["specs"][variant].astype(np.float32)
            frames = int(saved["frames"])
        value = (value - self.mean) / self.std
        value[:, frames:] = 0
        value = torch.from_numpy(value.copy())
        if self.augment:
            if torch.rand(()) < 0.5:
                start = int(torch.randint(0, 60, ()).item())
                value[start:start + 4] = 0
            if torch.rand(()) < 0.5 and frames > 20:
                start = int(torch.randint(0, frames - 19, ()).item())
                value[:, start:start + 20] = 0
        return value[None, :], torch.tensor(float(label)), torch.tensor(frames)


def device_for(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_loader(model, loader, device):
    model.eval()
    labels, scores = [], []
    with torch.inference_mode():
        for inputs, targets, lengths in loader:
            values = torch.sigmoid(model(inputs.to(device), lengths.to(device)))
            labels.extend(targets.numpy().tolist())
            scores.extend(values.cpu().numpy().tolist())
    return np.array(labels, dtype=int), np.array(scores)


def train(args):
    if (args.out / "frozen.json").exists():
        raise ValueError("CNN selection is frozen; use another run directory")
    if args.epochs < 1 or args.views < 1:
        raise ValueError("Epochs and augmented views must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = device_for(args.device)
    print(f"Training CNN on {device}", flush=True)
    rows = read_manifest(args.manifest)
    training = [r for r in rows if r["split"] == "train"]
    development = [r for r in rows if r["split"] == "dev"]
    train_entries, mean, std = prepare(training, args.cache, args.seed, True, args.views)
    dev_entries = prepare(development, args.cache, args.seed, False, args.views)
    train_loader = DataLoader(SpecDataset(train_entries, mean, std, True), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
    dev_loaders = {condition: DataLoader(SpecDataset([entry for entry in dev_entries if entry[2] == variant], mean, std),
                                         batch_size=args.batch_size, num_workers=args.workers)
                   for variant, condition in enumerate(("clean", "noise_20_db", "noise_10_db"))}
    model = SpectrogramCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    history, best, bad_epochs, start_epoch = [], -1.0, 0, 1
    metadata = {"mean": mean.tolist(), "std": std.tolist(), "seed": args.seed,
                "training_ids": [r["utterance_id"] for r in training],
                "training_speakers": sorted({r["speaker_id"] for r in training}),
                "training_sha256": [r["sha256"] for r in training],
                "dev_sha256": [r["sha256"] for r in development], "views": args.views,
                "learning_rate": args.lr, "threshold": 0.5, "name": "log_mel_cnn", "torch_version": str(torch.__version__)}
    if args.resume and (args.out / "last.pt").exists():
        state = torch.load(args.out / "last.pt", map_location=device, weights_only=True)
        if any(state["metadata"].get(key) != metadata[key]
               for key in ("training_sha256", "dev_sha256", "views", "seed", "learning_rate")):
            raise ValueError("Cannot resume with different data or training settings")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best, bad_epochs, history = state["epoch"] + 1, state["best"], state["bad_epochs"], state["history"]
        torch.set_rng_state(state["rng"].cpu())
        if device.type == "cuda" and "cuda_rng" in state:
            torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    elif (args.out / "last.pt").exists():
        raise ValueError("An interrupted run exists; pass --resume or choose another directory")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        loss_sum, samples = 0.0, 0
        for inputs, targets, lengths in train_loader:
            inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs, lengths), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(targets)
            samples += len(targets)
        conditions = {}
        for condition, loader in dev_loaders.items():
            labels, scores = score_loader(model, loader, device)
            conditions[condition] = compute_metrics(labels, scores >= 0.5, scores)
        score = float(np.mean([metric["balanced_accuracy"] for metric in conditions.values()]))
        history.append({"epoch": epoch, "loss": loss_sum / samples,
                        "mean_development_balanced_accuracy": score, "conditions": conditions})
        if score > best:
            best, bad_epochs = score, 0
            torch.save({"model": model.state_dict(), "metadata": metadata, "epoch": epoch}, args.out / "best.pt")
        else:
            bad_epochs += 1
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "metadata": metadata,
                 "epoch": epoch, "best": best, "bad_epochs": bad_epochs, "history": history, "rng": torch.get_rng_state()}
        if device.type == "cuda":
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        torch.save(state, args.out / "last.pt")
        write_json(args.out / "history.json", history)
        print(f"Epoch {epoch}: loss={loss_sum / samples:.4f}, mean dev BA={score:.3f}, clean={conditions['clean']['accuracy']:.3f}", flush=True)
        if bad_epochs >= args.patience:
            break
    checkpoint = torch.load(args.out / "best.pt", map_location="cpu", weights_only=True)
    write_json(args.out / "frozen.json", {"best_epoch": checkpoint["epoch"], "mean_development_balanced_accuracy": best,
               "train_recordings": len(training), "dev_recordings": len(development),
               "selection": "mean development balanced accuracy across clean, 20 dB and 10 dB noise",
               "normalization": "per-Mel-bin training mean and standard deviation; padding excluded",
               "augmentation": {"white_noise_views": args.views, "snr_uniform_db": [8, 35], "training_only_spec_masks": True},
               "evaluation_used": False, "seed": args.seed})


def infer(checkpoint_path, audio, device="auto", batch_size=32):
    device = device_for(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SpectrogramCNN().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    metadata = checkpoint["metadata"]
    mean, std = np.array(metadata["mean"])[:, None], np.array(metadata["std"])[:, None]
    scores = []
    with torch.inference_mode():
        for offset in range(0, len(audio), batch_size):
            values, lengths = [], []
            for clip in audio[offset:offset + batch_size]:
                value, frames = spectrogram(clip / np.max(np.abs(clip)))
                value = (value - mean) / std
                value[:, frames:] = 0
                values.append(value)
                lengths.append(frames)
            inputs = torch.tensor(np.stack(values), dtype=torch.float32, device=device)[:, None]
            length_tensor = torch.tensor(lengths, device=device)
            scores.extend(torch.sigmoid(model(inputs, length_tensor)).cpu().tolist())
    return np.array(scores), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("cache/spectrograms"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=475)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=1):
        train(args)


if __name__ == "__main__":
    main()
