# VoxGuard

VoxGuard is a research prototype for distinguishing genuine speech from synthetic or voice-converted speech. It includes an audio feature pipeline, three classical classifiers, noise-augmented MFCC/LFCC models, a PyTorch spectrogram CNN, reproducible evaluations, and single-file acoustic-model inference.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Abelar-Ai/voxguard/blob/main/notebooks/VoxGuard_Colab.ipynb)

## Latest improvement

After development-only selection, the augmented MFCC/LFCC SVM improved accuracy on a fresh 1,000-recording held-out sample:

| Condition | Original SVM | Improved SVM |
|---|---:|---:|
| Clean | 86.4% | 89.4% |
| White noise at 20 dB SNR | 50.0% | 83.8% |
| White noise at 10 dB SNR | 50.0% | 76.7% |

The comparison uses the same evaluation recordings and the same source training recordings. The original 400 evaluation recordings are excluded. These are balanced-subset results, with remaining attack-specific failures; see [IMPROVEMENTS.md](IMPROVEMENTS.md) for protocol, limitations, and all changes. The CNN is implemented and locally smoke-tested; a complete GPU training experiment is pending.

## Run in Google Colab

Open the notebook above, choose a GPU runtime, and run the cells in order. Default counts are 6,000 training, 1,200 development, and 2,000 evaluation recordings. The notebook excludes all 1,400 locally inspected evaluation recordings, retrains a comparable MFCC baseline, trains augmented acoustic models, and trains the CNN. Results and epoch checkpoints are saved to Google Drive. It exports a summary and complete result ZIP.

For a quick functional run, reduce counts to 1,200 / 400 / 400 and CNN epochs to 3. GPU availability and speed depend on your Colab runtime. Smaller runs do not establish CNN performance. For local neural development, install `requirements-neural.txt`.

To use the improved acoustic model immediately:

```bash
python -m voxguard.cli predict --model runs/robust_v1/selected.joblib --audio /absolute/path/to/recording.wav
```

Bundled scikit-learn model files were saved with the versions in `requirements-lock.txt`. The Colab notebook retrains models in its current environment rather than relying on cross-version pickle compatibility.

## First experiment

The first run used 2,000 real recordings sampled from the official ASVspoof 2019 Logical Access archive: 1,200 training, 400 development, and 400 evaluation recordings. Each subset is balanced by class. Original partitions are retained, and the loader checks for overlapping speaker IDs, utterance IDs, and audio hashes. The development set selects model settings; evaluation data does not choose settings or thresholds.

The selected SVM achieved **87.0% accuracy**, **0.9303 ROC-AUC**, and **14.5% interpolated EER** on the clean evaluation subset. Added Gaussian white noise reduced its accuracy to **50.0%** at both 20 dB and 10 dB SNR. These are preliminary results on a small, deliberately balanced subset, not full-benchmark or deployment results.

See [the experiment report](EXPERIMENT_RESULTS.md) for the comparison, error analysis, and completed work. Machine-readable metrics, per-recording predictions, confusion matrices, model-selection settings, and trained models are under `runs/pilot/`.

## Setup

Python 3.12 was used. From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
```

The lock file records exact versions from the completed run. `requirements.txt` lists the direct dependencies if a different supported Python/platform needs a fresh resolution.

## Predict an audio file

The trained SVM is included. It accepts formats supported by SoundFile, including WAV and FLAC.

```bash
python -m voxguard.cli predict --model runs/pilot/svm_rbf.joblib --audio /absolute/path/to/recording.wav
```

Output includes the label, synthetic score, model name, and elapsed time. The score is not calibrated confidence. Audio is converted to mono and 16 kHz, peak-normalized, and limited to its first four seconds. Files shorter than 0.1 seconds and silent files are rejected. A successful prediction is a model output, not proof of authenticity.

## Reproduce the pilot

Download the selected recordings directly from the official University of Edinburgh archive. The downloader uses byte ranges and ZIP integrity checks, rather than downloading the entire archive. The completed download here occupied approximately 126 MB including filesystem overhead; the full archive is approximately 7.12 GiB. Internet access is needed for this step.

```bash
python scripts/download_subset.py --out data --train 1200 --dev 400 --eval 400 --seed 475
python -m voxguard.cli train --manifest data/manifest.csv --out runs/reproduced --seed 475
python -m voxguard.cli evaluate --manifest data/manifest.csv --run runs/reproduced
python -m voxguard.cli robustness --manifest data/manifest.csv --run runs/reproduced
```

Use a new run directory when retraining. Training refuses to overwrite a frozen model selection. The commands cache features and save all candidate development scores. SVM and Logistic Regression scaling is fit exclusively on training features. Hyperparameters are fixed before evaluation. The original recordings are retained in the local workspace separately and are not redistributed in the code ZIP.

`runs/pilot/subset_manifest.csv` records the exact source member, label, attack, speaker, and SHA-256 of every pilot recording. Its audio paths are relative to the downloaded dataset directory. To repeat training, use the manifest created by the downloader in your data directory. The original run's raw-manifest hash reflects its original local file locations; fresh reproduction writes its own audit hash.

## Features and models

- First four seconds or the complete recording if shorter; no duration padding.
- 20 MFCCs from 64 Mel bands, 512-point FFT, 400-sample window, and 160-sample hop at 16 kHz.
- Mean and standard deviation of MFCCs and first temporal differences: 80 features per recording.
- Logistic Regression: C in {0.1, 1, 10}; StandardScaler; maximum 2,000 iterations.
- RBF SVM: C in {0.1, 1, 10}; gamma=scale; StandardScaler.
- Random Forest: 200 trees; maximum depth in {8, unlimited}.
- DummyClassifier most-frequent reference. Balanced training produces a tie resolved as the genuine class.
- Selection uses development balanced accuracy. Default model thresholds are retained.

## Tests

The current suite has 29 passing tests when PyTorch is installed. It covers sample-rate conversion, stereo input, feature stability, four-second cropping, short and silent input rejection, EER edge cases, false-alarm orientation, partition leakage, class coverage, model serialization, training-only scaler fitting, reproducible balanced sampling, MFCC/LFCC dimensions, exact noise SNR, threshold orientation, exclusion of inspected evaluation data, and CNN forward/backward behavior. Neural tests are skipped if the optional PyTorch dependency is absent.

Real-data checks also exercised training, held-out comparison, noise evaluation, saved-model inference, and a separate 78-recording downloader check. CNN training completed a one-epoch CPU smoke run and a checkpoint-resume check through epoch two, using only 26 training and 26 development recordings. These small checks validate the workflow and do not establish neural detection performance. Reports are in `runs/validation/`.

Synthetic tone fixtures in software tests check code behavior only. They are not used for the reported speech-detection performance.

`docs/github-actions-tests.yml.example` is an optional automated-test workflow template. It is not active: the GitHub connection used to publish this repository did not have workflow-write permission. The software tests were run locally.

## Remaining project work

The LFCC comparison and noise-augmentation experiment are implemented and measured. The CNN code is implemented but its substantive GPU experiment remains to be run. Recorded background noise, compression tests, independent dataset evaluation, and the FastAPI/React interface remain future work. All inspected evaluation IDs must stay out of future tuning and fresh evaluation; the notebook uses `protocols/inspected_evaluation.csv` for this purpose.

## Dataset attribution

ASVspoof 2019 database: https://doi.org/10.7488/ds/2555

M. Todisco et al., “ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection,” Interspeech 2019, pp. 1008–1012, doi:10.21437/Interspeech.2019-2249.

The dataset is provided under the Open Data Commons Attribution License (ODC-By). The downloader records dataset provenance and this citation. Source archive: University of Edinburgh DataShare. Model performance here is our pilot result, not a result reported by the dataset authors.
