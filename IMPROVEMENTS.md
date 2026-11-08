# VoxGuard robustness improvements

Experiment date: November 8, 2026. The original pilot is documented separately in `EXPERIMENT_RESULTS.md`.

## What changed

The original clean-trained MFCC classifier failed under the white-noise stress test. This experiment adds 40 linear-frequency cepstral coefficients (LFCCs) to the original MFCC statistics and trains on clean plus noise-augmented versions of the same 1,200 training recordings. The combined representation has 240 features. Each training recording contributes two seeded Gaussian-white-noise variants, with signal-to-noise ratios drawn uniformly from 8 to 35 dB, for 3,600 training views in total.

Fifteen candidates were compared on 400 development recordings under clean, 20 dB white-noise, and 10 dB white-noise conditions. Candidates included MFCC, LFCC, and combined representations with Logistic Regression, RBF SVM, and Random Forest classifiers. Selection used mean development balanced accuracy across the three conditions. One global decision threshold was chosen from development scores. Evaluation data did not select features, hyperparameters, or the threshold.

The selected model uses combined MFCC/LFCC features and an RBF SVM with C=1 and gamma=scale. Its frozen synthetic-score threshold is -0.029748856495285136. It reached 85.75% mean development balanced accuracy across the three conditions. All candidate development metrics and frozen settings are in `runs/robust_v1/`.

Our LFCC implementation uses 128 triangular linear-frequency bands, a 1,024-point FFT, a 400-sample analysis window, a 160-sample hop, and the first 40 cepstral coefficients. Mean and standard deviation of coefficients and first temporal differences produce 160 LFCC statistics. This is our implementation, not an exact reproduction of the official LFCC-GMM or LFCC-LCNN baseline. The official challenge baselines motivate this representation: https://github.com/asvspoof-challenge/2021.

## Fair comparison on fresh evaluation recordings

After selection was frozen, a new balanced sample of 1,000 evaluation recordings was downloaded: 500 genuine and 500 synthetic, approximately balanced across A07–A19. All 400 previously inspected evaluation utterances were excluded. The evaluator checks both IDs and hashes against the earlier sample and checks IDs, speakers, and hashes against each model's training data.

Both models use exactly the same source training recordings. The augmented model has additional transformed views of those recordings. Both models are scored on identical evaluation recordings and identical seeded noise realizations. Original model parameters, augmented model parameters, and both thresholds are fixed before scoring. The new evaluation seed is 1475.

| Condition | Original MFCC SVM accuracy | Augmented MFCC/LFCC SVM accuracy | Difference |
|---|---:|---:|---:|
| Clean | 86.40% | 89.40% | +3.00 percentage points |
| White noise at 20 dB SNR | 50.00% | 83.80% | +33.80 percentage points |
| White noise at 10 dB SNR | 50.00% | 76.70% | +26.70 percentage points |

| Condition | Original ROC-AUC | Improved ROC-AUC | Original EER | Improved EER |
|---|---:|---:|---:|---:|
| Clean | 0.9258 | 0.9552 | 14.0% | 10.8% |
| White noise at 20 dB SNR | 0.4938 | 0.9138 | 51.6% | 16.0% |
| White noise at 10 dB SNR | 0.5853 | 0.8373 | 43.4% | 23.6% |

On clean audio, the improved model reduced genuine false alarms from 55/500 to 31/500. It missed 75/500 synthetic examples, compared with 81/500 for the original model. Under 20 dB noise, it falsely flagged 78/500 genuine examples and missed 84/500 synthetic examples. These results demonstrate a useful gain on this predefined pilot comparison, with remaining errors.

The change is not uniformly better for every attack. On clean recordings, A19 recall improved from 14/38 to 38/38, while A17 recall fell from 10/38 to 5/38 and A18 recall fell from 16/38 to 10/38. These small attack-level counts identify limitations; pooled accuracy must not hide them. All per-attack results and per-recording predictions are in `runs/fresh_v1/`.

The original pilot's 87.0% accuracy used a different 400-recording sample. The fair improvement comparison above therefore uses 86.4%, the original model's result on the same new 1,000 recordings.

## CNN and Colab

`notebooks/VoxGuard_Colab.ipynb` provides a larger experiment with 6,000 training, 1,200 development, and 2,000 evaluation recordings by default. It excludes all 1,400 locally inspected evaluation recordings. The notebook retrains the MFCC baseline in Colab, trains the augmented combined-feature classifiers, and trains a small log-Mel CNN using the same source partitions. It keeps data and caches on Colab's local disk and saves model checkpoints and reports to Drive.

The CNN uses three convolution blocks, group normalization, 64 Mel bands, padding-aware temporal pooling, and a binary output. Training includes noise-augmented views and frequency/time masks. Normalization statistics use training spectrograms only, excluding padding. Best epoch selection and early stopping use development data only. Interrupted training can resume from an epoch checkpoint with matching data and training settings.

The CNN has passed local forward/backward and spectrogram tests and a one-epoch CPU training check. It has not completed a substantive GPU experiment, and no CNN benchmark accuracy is claimed. Its purpose is to test whether a learned time-frequency representation improves on the acoustic baseline.

## Limits

These are class-balanced, attack-balanced subset results with one evaluation seed, not full-ASVspoof challenge results. No confidence intervals or statistical significance tests have been performed. Gaussian white noise is not representative of every environment. Compression, real background noise, multilingual speech, newer generation systems, and cross-dataset transfer remain untested. The experiment changes feature representation, augmentation, model selection objective, and threshold selection together; the gains cannot be attributed to any one change without ablation experiments.

The new 1,000-recording evaluation sample is now inspected. It must be excluded from later tuning and from a claimed fresh evaluation. `protocols/inspected_evaluation.csv` combines it with the original 400 IDs for future runs.

## Reproduce

Follow the quick-start notebook or run the robust trainer on a prepared manifest:

```bash
python -m voxguard.robust_train --manifest data/manifest.csv --out runs/new_robust --views 2
```

Use `scripts/download_subset.py --exclude-manifest protocols/inspected_evaluation.csv` to reserve fresh evaluation data. Then call `python -m voxguard.fresh_benchmark` with the frozen model paths. The notebook contains complete ordered commands, exports predictions and metrics, and prepares an updated exclusion list for subsequent experiments.

Dataset attribution: ASVspoof 2019, University of Edinburgh DataShare, https://doi.org/10.7488/ds/2555, Open Data Commons Attribution License. M. Todisco et al., Interspeech 2019, doi:10.21437/Interspeech.2019-2249.
