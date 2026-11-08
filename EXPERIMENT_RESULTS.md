# VoxGuard preliminary experiment results

Run date: November 8, 2026. Course: CSCI 475. Random seed: 475.

This report records the first baseline experiment. Subsequent LFCC, noise-augmentation, and CNN implementation work is documented in [IMPROVEMENTS.md](IMPROVEMENTS.md); the implementation status below describes the original pilot stage.

## Completed components

The current implementation downloads and verifies an ASVspoof 2019 LA subset, preserves official partitions, checks for data overlap, preprocesses audio, extracts acoustic features, trains and compares three classifiers, saves trained models, classifies individual audio files, and exports evaluation metrics and confusion matrices. Eighteen software tests passed. The CNN and web interface are not implemented.

## Dataset and protocol

The pilot contains 1,200 training recordings, 400 development recordings, and 400 evaluation recordings, sampled directly from the official archive. Every partition has equal genuine and synthetic counts. Synthetic examples are distributed approximately equally across available attack IDs within each partition. The sampled training, development, and evaluation partitions contain 20, 20, and 67 distinct speaker IDs, respectively; the integrity checks found no overlap between partitions. These speaker counts are observed in the downloaded sample.

Training has 100 synthetic examples from each of A01–A06. Development has 33 or 34 examples per training attack. Evaluation has 15 or 16 examples from each of A07–A19. An evaluation attack ID does not by itself guarantee an entirely unfamiliar generation method; the official evaluation set includes both known and unknown systems. No claim of universal detection of unseen generators follows from these results.

Recordings are reduced to mono, resampled to 16 kHz if needed, peak-normalized, and truncated to their first four seconds if longer. Twenty MFCCs and their first temporal differences are summarized by mean and standard deviation, producing 80 features. Shorter valid clips use their available frames without duration padding.

The feature scalers are fit only on training data. Candidate hyperparameters are compared using development balanced accuracy. The winning setting within each family and the overall model choice are frozen before running evaluation. Classifier thresholds remain at their defaults. Evaluation EER is interpolated from the ROC crossing and is a descriptive metric, not a threshold chosen for deployment.

## Development comparison

| Model | Selected setting | Development accuracy |
|---|---|---:|
| Trivial most-frequent reference | Genuine class after balanced-training tie | 50.00% |
| Logistic Regression | C = 0.1 | 82.50% |
| RBF SVM | C = 1, gamma = scale | 89.75% |
| Random Forest | 200 trees, unlimited depth | 87.75% |

The SVM was selected for the prototype using development data. Because these subsets are balanced, accuracy equals balanced accuracy.

## Clean held-out evaluation

| Model | Accuracy | Synthetic F1 | ROC-AUC | EER |
|---|---:|---:|---:|---:|
| Trivial reference | 50.00% | 0.0000 | 0.5000 | 50.0% |
| Logistic Regression | 84.50% | 0.8377 | 0.8908 | 18.5% |
| RBF SVM | 87.00% | 0.8667 | 0.9303 | 14.5% |
| Random Forest | 83.25% | 0.8251 | 0.8964 | 18.5% |

The SVM correctly classified 348 of 400 recordings. It falsely labeled 21 of 200 genuine recordings synthetic (10.5%) and missed 31 of 200 synthetic recordings (15.5%). Synthetic precision was 88.95%, and synthetic recall was 84.50%.

| Actual class | Predicted genuine | Predicted synthetic |
|---|---:|---:|
| Genuine | 179 | 21 |
| Synthetic | 31 | 169 |

The SVM detected every sampled synthetic recording from A07–A15, but results were weaker on A16–A19. It detected 13/15 examples from A16, 5/15 from A17, 7/15 from A18, and 4/15 from A19. These small counts are diagnostic observations and do not support precise attack-level performance estimates.

## Noise robustness

The frozen SVM was evaluated on the same 400 held-out clips after adding seeded Gaussian white noise. Noise was scaled per clip to the requested signal-to-noise ratio before the standard feature normalization. This tests synthetic white noise; it does not represent all real background noise.

| Condition | Accuracy | Genuine false-alarm rate | ROC-AUC |
|---|---:|---:|---:|
| Clean | 87.00% | 10.5% | 0.9303 |
| White noise at 20 dB SNR | 50.00% | 100.0% | 0.4846 |
| White noise at 10 dB SNR | 50.00% | 100.0% | 0.5794 |

At both noise levels, the SVM predicted synthetic for every recording. This is a substantial failure of the current clean-trained MFCC baseline under the tested noise conditions. The experiment does not establish the cause; loss of useful acoustic cues and a shift in the feature distribution are hypotheses for later investigation. The higher AUC at 10 dB in this single seeded experiment should not be interpreted as evidence that stronger noise improves robustness.

## Software validation

The final run passed eighteen tests in 1.99 seconds. Coverage includes 8, 16, and 22.05 kHz input; mono and stereo input; deterministic finite features; cropping; short and silent recording rejection; EER for perfect, inverted, and tied scores; class orientation of false alarms; overlap rejection by speaker, utterance, and hash; both-class coverage; saved-model consistency; training-only scaling; reproducible balanced sampling; and rejection of invalid subset sizes. The machine-readable test log is included. Separate end-to-end commands completed real-data training, held-out evaluation, noise evaluation, and single-recording prediction with the saved SVM. A separate 78-recording download verified the final downloader, file integrity, and portable relative paths.

## Limits and next experiments

The subset was deliberately balanced by class and approximately balanced by synthetic attack. Its metrics are not full-ASVspoof results and do not estimate error rates under natural class prevalence. Only one sampling seed and one training seed were used, without confidence intervals or statistical significance testing. Recordings longer than four seconds are truncated. A small number of examples represents each evaluation attack. The project has not tested a modern external dataset, compression, a CNN, or uploaded microphone recordings.

Next steps are to expand data coverage, investigate LFCC or other representations, compare noise augmentation using development data, and collect results on a separately reserved experiment. The current evaluation data must not guide subsequent hyperparameter or threshold tuning. The written milestone can now report an implemented pipeline, three learned baselines, actual error analysis, and an observed robustness failure.

## Attribution

Dataset: ASVspoof 2019, University of Edinburgh DataShare, https://doi.org/10.7488/ds/2555, Open Data Commons Attribution License (ODC-By).

M. Todisco et al., “ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection,” Interspeech 2019, pp. 1008–1012, doi:10.21437/Interspeech.2019-2249.
