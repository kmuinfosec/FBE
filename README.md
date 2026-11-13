# Encrypted Traffic Classification Beyond the Handshake

This repository contains the reproducible experimental code and data for First-Burst Exclusion (FBE), the input rule of the paper currently under review at IEEE Network.

## 📋 Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Repository Layout](#repository-layout)
- [Troubleshooting](#troubleshooting)

## Overview

Early classifiers of encrypted traffic identify the service behind a TLS flow from the signed payload lengths of its first n packets. Their most informative packets are the handshake: the ClientHello length carries the server name, and the server's first flight carries the certificate chain. Encrypted Client Hello (ECH) and post-quantum (PQC) key exchange change exactly these packets, so a classifier trained on today's handshakes fails when they change.

FBE removes the first client burst and the first server burst, so that the input starts where the handshake ends. The code compares five inputs, each trained on unchanged flows and tested under handshake changes:

- **CH included**: the first n packets from the ClientHello
- **CH burst removed**: the same window without the first client burst
- **Drop-4** / **Drop-8**: the window without its first 4 or 8 packets
- **FBE**: the window without the first client burst and the first server burst

Test conditions:

- **Unchanged**: the test flows as captured
- **CH length replaced**: every ClientHello length replaced by its most common value in the training flows, taken separately for each number of ClientHello packets, so that the length no longer carries the server name, as with ECH
- **Emulated PQC**: the X25519MLKEM768 key-share sizes added to the ClientHello (1,220 bytes) and to the server's first flight (1,088 bytes) of the CipherSpectrum Firefox flows; each enlarged flight is re-split into packets no larger than its largest packet before the change (1,448 bytes for a single packet), and everything after the handshake is left unchanged
- **Real ECH**: classifiers trained on the non-ECH flows of ECHPair are tested on its ECH flows

Two classifiers are provided: the **1D-CNN**, the main classifier of the paper, and **LightGBM**, the additional classifier mentioned in Section IV. The metric is service macro-F1. Every input counts the budget n from the ClientHello, so the packets that it removes still count against its budget.

## Environment Setup

### 1. Python Version
- Python 3.12

### 2. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains the following packages:

```
numpy==2.5.2
pyarrow==25.0.1
scikit-learn==1.9.0
lightgbm==4.7.0
torch==2.14.1
```

**Note**: `torch` is needed for the 1D-CNN, the main classifier; LightGBM runs without it. The 1D-CNN runs on a GPU when one is available and on the CPU otherwise (`device` in `config.ini`). The 1D-CNN numbers in the paper come from CPU runs, which are deterministic; on a GPU, floating-point nondeterminism changes them by up to about 0.03 macro-F1 on the smallest test set (CipherSpectrum) and by less than 0.01 on the others, without changing the order of the inputs. LightGBM results are identical on every machine.

## Configuration

All settings are in `config.ini`; no environment variables are needed.

| Section | Contents |
|---|---|
| `[paths]` | data and output directories, locations of the raw files |
| `[flows]` | window width (71 packets), MSS used for re-segmentation |
| `[cesnet_tls22]`, `[cipherspectrum]`, `[kmu_tls]`, `[echpair]` | per-dataset sampling and split settings used by `preprocess` and by the splits |
| `[experiment]` | budgets `ns`, maximum budget per dataset, inputs, test conditions, threads and parallel jobs |
| `[emulation]` | added bytes of the PQC emulation (1,220 client, 1,088 server) |
| `[lightgbm]` | LightGBM hyperparameters |
| `[cnn]` | 1D-CNN architecture and training settings; `device = auto`, `cpu` or `cuda` |

## Data Preparation

### 1. Provided Data

`data/preprocessed/` contains the preprocessed arrays of all four datasets of the paper, so every experiment runs without preprocessing (`SHA256SUMS` lists the checksums):

| File | Dataset | Services | Flows (train / test) | Split |
|---|---|---|---|---|
| `kmu_tls.npz` | KMU-TLS (our Chrome captures, TLS 1.3) | 129 | 256,585 / 256,565 | March–May 2026 train, June–August test |
| `cipherspectrum.npz` | CipherSpectrum (Firefox) | 37 | 11,730 / 8,418 | sessions before 2024-02-20 train, after test |
| `cesnet_tls22.npz` | CESNET-TLS22 (services with at least 2,000 flows in both weeks, 2,000 drawn per week) | 185 | 370,000 / 370,000 | week 40 train, week 41 test |
| `echpair.npz` | ECHPair (our Chrome captures, ECH and non-ECH, July–October 2026) | 7 | 10,364 non-ECH / 15,513 ECH | split by capture session; for each of ten seeds, 100 non-ECH flows per service train, other sessions test |

Each file holds `P` (signed payload lengths of the first 71 packets with payload, starting at the ClientHello, client to server negative), `c0` and `c1` (packets in the first client and server bursts), `y`, `split`, `kind` (0 non-ECH, 1 real ECH) and `services`. Details in `data/README.md`.

The emulated test conditions are not stored; they are generated in memory from the test flows at run time. Training always uses unchanged flows.

### 2. Raw Data

`data/raw/` holds the inputs of `preprocess` for the two datasets collected by the authors (`kmu_tls/flows.npz`, `echpair/hosts.parquet`, `echpair/probe.parquet`). The public datasets are downloaded by the user into `data/raw/<dataset>/` only if preprocessing is to be rerun:

| Dataset | Files | Source |
|---|---|---|
| `cipherspectrum` | `train_flow.parquet`, `valid_flow.parquet`, `test_flow.parquet` | CipherSpectrum release (Wickramasinghe et al., IEEE S&P 2025) |
| `cesnet_tls22` | `cesnet-tls22.zip` | Zenodo 10610895 |

The raw packet captures behind KMU-TLS and ECHPair are available on request from the first author (forever2331@kookmin.ac.kr).

### 3. Data Preprocessing

```bash
python main.py preprocess <dataset>     # kmu_tls | echpair | cipherspectrum | cesnet_tls22
```

This writes `data/preprocessed/<dataset>.npz`. For `kmu_tls` and `echpair` the output is identical to the distributed file.

## Usage

### 1. Single Run

One run trains one model on one dataset with one input and sweeps the budget n; the last argument is the seed (0, or 0 to 9 for `echpair`):

```bash
python main.py run <lgbm|cnn> <dataset> <input> <seed>

python main.py run cnn cipherspectrum fbe 0
python main.py run cnn echpair ch_included 3
python main.py run lgbm kmu_tls drop8 0
```

Inputs: `ch_included`, `ch_burst_removed`, `drop4`, `drop8`, `fbe`. Each run evaluates every test condition of its dataset (`unchanged` and `ch_length_replaced` everywhere, `emulated_pqc` on CipherSpectrum, `real_ech` on ECHPair). Runs resume from the result file.

Results are saved in `outputs/results/<model>/<dataset>/<input>_<seed>.json` (service macro-F1 for every budget and condition).

### 2. Summary

```bash
python main.py summary <lgbm|cnn>
```

This writes `outputs/summary/table_<model>.csv` (macro-F1 at n = 10 and 30, mean and standard deviation over seeds) and `outputs/summary/curves_<model>_<dataset>_<condition>.csv` (every budget). The outputs correspond to the paper as follows:

| Paper | Output |
|---|---|
| Table III | `table_cnn.csv` |
| Fig. 1(c) | `table_cnn.csv`, n = 10, CipherSpectrum and ECHPair |
| Fig. 2(a) | `curves_cnn_cipherspectrum_emulated_pqc.csv` (dotted line: `curves_cnn_cipherspectrum_unchanged.csv`) |
| Fig. 2(b) | `curves_cnn_echpair_real_ech.csv` (dotted line: `curves_cnn_echpair_unchanged.csv`) |
| LightGBM results of Section IV | `table_lgbm.csv`, `curves_lgbm_*.csv` |

### 3. Full Reproduction

```bash
./reproduce.sh cnn 5      # all 1D-CNN runs of the paper, 5 in parallel, then the summary
./reproduce.sh lgbm 5     # the same for LightGBM
```

Each set has 65 runs: five inputs on each of the three single-seed datasets (KMU-TLS, CipherSpectrum, CESNET-TLS22), and five inputs × ten seeds on ECHPair. For LightGBM on a 32-core machine, the set is dominated by CESNET-TLS22 and KMU-TLS, while CipherSpectrum and ECHPair finish in minutes. The 1D-CNN set is much faster on a GPU than on the CPU.

## Repository Layout

```
main.py       preprocess | run | summary
reproduce.sh  all runs of the paper for one model
config.ini    all settings
utils/        config helpers, preprocessing (raw data to arrays), experiment (splits, test conditions, inputs, training sweep), summary
models/       LightGBM, 1D-CNN
data/         raw inputs and preprocessed arrays (see data/README.md)
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`
- `torch` is required only for `cnn`; LightGBM runs without it

### Out of Memory
- Lower `n_jobs` and `threads` in `[experiment]`, or the parallelism argument of `reproduce.sh`
- For the 1D-CNN, lower `large_batch` and `eval_batch` in `[cnn]`, or set `device = cpu`

### Preprocessing Errors
- Verify the raw file locations in `[paths]`
- The public datasets must be downloaded first; the distributed arrays in `data/preprocessed/` do not need them

### Results Differ from the Paper
- Check the checksums: `cd data/preprocessed && sha256sum -c SHA256SUMS`
- Delete stale result files in `outputs/results/`; runs resume from existing files
- 1D-CNN on a GPU: see the note in [Environment Setup](#environment-setup)
