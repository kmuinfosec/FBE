# Data

```
data/
  preprocessed/   one <dataset>.npz per dataset, ready for `python main.py run`
  raw/            inputs of `python main.py preprocess <dataset>`
```

## preprocessed/

All four datasets of the article are included, so every experiment runs without preprocessing. The files hold unchanged flows only (plus the real ECH flows of `echpair`, `kind = 1`); the `ch_length_replaced` and `emulated_pqc` test conditions are generated from the test flows at run time by `utils/experiment.py`, not stored. `SHA256SUMS` lists the checksums.

| file | content | split |
|---|---|---|
| `kmu_tls.npz` | KMU-TLS, our Chrome captures: 129 services, 513,150 TLS 1.3 flows | March to May 2026 train, June to August test |
| `echpair.npz` | ECHPair, our Chrome captures from July to October 2026: ECH and non-ECH flows to seven Cloudflare-hosted services, 25,877 flows | per seed, 100 non-ECH flows per service for training, other capture sessions for test |
| `cipherspectrum.npz` | CipherSpectrum (Wickramasinghe et al., IEEE S&P 2025), Firefox flows to the 37 domains with at least 50 flows in each period, 20,148 flows | sessions before 2024-02-20 train, after test |
| `cesnet_tls22.npz` | CESNET-TLS22 (Zenodo 10610895), the 185 services with at least 2,000 flows in both weeks, 740,000 flows | week 40 train, week 41 test, 2,000 flows per service and period |

Arrays in every file:

| key | shape | meaning |
|---|---|---|
| `P` | `[N, 71]` int16 | signed payload lengths of the packets with payload, client to server negative, starting at the ClientHello |
| `c0`, `c1` | `[N]` | packets in the first client burst and in the first server burst |
| `y` | `[N]` | service index into `services` |
| `kind` | `[N]` | 0 non-ECH, 1 real ECH |
| `split` | `[N]` | 0 train / 2 test (ECHPair is split by capture session at run time) |
| `services` | `[K]` | class names |

`echpair.npz` also stores `pcap`, `t0`, `month`, `host_group` and `source` for the session split.

## raw/

Inputs of `preprocess`. The two datasets collected by the authors are included; the public datasets are downloaded by the user.

| dataset | files in `raw/<dataset>/` | note |
|---|---|---|
| `kmu_tls` | `flows.npz` (included): `X` packet lengths from the ClientHello, `y`, `split`, `tls_version` (3 = TLS 1.3, 2 = TLS 1.2, 0 = unknown, read from the ServerHello of the capture), `services`, `capture_month` | the 598,561 flows drawn from the full capture (150 services, 2,000 flows per service and period, seed 0); `preprocess` keeps the TLS 1.3 flows of the services with at least 800 of them in each period |
| `echpair` | `hosts.parquet`, `probe.parquet` (included) | all flows of the seven services with their capture session |
| `cipherspectrum` | `train_flow.parquet`, `valid_flow.parquet`, `test_flow.parquet` | from the CipherSpectrum release |
| `cesnet_tls22` | `cesnet-tls22.zip` | from Zenodo 10610895 |

`python main.py preprocess kmu_tls` and `python main.py preprocess echpair` reproduce the distributed arrays exactly. The raw packet captures behind `kmu_tls` and `echpair` are available on request from the first author (forever2331@kookmin.ac.kr).
