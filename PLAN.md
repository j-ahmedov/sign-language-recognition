# SignAdapt — Project Plan

**Personalized Federated Learning for Signer-Adaptive Sign Language Recognition**

Version 1.0 · 17 August 2026 · Target: working system + results + presentation by mid-October 2026

---

## 0. The one-sentence version

Sign language recognition models collapse on signers they weren't trained on; this project measures that gap, then tests whether a *shared encoder trained federatedly + a private per-user classifier head* closes it with fewer local examples than training alone — with raw video never leaving the device.

---

## 1. Research questions

| ID | Question | Answered by |
|---|---|---|
| **RQ1** | How large is the signer-independent generalization gap on a controlled isolated-sign benchmark? | E1 vs E2 |
| **RQ2** | How many labelled examples of their own signing does a new signer need before personalization recovers most of that gap? | E3–E5, swept over *k* |
| **RQ3** | Does a federatedly-trained shared encoder reduce the number of local examples needed, compared with a user training alone? | E3 vs E5 |
| **RQ4** | What is the on-device cost — inference latency, model size, per-round communication — of running this in a live video call? | Demo instrumentation |

RQ3 is the thesis. RQ1 is the motivation. RQ2 is the practical result someone would actually use. RQ4 is what makes it real rather than a simulation exercise.

**A negative answer to RQ3 is still a thesis.** If federated pretraining doesn't beat local-only training, the contribution becomes *"here is the crossover point — below N users and above k examples, federation isn't worth its complexity."* Design the experiments so that answer is publishable too, and you can't lose.

---

## 2. Scope — fixed, and deliberately narrow

**In scope**

- Isolated sign recognition over a bounded vocabulary (64 signs → optionally 226)
- Keypoint-based pipeline (no raw-video models)
- Simulated federated learning, one client per signer
- Personalization via a split encoder/head architecture
- Live webcam demo with captions injected into Zoom / Meet via virtual camera

**Explicitly out of scope** — write these into the proposal so they can't be moved later

- Continuous sign language *translation* into grammatical sentences
- Sign language *generation* / avatars
- Non-manual grammar (facial expression, mouthing) as a modelled linguistic channel — face landmarks are used as features, not interpreted
- Real multi-device federated deployment (simulated on one machine, which is standard practice)
- Any claim of interpreter replacement or clinical/production readiness

---

## 3. The limitation you have to handle openly

You cannot recruit Deaf participants. That is a real constraint, and the way to deal with it is to state it prominently rather than let a reviewer find it.

**How this changes the project:**

1. **Reposition the contribution as *methodological*, not as a product.** You are studying personalization and federated adaptation using sign language recognition as the task domain. You are not shipping an assistive tool. This is honest and it is defensible.
2. **Evaluate only on public datasets recorded with consent.** No self-recorded data used as evidence; your own recordings are for the live demo only, clearly labelled as illustrative.
3. **Ground design choices in published Deaf-authored and community-consultation literature**, and cite it. The 2025 global community-perspectives study gives you documented positions on accuracy expectations, interpreter replacement, and Deaf leadership — cite it in the motivation *and* in the limitations.
4. **Write a dedicated "Limitations and Ethical Considerations" section** that says plainly: no Deaf participants were involved; the system is not validated with its intended users; participatory validation with Deaf signers is required future work before any deployment claim.
5. **Do not use framing you can't support** — no "breaking the silence," no "giving Deaf people a voice," no accessibility-impact claims. Describe the technical result and stop there.

If a single Deaf contact ever does become possible — one interpreter, one online conversation, one email exchange with an association — even that materially improves the work. Keep the door open, but don't block on it.

---

## 4. System architecture

```
webcam / dataset video
        │
        ▼
MediaPipe Holistic Landmarker          pose(33) + hands(21+21) + face(subset ~40)
        │                              → per-frame landmark array
        ▼
normalization                          centre on mid-shoulder, scale by shoulder width,
        │                              optional handedness mirroring, z kept but de-weighted
        ▼
temporal resampling                    variable-length clip → fixed T = 64 frames
        │
        ▼
┌───────────────────────────┐
│  SHARED ENCODER           │          temporal transformer (4 layers, d=128)
│  ~0.5–2 M params          │          ← federated: aggregated across clients
└───────────────────────────┘
        │  128-d embedding
        ▼
┌───────────────────────────┐
│  PRIVATE HEAD             │          linear or prototypical classifier
│  128 × n_classes          │          ← never leaves the device
└───────────────────────────┘
        │
        ▼
predicted sign  →  caption renderer  →  pyvirtualcam  →  Zoom / Meet / Teams
```

**Why keypoints and not video:** it fits your 16 GB M4, it runs real-time, it makes the model small enough that federated communication cost is a footnote rather than a blocker — and it strips identity from the transmitted representation, which strengthens the privacy argument you're making.

---

## 5. Data

**Phase A — LSA64 (start here).** 64 Argentinian signs, 10 signers, 3,200 videos. Directly downloadable, small, clean. Its purpose is to get the whole pipeline working end to end in week 2 — including the federated loop — before any large dataset is involved. Do not skip this and go straight to the big dataset; the small dataset is what lets you debug in minutes instead of hours.

**Phase B — AUTSL (the real experiments).** 226 Turkish signs, 43 signers, ~38k video clips, with an *official signer-independent split* — which is exactly the structure this project needs. Requires registration via ChaLearn LAP. **Register in week 1**, because approval latency is the one thing you can't compress.

**Deliberately not used:** WLASL and MS-ASL (distributed as YouTube links, significant link rot), PHOENIX-2014T (documented train/test overlap that inflates reported scores, and it's a translation benchmark, not what you're doing).

**The single most dangerous bug in this project is signer leakage** — the same signer appearing in both train and test, which silently inflates every number and invalidates the entire thesis. Write a unit test that asserts the intersection of signer IDs across splits is empty, and run it in CI. Do this in week 2, not week 5.

---

## 6. Experiment matrix

Let `S_train` = training signers, `S_held` = held-out signers, `k` = labelled examples per class from the held-out signer.

| ID | Name | Setup | What it tells you |
|---|---|---|---|
| **E1** | Centralized, signer-dependent | All signers in train and test | Optimistic ceiling |
| **E2** | Centralized, signer-independent | Train on `S_train`, test on `S_held`, k=0 | **The gap (RQ1)** |
| **E3** | Local-only | Held-out signer trains from scratch on k samples | **Null hypothesis — the "you don't need FL" baseline** |
| **E4** | FedAvg + fine-tune | Federated across `S_train`, then fine-tune whole model on k samples | Standard FL baseline |
| **E5** | **FedPer (proposed)** | Federated encoder across `S_train`, private head trained on k samples | **The method (RQ3)** |
| **E6** | Centralized pretrain + head | Non-federated upper bound for E5 | Cost of federation |

**Sweep:** `k ∈ {0, 1, 2, 3, 5, 10, 20}` × 3 random seeds × leave-one-signer-out across all held-out signers.
**Report:** top-1 and top-5, mean ± std *across signers* (not just across seeds — inter-signer variance is a finding in itself).

**The money chart:** x-axis = k (local examples per sign), y-axis = top-1 accuracy, one line per method (E3, E4, E5), horizontal dashed lines for E1 (ceiling) and E2 (zero-shot gap). If E5 sits above E3 and reaches the E1 line at a lower k, RQ3 is answered positively and the whole thesis is on one slide.

---

## 7. Repository layout

```
signadapt/
├── README.md
├── pyproject.toml
├── Makefile                      # make data | train | federated | figures | demo
├── configs/
│   ├── data.yaml                 # dataset paths, landmark subsets, T, normalization
│   ├── model.yaml                # encoder dims, layers, dropout
│   └── fl.yaml                   # rounds, clients/round, local epochs, strategy
├── src/signadapt/
│   ├── data/
│   │   ├── download.py           # LSA64 fetch; AUTSL instructions + integrity check
│   │   ├── keypoints.py          # MediaPipe extraction → .npy cache
│   │   ├── normalize.py          # anchoring, scaling, handedness
│   │   └── dataset.py            # torch Dataset + signer-aware splits
│   ├── models/
│   │   ├── encoder.py            # shared temporal transformer
│   │   ├── head.py               # linear / prototypical head
│   │   └── model.py
│   ├── train/
│   │   ├── centralized.py        # E1, E2, E6
│   │   ├── local_only.py         # E3
│   │   └── evaluate.py
│   ├── federated/
│   │   ├── client.py             # Flower NumPyClient
│   │   ├── strategy.py           # FedAvg + FedPer (encoder-only aggregation)
│   │   └── simulation.py         # E4, E5
│   ├── personalize/
│   │   └── adapt.py              # k-shot adaptation sweep
│   ├── demo/
│   │   ├── realtime.py           # webcam → keypoints → prediction, with latency logging
│   │   └── virtualcam.py         # caption overlay → virtual camera
│   └── utils/                    # seeding, metrics, json logging
├── experiments/run_all.sh
├── results/                      # JSON, git-tracked — figures regenerate from these
├── figures/
└── tests/
    ├── test_splits.py            # ← signer leakage guard. Non-negotiable.
    ├── test_normalize.py
    └── test_fedper.py            # asserts head params are never aggregated
```

Results go to JSON and JSON is committed. Figures are generated *from* the JSON by a script, never hand-made. When your supervisor asks "can you redo that chart with error bars," it's one command, not one evening.

---

## 8. Six-week schedule

| Week | Deliverable | Definition of done |
|---|---|---|
| **1** | Setup + registration + reading | AUTSL registration submitted. Repo scaffolded, env reproducible. LSA64 downloaded. 15 papers in a bib file. One-page proposal sent to supervisor. |
| **2** | Data pipeline | Keypoints extracted and cached for all of LSA64. Normalization done. **`test_splits.py` passes.** Keypoint overlay video rendered for visual sanity check. |
| **3** | Centralized baselines | Model trains; overfits a 50-sample subset to ~100% (sanity gate). **E1 and E2 measured on LSA64 — you now have the gap number.** |
| **4** | Federated simulation | Flower simulation runs, 10 clients. FedAvg (E4) within a couple of points of centralized on an IID split — this is your correctness check. E3 local-only baseline measured. |
| **5** | Personalization | FedPer (E5) implemented, head-aggregation test passing. Full k-sweep run on LSA64. **Adaptation curve plotted.** AUTSL pipeline started if access granted. |
| **6** | Demo + presentation | Real-time webcam inference at ≥15 fps with measured latency. Virtual camera captions visible in a real Meet call. 12 slides. |

**Weekly discipline:** every Friday, commit a `results/*.json` and regenerate figures. If a week produces no new number, the schedule has slipped and you should cut scope rather than push the deadline.

**Cut order if you fall behind** (cut from the bottom): AUTSL → E6 → E4 → live demo polish. Never cut E2, E3, or E5 — those three *are* the thesis.

---

## 9. Cost

| Item | Cost |
|---|---|
| MediaPipe, PyTorch, Flower, OpenCV, pyvirtualcam, OBS | €0 (open source) |
| LSA64, AUTSL | €0 (free, registration required for AUTSL) |
| Compute | €0 — all on your M4; Colab free tier as overflow |
| Zoom / Meet integration | €0 — virtual camera needs no SDK, account, or review |
| **Total** | **≈ €0** |

---

## 10. Risk register

| Risk | Mitigation |
|---|---|
| **Signer leakage inflates results** | `test_splits.py` in CI from week 2. Assert empty signer intersection across every split. |
| AUTSL access delayed or refused | LSA64 alone is sufficient for a complete thesis with 10 clients. AUTSL is an upgrade, not a dependency. |
| FedPer shows no gain over local-only | Report the crossover point instead of a binary claim. Pre-register this framing in the proposal so it reads as a finding, not a failure. |
| MediaPipe extraction is slow | One-time cost; cache to `.npy` and never re-extract. Parallelize across CPU cores. Budget one overnight run for AUTSL. |
| Real-time fps too low | Reduce face landmark subset, lower input resolution, run detection every N frames with tracking between. Measure before optimizing. |
| Scope creep toward translation | The out-of-scope list in §2 is in the proposal. Point at it. |
| Deaf participants unavailable | §3. State it, reposition as methodological, don't overclaim. |

---

## 11. What "done" looks like in October

1. A chart showing the signer-independent gap and how fast each method closes it as a function of k.
2. A live demo: you sign at your laptop, captions appear in a Zoom window.
3. A repo where `make figures` reproduces every number in the thesis from committed JSON.
4. A limitations section you're not embarrassed by.

That is a complete, defensible, honest Master's thesis.
