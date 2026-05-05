# Engineering Prompt Log

This document records the key engineering prompts and decisions made during the
development of the Sine Wave Extraction and Signal De-noising project.

---

## Phase 1 — Environment & Tooling
**Prompt:** Initialize the project using `uv`, configure Ruff with `line-length=100` and
`target-version="py313"`, set the pytest global coverage gate to 85%, and establish the
mandatory directory structure (`src/`, `docs/`, `notebooks/`, `tests/`, `assets/`).

**Decision:** `uv` was chosen as the exclusive package manager to guarantee a reproducible,
locked dependency graph. Ruff was selected over flake8/pylint for its single-tool simplicity.

---

## Phase 2 — Configuration & SDK Foundation
**Prompt:** Implement a zero-hardcoding `config.py` as the single source of truth for all
constants. Design a `SignalDenoiserSDK` class in `src/sdk/interface.py` as the sole public
entry point; no external caller may bypass the SDK layer.

**Decision:** A flat `config.py` module (not a class) was chosen for simplicity and to allow
direct attribute mutation during sensitivity sweeps without requiring dependency injection.

---

## Phase 3 — Dataset Generation Engine
**Prompt:** Implement `SineWaveDatasetGenerator` using the formula
$S_i(t) = A_i \cdot \sin(2\pi f_i t + \theta_i)$. Inject Gaussian noise independently into
amplitude and phase. Export exactly 10 vectors per call, each of shape `(10000,)`.

**Decision:** Gaussian noise was selected (over uniform) to model real-world thermal and
electronic noise. Noise is applied per-component (amplitude + phase) to avoid correlated
artefacts.

---

## Phase 4 — Model Architectures
**Prompt:** Implement FC, RNN, and LSTM models all inheriting from `BaseModel(nn.Module, ABC)`.
Enforce 14-in / 10-out contract in `BaseModel._validate_dimensions()`. Use Xavier uniform
initialisation for linear layers and orthogonal initialisation for recurrent weights.

**Decision:** A shared `BaseModel` abstract class was chosen to satisfy the DRY constraint
and prevent contract drift between architectures. Dynamic layer stacking from `config.NUM_LAYERS`
avoids per-architecture hardcoding.

---

## Phase 5 — Training Pipeline
**Prompt:** Implement a `ModelTrainer` class with Adam optimiser, MSELoss, per-epoch
validation, best-weight checkpointing, and `psutil`-based peak-RAM tracking. Use a strict
70/15/15 dataset split with `numpy.random.permutation` shuffling.

**Decision:** Validation loss (not training loss) was used as the checkpoint criterion to
select the best generalising weights. `psutil.Process().memory_info().rss` was sampled after
each epoch to avoid missing intra-epoch peaks.

---

## Phase 6 — Research, Visualisation & Sensitivity Analysis
**Prompt:** Implement a `Visualizer` class exporting PNGs to `assets/`. Implement
`sdk.run_sensitivity_analysis()` sweeping noise α/β from 0.1 to 0.9 in 9 steps.

**Problem encountered:** Default sweep (9 levels × 3 models × 50 epochs × 60k samples)
ran for >1 hour.

**Fix:** Added three bounded runtime parameters to `config.py`:
`SENSITIVITY_DATASET_SIZE=12_000`, `SENSITIVITY_EPOCHS=12`, `SENSITIVITY_BATCH_SIZE=256`.
These are threaded through `prepare_data()`, `run_training()`, and `evaluate_on_test_set()`
as optional overrides. Real sweep now completes in ~30 seconds.

---

## Phase 7 — Documentation & Quality Polish
**Prompt:** Assemble a technical README with Installation, Usage, Configuration, Parameter
Rationale, and a real performance comparison table. Link all four PNG artefacts from `assets/`
with technical captions. Finalize PRDs with measured results. All files must remain under
150 rows.

**Measured results used to populate tables (50 epochs, 60k samples):**

| Model | MSE | MAE | Pearson r | Time | RAM |
|-------|----:|----:|----------:|-----:|----:|
| FC    | 0.1855 | 0.2844 | 0.8490 | 12.4 s | 291 MB |
| RNN   | 0.2815 | 0.3759 | 0.7588 | 53.2 s | 297 MB |
| LSTM  | 0.2632 | 0.3521 | 0.7765 | 54.4 s | 315 MB |
