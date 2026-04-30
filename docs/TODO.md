# Project Task Tracking (TODO.md)

## Phase 1: Environment & Tooling Setup
- [ ] **Initialize Project with `uv`**
    - **DoD:** `pyproject.toml` and `uv.lock` generated; `uv sync` completes successfully.
- [ ] **Configure Quality Enforcement (`Ruff`)**
    - **DoD:** `ruff` configuration file created; `ruff check .` returns 0 violations.
- [ ] **Set up Testing Suite (`pytest` & `pytest-cov`)**
    - **DoD:** Testing infrastructure initialized; coverage reports verify a minimum of 85%.
- [ ] **Establish Mandatory Directory Structure**
    - **DoD:** All directories (`src/sdk/`, `src/models/`, `src/data/`, `src/training/`, `src/utils/`, `docs/`, `notebooks/`, `tests/unit/`, `tests/integration/`, `assets/`) created.

## Phase 2: Configuration & SDK Foundation
- [ ] **Implement Dynamic `config.py`**
    - **DoD:** All parameters (frequencies, noise coefficients α/β, hyperparameters) reside in `config.py`; zero hardcoded values in `src/`.
- [ ] **Implement `SignalDenoiserSDK` Entry Point**
    - **DoD:** Public interface defined in `src/sdk/interface.py`; all external calls must go through this class.
- [ ] **Implement `Gatekeeper` Input Validation**
    - **DoD:** `src/sdk/gatekeeper.py` validates One-Hot vector structure and window dimensions (10 samples) before model execution.

## Phase 3: Dataset Generation Engine
- [ ] **Implement `SineWaveDatasetGenerator`**
    - **DoD:** Mathematical synthesis using $S_i(t) = A_i \cdot \sin(2\pi f_i t + \theta_i)$.
    - **DoD:** Explicit generation of **Gaussian noise** relative to signal intensity (α, β).
    - **DoD:** Export exactly 10 vectors: $\Sigma_{\text{noise}}$, $\Sigma_{\text{pure}}$, 4 pure waves $S_i$, and 4 noisy waves $S'_i$.
    - **DoD:** Output exactly 10,000 samples per wave (10s @ 1000Hz).
    - **DoD:** TDD applied; `src/data/generator.py` < 150 lines; >85% unit test coverage.

## Phase 4: Model Architectures (OOP & DRY)
- [ ] **Implement `BaseModel` Abstract Class**
    - **DoD:** Shared logic for weight initialization and input/output validation centralized in `src/models/base.py`.
- [ ] **Implement `RNNModel`, `LSTMModel`, and `FCModel`**
    - **DoD:** All models inherit from `BaseModel`; zero code duplication (DRY).
    - **DoD:** Each model file (`fc.py`, `rnn.py`, `lstm.py`) is strictly under 150 lines.
    - **DoD:** `tests/unit/test_models.py` verifies forward pass shapes (14-in, 10-out).

## Phase 5: Training Pipeline & Dynamic Sampling
- [ ] **Implement Dataset Partitioning Logic**
    - **DoD:** Implementation of a strict 70/15/15 split (Training, Validation, and Test sets) to evaluate model generalization.
- [ ] **Implement Data Integrity & Validation Checks**
    - **DoD:** Integration of `Gatekeeper` logic to verify that generated vectors (Sine waves and combined signals) match the required shapes (e.g., 10,000 samples) and types before training.
- [ ] **Apply TDD to the Training Orchestrator**
    - **DoD:** Unit tests in `tests/unit/` verify the generic training loop, MSE calculation logic, and backpropagation flow using mock data.
- [ ] **Implement Dynamic Sampling Logic**
    - **DoD:** Verified correct concatenation of the One-Hot vector $C$ (left) and the 10-sample window (right).
- [ ] **Implement Performance & Resource Tracking**
    - **DoD:** SDK logs training duration (via `time`) and peak memory usage (via `psutil`) for the performance report.

## Phase 6: Research, Visualization & Notebooks
- [ ] **Initialize and Integrate Research Notebook**
    - **DoD:** `notebooks/analysis.ipynb` created, importing logic exclusively via the SDK layer.
- [ ] **Perform Model Evaluation on Test Set**
    - **DoD:** Comparative metrics (MSE) reported specifically for the unseen Test Set to demonstrate true performance.
- [ ] **Sensitivity Analysis & Automated Visualizations**
    - **DoD:** Graphs showing reconstruction quality vs. noise intensity (α, β) and training loss curves exported to the `assets/` directory.

## Phase 7: Final Documentation & Quality Polish
- [ ] **Assemble Technical README.md**
    - **DoD:** Comprehensive User Manual included (System requirements, Installation via `uv sync`, Usage instructions, and Configuration guide for `config.py`).
    - **DoD:** Performance Comparison Table included (MSE/MAE metrics across FC, RNN, and LSTM architectures).
    - **DoD:** Parameter Justification provided (Detailed explanation of chosen frequencies vs. sampling rate and period/window ratio).
    - **DoD:** License, Credits for third-party libraries, and Repository link included.
- [ ] **Integrate Research Visualizations**
    - **DoD:** README contains exported high-resolution plots of Pure vs. Noisy vs. Reconstructed signals for various frequencies.
    - **DoD:** Training Loss curves and Sensitivity Analysis graphs (performance degradation in >80% noise cases) embedded.
- [ ] **Finalize Professional Documentation Suite**
    - **DoD:** Dedicated architecture PRDs (`PRD_RNN.md`, `PRD_LSTM.md`, `PRD_FC.md`) finalized in the `docs/` directory.
    - **DoD:** Prompt Engineering Log documented (All major AI prompts used for architecture, logic, and debugging included).
- [ ] **Global Versioning & Package Integrity**
    - **DoD:** `src/shared/version.py` established at version 1.00 and correctly synced with `pyproject.toml`.
    - **DoD:** Verification that every package directory contains a valid `__init__.py` file for Python package standard compliance.
- [ ] **Draft Cost & Resource Report**
    - **DoD:** A table summarizing total training time per model and peak memory (RAM) usage documented in the README.
- [ ] **Final Compliance Audit (Zero-Failure Gate)**
    - **DoD:** `ruff check .` returns exactly 0 violations.
    - **DoD:** `pytest --cov` confirms a minimum of 85% code coverage for all business logic.
    - **DoD:** Manual/Automated verification that no single file in the project exceeds the 150-line limit.
