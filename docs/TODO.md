# Project Task Tracking

## Phase 1: Environment & Tooling Setup
- [x] **Initialize Project with `uv`**
    - **DoD:** `pyproject.toml` and `uv.lock` generated; `uv sync` completes successfully.
    - [x] Run initialization commands to create configuration files.
    - [x] Verify `uv.lock` is generated.
- [x] **Configure Quality Enforcement (`Ruff`)**
    - **DoD:** `ruff` configuration file created; `ruff check .` returns 0 violations.
    - [x] Define `line-length = 100` and `target-version` in `pyproject.toml`.
    - [x] Run `ruff check .` to ensure compliance.
- [x] **Set up Testing Suite (`pytest` & `pytest-cov`)**
    - **DoD:** Testing infrastructure initialized; coverage reports verify a minimum of 85%.
    - [x] Configure `pytest` settings and coverage thresholds in `pyproject.toml`.
    - [x] Verify initial testing baseline passes.
- [x] **Establish Mandatory Directory Structure**
    - **DoD:** All directories created.
    - [x] Create `src/`, `docs/`, `notebooks/`, `tests/`, and `assets/` directories.
    - [x] Add `__init__.py` to all source packages.

## Phase 2: Configuration & SDK Foundation
- [x] **Implement Dynamic `config.py`**
    - **DoD:** All parameters (frequencies, noise coefficients α/β, hyperparameters) reside in `config.py`; zero hardcoded values in `src/`.
    - [x] Define data spec constants (`SAMPLING_RATE`, `DURATION`, `WINDOW_SIZE`).
    - [x] Define signal spec lists (`FREQUENCIES`, `AMPLITUDES`, `PHASES`).
    - [x] Define noise spec scalars (`NOISE_ALPHA`, `NOISE_BETA`).
    - [x] Define model hyperparameters (`LEARNING_RATE`, `HIDDEN_SIZE`, `NUM_LAYERS`).
    - [x] Add technical rationale for parameter choices as a docstring/comment block.
- [x] **Implement `SignalDenoiserSDK` Entry Point**
    - **DoD:** Public interface defined in `src/sdk/interface.py`; all external calls must go through this class.
    - [x] Create `SignalDenoiserSDK` class in `src/sdk/interface.py`.
    - [x] Implement `__init__` to store the config file path.
    - [x] Add public method stubs: `prepare_data()`, `run_training()`, and `generate_report()`.
    - [x] Create `tests/unit/test_interface.py`.
    - [x] Verify that the class initializes and stores the config path correctly.
    - [x] Verify stubs return the correct data types (None, dict, str).
- [x] **Implement `Gatekeeper` Input Validation**
    - **DoD:** `src/sdk/gatekeeper.py` validates One-Hot vector structure, sigma range, and window dimensions before model execution.
    - [x] Create `src/sdk/gatekeeper.py`.
    - [x] Implement `validate_input_vector` to check for the configured input length.
    - [x] Add logic to verify One-Hot encoding (values in {0, 1}, exactly one '1').
    - [x] Add logic to verify the sigma percentage is within the configured bounds.
    - [x] Implement `validate_window_dimensions` to check for 10-sample length.
    - [x] Ensure `ValueError` is raised with descriptive messages for all failures.
    - [x] Create `tests/unit/test_gatekeeper.py`.
    - [x] Verify valid input vectors are accepted without error.
    - [x] Verify `ValueError` is raised for incorrect shapes and malformed One-Hot vectors.
- [x] **Implement `BaseModel` Abstract Class**
    - **DoD:** Shared logic for weight initialization and input/output validation centralized in `src/models/base.py`.
    - [x] Create `src/models/base.py`.
    - [x] Inherit from `torch.nn.Module` and `abc.ABC`.
    - [x] Implement `_validate_dimensions` to enforce the configured input/output contract.
    - [x] Implement `init_weights` using `nn.init.xavier_uniform_`.
    - [x] Create `tests/unit/test_base_model.py`.
    - [x] Use a `MockModel` (concrete subclass) to test `BaseModel` initialization logic.
    - [x] Verify `ValueError` is raised if dimensions deviate from the 14/10 standard.

## Phase 3: Dataset Generation Engine
- [x] **Implement `SineWaveDatasetGenerator`**
    - **DoD:** Mathematical synthesis using $S_i(t) = A_i \cdot \sin(2\pi f_i t + \theta_i)$.
    - [x] Create the file `src/data/generator.py`.
    - [x] Import `numpy` and the constants from `config.py`.
    - [x] Define the `SineWaveDatasetGenerator` class.
    - [x] Write a helper method `_generate_time_axis()` to create a 1D array from 0 to 10 seconds with 10,000 points.
    - [x] Write a method to generate the 4 pure sine waves using the values in `config.AMPLITUDES`, `config.FREQUENCIES`, and `config.PHASES`.
- [x] **Implement Gaussian Noise Injection**
    - **DoD:** Explicit generation of **Gaussian noise** relative to signal intensity (α, β).
    - [x] Write a helper method `_generate_gaussian_noise(size)` using `np.random.normal`.
    - [x] Implement the configured Gaussian noise formula for amplitude and phase perturbation.
    - [x] Ensure noise is applied independently to both Amplitude and Phase for each of the 4 waves.
- [x] **Data Structuring & Export**
    - **DoD:** Export exactly 10 vectors: $\Sigma_{\text{noise}}$, $\Sigma_{\text{pure}}$, 4 pure waves $S_i$, and 4 noisy waves $S'_i$.
    - [x] Implement the `generate_all_vectors()` method.
    - [x] Calculate `sum_pure` by summing the 4 pure signals.
    - [x] Calculate `sum_noise` by summing the 4 noisy signals.
    - [x] Package all 10 vectors into a Python dictionary with clear keys (e.g., `'pure_1'`, `'noisy_1'`, `'sum_noise'`, etc.).
- [x] **Verification of Constraints**
    - **DoD:** Output exactly 10,000 samples per wave (10s @ 1000Hz).
    - [x] Add an internal check (assertion) ensuring every vector has a `shape` of `(10000,)`.
    - **DoD:** TDD applied; `src/data/generator.py` < 150 lines (Note: The 150 limit applies to the vertical number of rows per file, not the character line-length); >85% unit test coverage.
    - [x] Create the test file `tests/unit/test_generator.py`.
    - [x] If any file exceeds 150 lines, split it into two files instead of compressing or shortening the code.
    - [x] Write a test to verify the dictionary contains exactly 10 keys with the correct names.
    - [x] Write a test to verify that all vectors in the dictionary have exactly 10,000 elements.
    - [x] Write a test to verify that the `sum_noise` values are mathematically different from `sum_pure` (proving noise was added).
    - [x] Write a test to ensure that pure signals match the frequencies defined in `config.py`.

## Phase 4: Model Architectures (OOP & DRY)
- [x] **Finalize `BaseModel` Integration**
    - **DoD:** Shared logic for weight initialization and input/output validation centralized in `src/models/base.py`.
    - [x] Ensure all future model classes in this phase inherit from `BaseModel`.
    - [x] Row-Count Audit: Verify `base.py` is < 150 rows. If it exceeds this limit, split it into logical sub-modules immediately. Do NOT shorten or compress code.
    - [x] Verify that `self._validate_dimensions()` is called during initialization to prevent contract violations.
    - [x] Confirm `self.init_weights()` is available for use in concrete model constructors.
    - [x] Enhanced `BaseModel` to support recurrent layer weight initialization.
- [x] **Implement Concrete Neural Architectures**
    - **DoD:** All models inherit from `BaseModel`; zero code duplication (DRY).
    - [x] Create `src/models/fc.py` and implement the `FCModel` class using dynamic layer stacking (based on `config.NUM_LAYERS`).
    - [x] Row-Count Audit: Verify `fc.py` is < 150 rows; split if necessary without shortening code.
    - [x] Create `src/models/rnn.py` and implement the `RNNModel` class using `nn.RNN` with `Tanh` activation.
    - [x] Row-Count Audit: Verify `rnn.py` is < 150 rows; split if necessary without shortening code.
    - [x] Create `src/models/lstm.py` and implement the `LSTMModel` class using `nn.LSTM` with the dual-state system.
    - [x] Row-Count Audit: Verify `lstm.py` is < 150 rows; split if necessary without shortening code.
    - [x] In each class, implement the `forward` method to process the configured input shape and return the 10-element output.
    - **DoD:** Each model file is strictly under 150 lines (Note: The 150 limit applies to the vertical number of rows per file, not the character line-length).
    - [x] Audit the row count for `fc.py`, `rnn.py`, and `lstm.py` to ensure they are within the 150-row limit.
    - [x] Ensure `ruff check .` returns zero violations for all three model files.
    - [x] **Enhanced `BaseModel`** to support recurrent layer weight initialization.
- [x] **Verify Model Integrity (TDD)**
    - **DoD:** `tests/unit/test_models.py` verifies forward pass shapes against the configured contract.
    - [x] Create `tests/unit/test_models.py`.
    - [x] Row-Count Audit: Verify `test_models.py` is < 150 rows; split if necessary.
    - [x] Write a test case for `FCModel` that asserts the output shape is exactly `(1, 10)` given a valid configured input.
    - [x] Write a test case for `RNNModel` that asserts the output shape is exactly `(1, 10)` given a valid configured input.
    - [x] Write a test case for `LSTMModel` that asserts the output shape is exactly `(1, 10)` given a valid configured input.
    - [x] Write a test to ensure `init_weights` modifies the default parameters (proves initialization logic is executing).
    - **DoD:** `tests/unit/` contains full coverage (>85%) for Phase 4 components.
    - [x] Run `uv run pytest --cov=src/models` and verify the coverage meets the 85% project gate.

## Phase 5: Training Pipeline & Dynamic Sampling
- [x] Strict 70/15/15 split implemented in the SDK.
- [x] Dataset generation yields 60,000 random windows.
- [x] Current contract uses 15-value inputs: 4 OHE bits + 1 sigma + 10 noisy samples.
- [x] `Gatekeeper` validates batch shape, OHE structure, sigma range, and window dimensions.
- [x] `ModelTrainer` handles generic training, validation, best-weight retention, and MSE loss.
- [x] Performance tracking returns `total_time` and `peak_ram_mb`.
- [x] Training loop behavior is covered by unit tests.

## Phase 6: Research, Visualization & Notebooks
- [x] `notebooks/analysis.ipynb` imports and runs the project only through `SignalDenoiserSDK`.
- [x] `evaluate_on_test_set()` reports MSE, MAE, and Pearson correlation on the held-out test split.
- [x] Sensitivity analysis sweeps default noise levels from 0.1 to 0.9.
- [x] Reconstruction, loss-curve, sensitivity, and per-frequency MSE figures export to `assets/`.
- [x] Frequency comparison plot is exported as `frequency_mse_comparison.png`.
- [x] Visual utilities and sensitivity logic are covered by tests.

## Phase 7: Final Documentation & Quality Polish
- [x] README covers setup, usage, configuration, results, and technical metadata.
- [x] Research figures are referenced from committed assets.
- [x] PRD companion documents and prompt log exist under `docs/`.
- [x] Version 1.00 is defined in `src/shared/version.py` and mirrored in `pyproject.toml`.
- [x] Resource usage is documented in the report material.

## Phase 8: High-Frequency Experimental Comparison
- [x] Archived low-frequency assets live under `assets/v1_low_freq/`.
- [x] Active high-frequency configuration uses 25, 50, 100, and 150 Hz.
- [x] Experiment-B assets export to `assets/v2_high_freq/`.
- [x] High-frequency analysis is documented in the report material.
- [x] Repo quality gates currently pass: Ruff clean, pytest green, coverage above 85%.
