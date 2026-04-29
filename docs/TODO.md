# Project Tracking Checklist (TODO)

## Phase 1: Foundation & Environment
- [ ] Initialize Git repository.
- [ ] Initialize project structure using `uv init` and ensure `pyproject.toml` and `uv.lock` are created.
- [ ] Create a `docs/prompt_book.md` to log all AI prompts used during development.
- [x] Implement `config.py` with all centralized constants:
    - [x] Frequencies (5, 10, 15, 20 Hz).
    - [x] Sampling rate (1000 Hz) and Duration (10s).
    - [x] Window size (10).
    - [x] Base Amplitude (A) e.g., 1.0.
    - [x] Base Phase (theta) e.g., 0.0.
    - [x] Noise parameters (alpha, beta, using Gaussian distribution).
    - [x] Model hyperparameters (hidden units, layers, LR, etc.).
- [ ] Verify environment setup by importing libraries.

## Phase 2: Data & Utilities
- [x] Implement `src/utils.py`:
    - [x] `to_one_hot(index, total_classes)` function.
    - [x] `sliding_window_split(data, window_size)` function.
- [ ] Implement `src/data_gen.py`:
    - [ ] `generate_pure_sine(freq, duration, sampling_rate, amplitude, phase)`
    - [ ] `apply_noise(signal, alpha, beta, distribution)`
    - [ ] `create_combined_signal(noisy_signals)`
    - [ ] `prepare_dataset(config)` - function to orchestrate the creation of X and Y, **and split them into Train, Validation, and Test sets**.
- [x] Implement `tests/test_utils.py`:
    - [x] Test One-Hot vector dimensions.
    - [x] Test windowing output shapes and values.
- [ ] Implement `tests/test_data.py`:
    - [ ] Test signal duration and sample count.
    - [ ] Test noise bounds (amplitude and phase).
    - [ ] Test combined signal summation.
    - [ ] Test final X_input shape equals 14 (4 OHE + 10 Signal).
    - [ ] Test final Y_true shape equals 10.
    - [ ] **Test Time-Alignment:** Verify that X_input windows and Y_true windows correspond to the exact same time indices.
- [ ] **Milestone:** All tests in `tests/` pass with `pytest`.
- [ ] **Milestone:** Ensure minimum 85% test coverage using `pytest-cov`.
- [ ] **Milestone:** Run `ruff check` and ensure exactly 0 violations.

## Phase 3: Modeling
- [ ] Implement `src/models.py`:
    - [ ] `FCNet` class definition (dynamically built using `config.py` parameters).
    - [ ] `RNNNet` class definition (dynamically built using `config.py` parameters).
    - [ ] `LSTMNet` class definition (dynamically built using `config.py` parameters).
- [ ] Implement `tests/test_models.py`:
    - [ ] Verify `FCNet` forward pass (Input shape: Batchx14, Output shape: Batchx10).
    - [ ] Verify `RNNNet` forward pass (Input shape: Batchx14, Output shape: Batchx10).
    - [ ] Verify `LSTMNet` forward pass (Input shape: Batchx14, Output shape: Batchx10).
- [ ] **Milestone:** Ensure minimum 85% test coverage using `pytest-cov`.
- [ ] **Milestone:** Run `ruff check` and ensure exactly 0 violations.

## Phase 4: Training Pipeline
- [ ] Implement `src/train.py`:
    - [ ] Dataset and DataLoader setup (using Batch Size from `config.py`).
    - [ ] Training loop with MSE loss and Adam Optimizer (using LR from `config.py`).
    - [ ] **Mandatory** Validation step to track generalization.
    - [ ] Model saving logic (`.pth` or `.h5`).
    - [ ] Save loss history (Train Loss & Validation Loss per Epoch) for later plotting.
- [ ] Train the FC model and save weights + loss history.
- [ ] Train the RNN model and save weights + loss history.
- [ ] Train the LSTM model and save weights + loss history.
- [ ] **Milestone:** Ensure minimum 85% test coverage using `pytest-cov`.
- [ ] **Milestone:** Run `ruff check` and ensure exactly 0 violations.

## Phase 5: Evaluation & Reporting
- [ ] Implement `src/eval.py`:
    - [ ] Function to load models and run inference.
    - [ ] Plotting logic for waveforms: "Pure vs. Noisy vs. Reconstructed".
    - [ ] Plotting logic for training errors (Loss curves over epochs).
    - [ ] "Extreme Noise" test script to evaluate performance degradation.
- [ ] Finalize `README.md`:
    - [ ] **Parameter Documentation & Justification:** Explain choices for frequencies (vs. 10ms window), base amplitude, phase, network structures (layers/neurons), and why Gaussian noise distribution was chosen.
    - [ ] **Visualizations:** Embed screenshots of graphs (pure sine waves, combined noisy signals, training error curves, and reconstructed waves).
    - [ ] **Performance Comparison:** Tabulate and compare the results of Fully Connected, RNN, and LSTM.
    - [ ] **Noise vs. Error Analysis:** Demonstrate and explain the relationship between noise intensity and reconstruction error quality using graphs and data.
    - [ ] **Architecture Insights & Failure Analysis:** - Explain when RNN is better, when LSTM is preferred, and when FC is appropriate.
        - Show specifically when and why each network fails (e.g., FC failing due to lack of time context, RNN/LSTM failing at extreme noise).
- [ ] Final code review to ensure all Python/Test files strictly respect the < 150 lines limit.
