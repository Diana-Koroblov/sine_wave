# Technical Design & Implementation Plan: Sine Wave Extraction

## 1. Architecture Overview
The system is designed as a modular pipeline where data generation, model definition, training, and evaluation are decoupled. This modularity ensures that each component can be tested independently and kept under the 150-line limit.

### 1.1 Directory Structure
```text
project_root/
│
├── config.py           # Centralized constants & hyperparameters
├── src/
│   ├── data_gen.py     # Signal synthesis and dataset preparation
│   ├── models.py       # Neural network architectures (FC, RNN, LSTM)
│   ├── train.py        # Training loops and model persistence
│   ├── eval.py         # Metrics, plotting, and comparison logic
│   └── utils.py        # Shared helpers (OHE, windowing)
├── tests/              # Pytest suite
│   ├── test_data.py
│   ├── test_models.py
│   └── test_utils.py
├── docs/
│   ├── PRD.md
│   ├── PLAN.md
│   └── TODO.md
└── README.md           # Final report and analysis
```

## 2. Component Breakdown

### 2.1 `config.py`
*   **Purpose:** The single source of truth for all constants.
*   **Contents:** Frequency list, amplitude, noise intensity ($\alpha, \beta$), noise distribution, sampling rate (1000Hz), duration (10s), window size (10), hidden layers, neurons, learning rate, and batch size.

### 2.2 `src/data_gen.py`
*   **Purpose:** Pure functional approach to signal generation.
*   **Key Functions:** `generate_pure_sine()`, `apply_noise()`, `create_combined_signal()`.
*   **Constraint:** No training logic allowed here.

### 2.3 `src/models.py`
*   **Purpose:** Definition of PyTorch/TensorFlow models.
*   **Classes:** `FCNET`, `RNNNet`, `LSTMNet`.
*   **Constraint:** Models must accept the combined input shape (One-Hot vector + 10-sample window).

### 2.4 `src/utils.py`
*   **Purpose:** Data transformations.
*   **Key Functions:** `to_one_hot()`, `sliding_window_split()`.
*   **Constraint:** Must be stateless and purely mathematical.

### 2.5 `src/train.py`
*   **Purpose:** Orchestrate the training process.
*   **Logic:** Dataset loading, backpropagation, checkpoint saving, and loss history logging.

### 2.6 `src/eval.py`
*   **Purpose:** Post-training analysis.
*   **Logic:** Signal reconstruction visualization (Pure vs. Noisy vs. Pred), MSE calculation across architectures, and noise impact plotting.

## 3. Implementation Phases

### Phase 1: Foundation (Environment & Config)
*   Initialize Git repository.
*   Define all constants in `config.py`.
*   Setup requirements (NumPy, PyTorch/TensorFlow, Matplotlib, Pytest).

### Phase 2: Data & Utils
*   Implement `utils.py` (Windowing/OHE).
*   Implement `data_gen.py` (Sine generation).
*   **Milestone:** Run `pytest tests/test_data.py` to verify signal integrity.

### Phase 3: Modeling
*   Implement `models.py` for FC, RNN, and LSTM.
*   Verify output shapes for a single forward pass with dummy data.

### Phase 4: Training Pipeline
*   Implement `train.py`.
*   Establish a "baseline" run with low noise to verify the models can learn.
*   Save model weights for the evaluation phase.

### Phase 5: Evaluation & Reporting
*   Implement `eval.py` to generate plots.
*   Run "Extreme Noise" experiments (>80%).
*   Compile results into the `README.md`.

## 4. Testing Strategy (`pytest`)
*   **Signal Integrity:** Assert that `generate_pure_sine` produces a signal with the correct frequency (using FFT or zero-crossing counts).
*   **Time Alignment:** Verify that the 10th sample in an input window corresponds to the 10th sample in the target window.
*   **Shape Validation:** Ensure the input to the models is exactly `(Batch, 14)` (4 for OHE + 10 for signal).
*   **Noise Range:** Assert that noise added with $\alpha=0.1$ stays within the $\pm 10\%$ amplitude bound.

## 5. Coding Standards & Maintenance
*   **Line Limit:** Any file exceeding 150 lines will be refactored into smaller sub-modules or helper files.
*   **Type Hinting:** Mandatory for all function signatures to prevent runtime data-type errors.
*   **Documentation:** Every function must have a docstring explaining inputs and expected outputs.
