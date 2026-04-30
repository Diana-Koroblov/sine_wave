# Product Requirements Document (PRD): Sine Wave Extraction and Signal De-noising

## 1. Project Overview
This project aims to develop a robust system for extracting individual pure sine waves from a combined noisy signal using deep learning architectures (RNN, LSTM, and Fully Connected networks). The system must handle signal synthesis, noisy dataset generation, and model-based reconstruction with high precision and adherence to strict engineering standards.

---

## 2. V3NEW Engineering Constraints (MANDATORY)

### 2.1 SDK Layer Architecture
*   **Encapsulation:** All business logic, including data generation, model definition, training, and evaluation, must be encapsulated within a dedicated SDK (Software Development Kit).
*   **Strict Access:** External consumers (CLI, GUI, or Scripts) must interact with the system EXCLUSIVELY through the SDK's single entry point. Internal modules must not be accessed directly by external layers.

### 2.2 Coding Standards
*   **Linter Compliance:** Mandatory use of the `Ruff` linter. The codebase must maintain exactly 0 violations at all times.
*   **File Length Limits:** No single file (including implementation and test files) may exceed 150 lines. Large modules must be refactored into smaller, cohesive units.
*   **Design Principles:** Mandatory use of Object-Oriented Programming (OOP) design. Zero code duplication (DRY principle - Don't Repeat Yourself) is strictly enforced.

### 2.3 Testing & Quality Assurance
*   **TDD Approach:** A Test-Driven Development (TDD) approach is mandatory. Tests must be written before or alongside the implementation.
*   **Coverage:** Minimum 85% test coverage is required for all business logic, verified using `pytest` and `pytest-cov`.

### 2.4 Configuration Management
*   **No Hardcoding:** Absolute ban on hardcoded values within the code.
*   **Centralized Config:** All parameters (frequencies, noise coefficients, model hyperparameters, sampling rates) must be loaded from external configuration files (e.g., `config.py`).

### 2.5 Dependency Management
*   **Tooling:** The project must be managed EXCLUSIVELY using the `uv` package manager.
*   **Files:** `pyproject.toml` and `uv.lock` must be present and reflect the exact environment.

### 2.6 Documentation Structure
The `docs/` directory must contain the following mandatory files:
*   `PRD.md`: This comprehensive requirement document.
*   `PLAN.md`: Mandatory Architecture document detailing design and implementation phases.
*   `TODO.md`: Granular task tracking.
*   `PRD_RNN.md`, `PRD_LSTM.md`, `PRD_FC.md`: Dedicated mechanism PRDs for each architecture.

### 2.7 Research & Analysis
*   **Notebooks:** A `notebooks/` directory must contain Jupyter Notebooks used for:
    *   Results analysis and comparison.
    *   Sensitivity analysis (noise vs. reconstruction error).
    *   Visualizations of signals and training progress.

### 2.8 Cost & Performance
*   **Documentation:** Mandatory documentation of compute resources, training time per model, and runtime optimizations.

---

## 3. Core Project Requirements (Home Task 1)

### 3.1 Dataset Generation

#### 3.1.1 Base Sine Waves
*   **Definition:** Generate 4 distinct sine waves using the formula: $S_i(t) = A_i \cdot \sin(2\pi f_i t + \theta_i)$.
*   **Frequencies ($f_i$):** 4 fixed frequencies (e.g., 5Hz, 10Hz, 15Hz, 20Hz).
*   **Parameters:** Amplitude ($A_i$) and Phase ($\theta_i$) must be configurable.
*   **Sample Count:** Exactly 10 seconds duration at a 1000Hz sampling rate, resulting in exactly 10,000 samples per wave.

#### 3.1.2 Noise Formulation
*   **Noisy Signal Definition:** $S'_i(t) = (A_i \pm \alpha \cdot \text{noise}) \cdot \sin(2\pi f_i t + (\theta_i \pm \beta \cdot \text{noise}))$.
*   **Noise Intensity:** $\alpha$ and $\beta$ represent percentage relative to the signal.
*   **Phase Noise Range:** Strictly in the range $[0, 2\pi]$.
*   **Distribution:** Noise must strictly follow a **Gaussian** distribution (Parameter Justification).

#### 3.1.3 Sampling & Duration
*   **Duration:** Exactly 10 seconds.
*   **Sampling Rate:** 1000Hz.

#### 3.1.4 Data Exports (CRITICAL)
The generator MUST export exactly 10 vectors:
1.  $\Sigma_{\text{noise}}$: The combined noisy signal (sum of all 4 noisy waves $S'_i$).
2.  $\Sigma_{\text{pure}}$: The combined pure signal (sum of all 4 pure waves $S_i$).
3.  $S_{1 \dots 4}$: The 4 pure sine waves.
4.  $S'_{1 \dots 4}$: The 4 noisy sine waves.

### 3.2 Training Example Structure

#### 3.2.1 Target Selection (C)
*   A randomly generated One-Hot Encoded vector of length 4 (e.g., `[0, 1, 0, 0]` to target $S_2$).

#### 3.2.2 Input ($X_{\text{input}}$)
*   **Format:** Concatenation of the One-Hot vector $C$ on the **left**, and a dynamically sampled window of exactly 10 consecutive samples from the combined noisy signal ($\Sigma_{\text{noise}}$) on the **right**.
*   **Window Size:** Exactly 10 samples.

#### 3.2.3 Target Output ($Y_{\text{true}}$)
*   A window of exactly 10 consecutive samples from the corresponding pure sine wave $S_i$, from the exact same time location as the input window.

### 3.3 Architectures & Training

#### 3.3.1 Models
Implement and train three separate architectures: RNN, LSTM, and Fully Connected (FC).

#### 3.3.2 Loss Function (CRITICAL)
*   **Requirement:** The Loss Function MUST be Mean Squared Error (MSE) comparing the predicted 10-sample window and $Y_{\text{true}}$.

#### 3.3.3 Flexible Parameters
The user must choose, document, and justify:
*   Number of layers.
*   Number of perceptrons/neurons per layer.
*   Activation functions.

---

## 4. Deliverables & Documentation

### 4.1 README.md
A highly detailed README including:
*   **Comparative Analysis:** Detailed comparisons explaining exactly when RNN is better, when LSTM is preferred, and when FC is suitable.
*   **Failure Mode Analysis:** Analysis explaining when and why each network fails.
*   **Sensitivity Analysis:** Graphs and data demonstrating the correlation between noise intensity and reconstruction quality (error).
*   **Automated Visualizations:** Mandatory automated exports/screenshots of graphs (pure sine waves, combined signals, training errors, and reconstructions).
*   **Parameter Justification:** Documentation and rationale for all chosen frequencies, noise levels, and network structures.


