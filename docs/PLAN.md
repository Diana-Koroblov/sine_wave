# Architecture & Design Document (PLAN.md)

## 1. High-Level Architecture (C4 Model Approach)

### 1.1 System Context
The **Signal De-noising System** is a standalone Python library (SDK).
*   **External Consumer:** Researchers, CLI tools, or Jupyter Notebooks.
*   **Interaction:** Consumers interact **EXCLUSIVELY** with the `SignalDenoiserSDK` class. No direct access to internal signal generation or model weight manipulation is permitted.

### 1.2 Containers
*   **SDK Entry Point:** Orchestrates data generation, training, and evaluation.
*   **Data Generation Engine:** Handles mathematical synthesis of sine waves and Gaussian noise injection.
*   **Model Training Pipeline:** Manages the lifecycle of RNN, LSTM, and FC models.
*   **Evaluation & Visualization Engine:** Generates metrics (MSE) and automated plots.

### 1.3 Components (Module Breakdown)
To respect the **150-line file limit**, the logic is decomposed as follows:
*   `src/sdk/interface.py`: Public entry point (`SignalDenoiserSDK`).
*   `src/sdk/gatekeeper.py`: Validates input data (One-Hot vector structure and window dimensions) before model processing.
*   `src/data/generator.py`: Mathematical synthesis of $S_i(t)$ and $S'_i(t)$.
*   `src/data/processor.py`: Sliding window extraction and One-Hot encoding.
*   `src/models/base.py`: Abstract Base Class (`BaseModel`) for shared OOP logic.
*   `src/models/fc.py`, `src/models/rnn.py`, `src/models/lstm.py`: Specific model implementations.
*   `src/training/trainer.py`: Generic training loop and loss computation.
*   `src/utils/config.py`: Dynamic configuration loader (zero hardcoding).
*   `src/utils/visuals.py`: Automated graph exports and plotting.

---

## 2. Architecture Decision Records (ADRs)

### ADR 1: SDK Abstraction
*   **Decision:** All business logic is hidden behind a single SDK class.
*   **Rationale:** This ensures that experimental code in Notebooks does not bypass validation or configuration checks. It provides a stable API for future GUI or web integrations.

### ADR 2: OOP & Module Decoupling (DRY)
*   **Decision:** Use an abstract `BaseModel` class inheriting from `torch.nn.Module`.
*   **Rationale:** RNN, LSTM, and FC architectures share identical input/output requirements (14-in, 10-out). A base class centralizes weight initialization and common forward-pass validation, preventing code duplication across the three model files.

### ADR 3: Configuration Management
*   **Decision:** Use a centralized `Config` object initialized from a Python-based configuration file.
*   **Rationale:** This eliminates hardcoded values. Changes to frequencies, noise levels ($\alpha, \beta$), or hidden layer sizes are applied globally by modifying one file, ensuring experiments are reproducible.

### ADR 4: Quality & TDD
*   **Decision:** Strict commitment to a Test-Driven Development (TDD) workflow.
*   **Rationale:** To ensure high reliability and maintainability, tests must be written before implementation. We maintain a minimum of 85% coverage via `pytest-cov`, with tests separated into `tests/unit` (component isolation) and `tests/integration` (end-to-end SDK flows).

---

## 3. API Contracts & Interfaces

### 3.1 `SineWaveDatasetGenerator` (src/data/generator.py)
```python
class SineWaveDatasetGenerator:
    def generate_all_vectors(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary containing 10 vectors:
        - 'sum_noise': Σ_noise
        - 'pure_1'...'pure_4': Individual S_i
        - 'noisy_1'...'noisy_4': Individual S'_i
        - 'sum_pure': Optional Σ_pure for visualization
        """
        pass
```

### 3.2 `SignalDenoiserSDK` (src/sdk/interface.py)
```python
class SignalDenoiserSDK:
    def __init__(self, config_path: str): ...
    def prepare_data(self) -> None: ...
    def run_training(self, model_type: str) -> Dict[str, Any]: ...
    def generate_report(self) -> str: ...
```

### 3.3 `BaseModel` (src/models/base.py)
```python
class BaseModel(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (Batch, 14), Output: (Batch, 10)"""
        pass
```

---

## 4. Data Flow & Sequence

1.  **Initialization:** `SignalDenoiserSDK` loads `config.py`.
2.  **Synthesis:** `SineWaveDatasetGenerator` produces 10,000 samples for 4 waves using $S_i(t) = A_i \cdot \sin(2\pi f_i t + \theta_i)$.
3.  **Noise Injection:** Apply Gaussian noise to generate $S'_i(t)$ and sum them to form $\Sigma_{\text{noise}}$.
4.  **Sampling Loop (Training Step):**
    *   Select random index $t$ and target $i \in \{1,2,3,4\}$.
    *   Generate One-Hot vector $C$ for target $i$.
    *   Extract 10-sample window from $\Sigma_{\text{noise}}$ at index $t$.
    *   Concatenate $C$ (left) + Window (right) to form $X_{\text{input}}$.
    *   **Gatekeeping:** `Gatekeeper` validates $X_{\text{input}}$ dimensions and values.
    *   Extract 10-sample ground truth window from $S_i$ at index $t$ ($Y_{\text{true}}$).
5.  **Execution:** $X_{\text{input}}$ passes through the selected model (FC/RNN/LSTM).
6.  **Optimization:** Compute **MSE Loss** between prediction and $Y_{\text{true}}$; perform backpropagation.

---

## 5. Deployment & Environment

### 5.1 Dependency Management
*   **Tool:** `uv`
*   **Lockfiles:** `pyproject.toml` and `uv.lock` are the source of truth for the environment.

### 5.2 Directory Structure
```text
sine_waves_project/
├── pyproject.toml
├── uv.lock
├── config.py           # Global Parameters
├── src/
│   ├── sdk/            # Public Interface & Gatekeeper
│   ├── data/           # Synthesis & Windowing logic
│   ├── models/         # RNN, LSTM, FC (OOP implementation)
│   ├── training/       # Training loops (MSE logic)
│   └── utils/          # Config loading & Visualization
├── tests/              # TDD Suite (min 85% coverage)
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── docs/               # PRDs, PLAN.md, TODO.md
└── notebooks/          # Analysis & Sensitivity Graphs
```

---

## 6. Performance Tracking

The system monitors and reports compute resources to ensure optimization and transparency.
*   **Metrics:** Training time per architecture, inference latency, and peak memory usage.
*   **Tools:** Uses the `time` library for duration tracking and `psutil` for memory monitoring.
*   **Reporting:** Metrics are logged during training and exported to the Jupyter Notebook for final visualization and comparison.
