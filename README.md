# Sine Wave Extraction and Signal De-noising

## 📖 Project Overview
This project implements a high-precision signal de-noising system using deep learning. It focuses on extracting individual pure sine waves from a composite signal corrupted by **Gaussian noise**. The system evaluates and compares three neural architectures: **Fully Connected (FC)**, **Recurrent Neural Networks (RNN)**, and **Long Short-Term Memory (LSTM)**.

---

## 🚀 Engineering Standards
This project strictly adheres to the following engineering and quality standards:
*   **SDK Layer Architecture:** All core logic is encapsulated within the `SignalDenoiserSDK`. External consumers must use this single entry point.
*   **Coding Standards:** Managed by **Ruff** (0 violations). No single file exceeds **150 lines**.
*   **OOP & DRY:** Implemented using Object-Oriented Programming with zero code duplication via abstract base classes.
*   **Testing:** **TDD** approach with >85% coverage verified by `pytest` and `pytest-cov`.
*   **Dependency Management:** Managed exclusively by **`uv`**.

---

## 🛠 Installation & Setup

### Prerequisites
*   Ensure the [`uv`](https://github.com/astral-sh/uv) package manager is installed on your system.

### Environment Initialization
1.  Clone the repository and navigate to the project root.
2.  Synchronize the environment and install dependencies:
    ```bash
    uv sync
    ```
3.  Activate the virtual environment:
    ```bash
    # Windows
    .venv\Scripts\activate
    # Linux/MacOS
    source .venv/bin/activate
    ```

---

## ⚙️ Configuration Guide (`config.py`)
All system parameters are managed dynamically in `config.py`. Key parameters include:
*   **Data Specs:** `SAMPLING_RATE` (1000Hz), `DURATION` (10s), `WINDOW_SIZE` (10 samples).
*   **Signal Specs:** `FREQUENCIES` (5, 10, 15, 20 Hz), `BASE_AMPLITUDE`.
*   **Noise Specs:** `NOISE_ALPHA` (Amplitude noise), `NOISE_BETA` (Phase noise).
*   **Model Hyperparameters:** `HIDDEN_SIZE`, `NUM_LAYERS`, `LEARNING_RATE`, `BATCH_SIZE`.

*Zero hardcoding is permitted in the `src/` directory.*

---

## 💻 Usage

### SDK Integration
The system is designed to be used via the `SignalDenoiserSDK` interface:

```python
from src.sdk.interface import SignalDenoiserSDK

# Initialize SDK with config path
sdk = SignalDenoiserSDK(config_path="config.py")

# Generate and partition dataset (70/15/15)
sdk.prepare_data()

# Train models
sdk.run_training(model_type="FC")
sdk.run_training(model_type="RNN")
sdk.run_training(model_type="LSTM")

# Evaluate on test set and generate report
sdk.evaluate_all()
sdk.generate_report()
```

### Running Tests
Execute the full test suite with coverage report:
```bash
uv run pytest --cov=src
```

---

## 📊 Results & Analysis
*(To be populated after model training)*

### Performance Comparison (Test Set MSE/MAE)
| Architecture | Test MSE | Test MAE | Training Time | Peak Memory |
|--------------|----------|----------|---------------|-------------|
| **FC**       | TBD      | TBD      | TBD           | TBD         |
| **RNN**      | TBD      | TBD      | TBD           | TBD         |
| **LSTM**     | TBD      | TBD      | TBD           | TBD         |

### Sensitivity Analysis
Automated high-resolution plots illustrating reconstruction quality versus noise intensity ($\alpha, \beta$) and training loss curves are exported to the `assets/` directory.

### Architecture Insights & Failure Modes
*   **RNN vs. LSTM:** Analysis of how temporal gates handle phase coherence.
*   **FC Baseline:** Evaluation of static window processing without sequential context.
*   **Failure Analysis:** Detailed report on performance degradation in extreme noise scenarios (>80%).

---

## 📈 Parameter Justification
*   **Frequencies:** Selected (5, 10, 15, 20 Hz) to ensure distinct periodicity within the 10ms (10-sample) window, avoiding aliasing.
*   **Noise Distribution:** **Gaussian** distribution was chosen to simulate realistic thermal and electronic noise in signal transmission.
*   **Sampling:** 1000Hz sampling rate provides sufficient resolution for the highest target frequency (20Hz) as per the Nyquist theorem.

---

## 📂 Project Metadata
*   **Version:** 1.00 (`src/shared/version.py`)
*   **License:** MIT
*   **Credits:** `numpy`, `torch`, `matplotlib`, `psutil`.

---
*Generated as part of the Sine Wave Extraction Home Task 1.*
