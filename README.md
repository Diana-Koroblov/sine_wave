# Sine Wave Extraction and Signal De-noising

## Project Overview
This project implements a signal de-noising system using deep learning. It focuses on extracting pure sine waves from a composite signal corrupted by Gaussian noise. Three architectures are compared: Fully Connected (FC), Recurrent Neural Networks (RNN), and Long Short-Term Memory (LSTM).

## 🚀 Engineering Standards (V3NEW)
*   **SDK Architecture:** Core logic is strictly encapsulated.
*   **Quality:** 0 Ruff violations, <150 lines per file.
*   **Testing:** TDD with >85% coverage via `pytest`.
*   **Tooling:** Managed by `uv`.

## 📊 Results & Analysis
*(This section will be populated after training)*

### Performance Comparison
| Architecture | Test MSE | Training Time | Reconstruction Quality |
|--------------|----------|---------------|------------------------|
| FC           | TBD      | TBD           | TBD                    |
| RNN          | TBD      | TBD           | TBD                    |
| LSTM         | TBD      | TBD           | TBD                    |

### Sensitivity Analysis
Detailed graphs showing the correlation between noise intensity ($\alpha, \beta$) and reconstruction error are available in the `notebooks/` directory and documented here.

### Architecture Insights
*   **RNN vs. LSTM:** [Analysis of temporal context handling]
*   **FC Baseline:** [Analysis of why FC fails/succeeds on static windows]
*   **Failure Modes:** [Discussion of extreme noise scenarios]

## 🛠 Installation & Usage
### Prerequisites
*   `uv` package manager installed.

### Setup
```bash
uv sync
```

### Run Tests
```bash
uv run pytest --cov=src
```

### Run Training
```bash
uv run python main.py
```

## 📈 Parameter Justification
*   **Frequencies:** 5Hz, 10Hz, 15Hz, 20Hz chosen to avoid aliasing and overlap within the 10ms window.
*   **Noise:** Gaussian distribution selected for its standard representation of thermal and environmental noise.
*   **Architectures:** [Justification for layers/neurons based on experiments]

## 📄 Submission Info
*   **Name:** [User Name]
*   **ID:** [User ID]
*   **Repo:** [Link]
*   **Format:** Final submission as PDF.
