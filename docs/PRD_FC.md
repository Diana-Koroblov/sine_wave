# Mechanism PRD: Fully Connected (FC) Architecture for Signal De-noising

## 1. Project Overview (FC Baseline)
The Fully Connected (FC), or Dense, model serves as the foundational baseline for the Signal De-noising project. Its primary objective is to reconstruct a pure 10-sample sine wave window from a 14-element input vector (4-bit One-Hot context + 10-sample noisy window). While the FC architecture lacks the explicit temporal recurrence of RNNs or LSTMs, it provides a high-capacity non-linear mapping capability that is essential for benchmarking more complex sequential models.

## 2. Theoretical Foundation (Universal Approximation & Dense Layers)
### 2.1 Universal Approximation Theorem
The selection of a Fully Connected architecture is grounded in the **Universal Approximation Theorem**. This theorem states that a feed-forward network with a single hidden layer and a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$, given a non-linear activation function. In this project, the FC model approximates the de-noising function $f: \Sigma_{\text{noise}} \to S_{\text{pure}}$.

### 2.2 Dense Connectivity
The model utilizes a "Dense" structure where every neuron in layer $L$ is connected to every neuron in layer $L+1$. Each connection is governed by a Weight ($W$) and each neuron by a Bias ($b$). The output of a layer is defined by the linear transformation $z = Wx + b$, followed by a non-linear activation function $\sigma(z)$.

## 3. Technical Specifications
### 3.1 Dimensions & Layer Stack
*   **Input Layer:** 14 units (4 units for the One-Hot Target $C$ concatenated with 10 units for the signal window).
*   **Hidden Layers:** Configurable depth and width (defaulting to 2 layers of 64 neurons).
*   **Output Layer:** 10 units representing the reconstructed 10-sample pure signal window.

### 3.2 Activations & Logic
*   **Hidden Activations:** Mandatory use of **ReLU** (Rectified Linear Unit) or **Tanh**. ReLU is preferred for its computational efficiency and its ability to mitigate vanishing gradients in deep feed-forward stacks.
*   **Static Processing:** Unlike recurrent models, the FC network processes the 10-sample window as a **static vector**. It ignores the explicit temporal order, relying instead on its ability to learn global correlations within the input feature space.

### 3.3 Engineering Constraints
*   **Inheritance:** Must inherit from `BaseModel` to centralize shared logic for weight initialization, validation, and metadata tracking.
*   **File Limit:** The implementation (`src/models/fc.py`) must strictly adhere to the **150-line limit**.

## 4. Acceptance Criteria
### 4.1 Performance Goals
*   **Loss Function:** Mandatory use of **Mean Squared Error (MSE)** to measure the squared difference between the predicted window and the ground truth $Y_{\text{true}}$.
*   **Benchmark Role:** The FC model's performance will serve as the "Floor" metric; RNN and LSTM models are expected to outperform the FC baseline by exploiting temporal periodicity.

### 4.2 Quality Gates
*   **Linter Compliance:** Exactly **0 Ruff violations**.
*   **Test Coverage:** Minimum **85% unit test coverage** for all forward pass and weight initialization logic.

## 5. Maintenance (Hyperparameter Tuning through Config)
*   **Zero Hardcoding:** All architecture-specific hyperparameters (e.g., `HIDDEN_SIZE`, `NUM_LAYERS`) must be loaded from `config.py`.
*   **Modularity:** The model must be easily swappable within the `SignalDenoiserSDK` to allow for rapid iterative testing and sensitivity analysis.
