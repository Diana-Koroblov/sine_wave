# Mechanism PRD: LSTM Architecture for Signal De-noising

## 1. Project Overview (Advanced Sequential Modeling)
The Long Short-Term Memory (LSTM) model represents the most advanced architecture in the Signal De-noising project. Its primary function is to reconstruct pure sine wave sequences from highly corrupted inputs by leveraging long-term temporal dependencies. Unlike standard RNNs, the LSTM is engineered to maintain phase coherence across extended sampling windows, making it the primary candidate for de-noising in extreme environment scenarios.

## 2. Theoretical Foundation (Gating Mechanism & Cell State)
### 2.1 Overcoming the Gradient Crisis
The LSTM is specifically designed to mitigate the vanishing and exploding gradient problems (the "Gradient Crisis") inherent in standard recurrent structures. It achieves this through a "Gradient Highway" established by the **Cell State**, which allows gradients to flow through long sequences with minimal attenuation.

### 2.2 The Two-State System
*   **Cell State ($C_t$):** Acts as a long-term memory conveyor belt, storing information across the entire sequence.
*   **Hidden State ($h_t$):** Represents short-term memory and serves as the immediate output of the unit at a given time step.

### 2.3 Triple-Gate Logic
The LSTM regulates information flow via three specialized gates:
1.  **Forget Gate ($f_t$):** Determines which information from the previous cell state is redundant and should be discarded using a sigmoid activation.
2.  **Input Gate ($i_t$):** Identifies which new information from the current input is relevant to be stored in the cell state.
3.  **Output Gate ($o_t$):** Decides which parts of the updated cell state will be filtered into the hidden state for the current output.

## 3. Technical Specifications
### 3.1 Dimensions & Flow
*   **Input Vector:** 15 elements (4-bit One-Hot context $C$ + 1 sigma feature + 10 noisy samples).
*   **Output Vector:** 10 elements (Predicted pure sine wave window).
*   **Internal Mapping:** The 10 samples are processed as a temporal sequence of 10 steps, each receiving the static One-Hot context and sigma feature.

### 3.2 Activation Functions
*   **Gate Activations:** Mandatory use of **Sigmoid** to produce gating values between 0 (completely blocked) and 1 (completely open).
*   **State/Output Activations:** Mandatory use of **Tanh** for the cell state candidate and the final output mapping to stabilize values between -1 and 1 and prevent numerical explosion.

### 3.3 Engineering Constraints
*   **Inheritance:** Must inherit from `BaseModel` to ensure architectural consistency and adherence to the **DRY** principle.
*   **File Limit:** The Python module (`src/models/lstm.py`) must be strictly under **150 lines**.

## 4. Acceptance Criteria
### 4.1 Performance Benchmarks
*   **Loss Function:** Mandatory use of **Mean Squared Error (MSE)**.
*   **Comparative Goal:** The LSTM is expected to outperform both FC and RNN models in reconstruction quality when noise intensity ($\alpha, \beta$) exceeds 50%.
*   **Generalization:** The model will be trained on a **60,000-sample dataset** and evaluated strictly on the unseen **15% Test set** to prove reconstruction robustness.

### 4.2 Quality Gates
*   **Testing:** Minimum **85% unit test coverage** verifying gate logic and state transitions.
*   **Linter:** **Zero Ruff violations** permitted.

## 5. Resource Management (Cost & Performance)
### 5.1 Resource Tracking
The SDK must systematically monitor the LSTM's execution footprint:
*   **Compute Time:** Total training duration per epoch and total convergence time.
*   **Memory Usage:** Peak RAM consumption (monitored via `psutil`) to quantify the overhead of the triple-gate mechanism compared to simpler models.

### 5.2 Efficiency Justification
The README must include a cost-benefit analysis of the LSTM, justifying the increased compute cost (memory/time) through its superior reconstruction accuracy in high-noise environments.

## 6. Achieved Results (Test Set — 60,000 samples, 50 epochs)
| Metric | Value |
|--------|------:|
| MSE | 0.2632 |
| MAE | 0.3521 |
| Pearson Correlation | 0.7765 |
| Training Time | 54.4 s |
| Peak RAM | 315 MB |

**Verdict:** LSTM ranked **second** behind FC. The triple-gate mechanism improves over the vanilla RNN (higher Pearson r, lower MSE) but cannot match FC on short 10-sample windows where long-range dependencies are absent. The LSTM carries the highest memory overhead (+24 MB over FC) for a net quality deficit — the gating advantage only materialises at longer sequence lengths or significantly higher noise intensities than tested here.
