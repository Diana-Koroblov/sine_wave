# Mechanism PRD: RNN Architecture for Signal De-noising

## 1. Project Overview (RNN Specific)
The Recurrent Neural Network (RNN) component is designed to extract pure sine wave windows from a composite noisy signal ($\Sigma_{\text{noise}}$). Unlike standard feed-forward networks, the RNN is specifically tasked with exploiting the temporal dependencies inherent in periodic signals. By maintaining an internal memory (hidden state), the RNN can contextualize the current 10-sample window relative to its internal temporal dynamics, providing a specialized baseline for sequential signal reconstruction.

## 2. Theoretical Background (Feedback & Unrolling)
### 2.1 Recurrent Hidden Layers
The RNN architecture incorporates a recurrent hidden layer where each neuron maintains an internal feedback loop. This hidden state $h_t$ acts as a summary of historical information within the current processing sequence.

### 2.2 Weight Sharing Principle
To ensure efficient learning and temporal consistency, the RNN must adhere to the **Weight Sharing** principle. The same set of weights ($W_x$ for input-to-hidden, $W_h$ for hidden-to-hidden, and $b$ for bias) is reused across all time steps. This reduces the parameter count and allows the model to generalize patterns regardless of their position in the sequence.

### 2.3 Unrolling Execution
During training, the "folded" representation of the RNN is "unrolled" through time. This transformation allows the backpropagation algorithm (Backpropagation Through Time - BPTT) to calculate gradients by treating the RNN as a deep feed-forward network with shared weights across layers.

## 3. Technical Specifications
### 3.1 Dimensions & Mapping
*   **Input Vector ($X_{\text{input}}$):** 15 elements (4 for One-Hot Target $C$ + 1 sigma feature + 10 noisy samples).
*   **Output Vector ($Y_{\text{pred}}$):** 10 elements (Reconstructed pure sine wave window).
*   **Sequential Processing:** The 10 noisy samples are processed as a time series, while the One-Hot vector $C$ and sigma feature provide static context for target wave selection.

### 3.2 Layers & Activations
*   **Recurrent Layer:** Minimum of one hidden layer with a configurable number of neurons (defaulting to 64).
*   **Activation Function:** Mandatory use of **Tanh** for the hidden recurrent layer. Tanh ensures zero-centered mapping ($[-1, 1]$), which helps stabilize gradients and prevent saturation during the unrolling process.
*   **Output Layer:** Linear layer mapping the final hidden state or sequence output to the 10-sample target.

### 3.3 Engineering Constraints
*   **Inheritance:** The implementation must inherit from `BaseModel` to enforce the **DRY** (Don't Repeat Yourself) principle and ensure consistent API contracts.
*   **File Limit:** The Python implementation (`src/models/rnn.py`) must be strictly under **150 lines**.

## 4. Acceptance Criteria
### 4.1 Performance Goals
*   **Loss Function:** Mandatory use of **Mean Squared Error (MSE)** comparing $Y_{\text{pred}}$ to the ground truth pure window $Y_{\text{true}}$.
*   **Accuracy:** The RNN must demonstrate superior phase coherence compared to the FC baseline in noise levels up to 50%.
*   **Generalization:** The model will be trained on a **60,000-sample dataset** and evaluated strictly on the unseen **15% Test set** to prove reconstruction robustness.

### 4.2 Quality Gates
*   **Code Quality:** Exactly **0 Ruff violations**.
*   **Test Coverage:** Minimum **85% unit test coverage** for the RNN module logic.
*   **Verification:** Successful forward pass verification against the configured input size and 10-out contract.

## 5. Scalability & Maintenance
*   **Building Blocks Design:** The RNN is designed as a modular component within the `SignalDenoiserSDK`. 
*   **Dynamic Hyperparameters:** All RNN-specific parameters (hidden size, number of layers) must be loaded from `config.py`, allowing for seamless sensitivity analysis and hyperparameter tuning without modifying the core model logic.

## 6. Achieved Results (Test Set — 60,000 samples, 50 epochs)
| Metric | Value |
|--------|------:|
| MSE | 0.2815 |
| MAE | 0.3759 |
| Pearson Correlation | 0.7588 |
| Training Time | 53.2 s |
| Peak RAM | 297 MB |

**Verdict:** RNN showed the **weakest reconstruction quality** of the three architectures (lowest Pearson r, highest MSE). The vanishing gradient problem limits its ability to maintain phase coherence across the 10-sample window at moderate-to-high noise levels, consistent with the theoretical expectation. Training cost is ~4× the FC baseline with no quality benefit.
