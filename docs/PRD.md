# Product Requirements Document (PRD): Sine Wave Extraction Comparison

## 1. Overview & Objective
The objective of this project is to evaluate and compare the performance of three neural network architectures—**Recurrent Neural Networks (RNN)**, **Long Short-Term Memory (LSTM)**, and **Fully Connected (FC)** networks—in the task of signal de-noising and source separation. Specifically, the models will be trained to extract a specific "pure" sine wave from a composite signal consisting of four noisy sine waves summed together.

## 2. Dataset Generation Requirements
The dataset will be synthetically generated to provide ground-truth "pure" signals for evaluation.

### 2.1 Base Sine Waves
Four base sine waves ($S_i$) are defined by the formula:
$$S_i(t) = A \cdot \sin(2\pi f_i t + \theta_i)$$
*   **Frequencies ($f$):** User-defined constants (e.g., 5Hz, 10Hz, 15Hz, 20Hz).
*   **Amplitude ($A$):** User-defined base amplitude.
*   **Phase ($\theta$):** Initial phase, which can be fixed or randomized.

### 2.2 Noise Formulation
Noise is added to each individual wave before summation to simulate real-world signal interference.
$$S'_i(t) = (A \pm \alpha \cdot \text{noise}) \cdot \sin(2\pi f_i t + (\theta \pm \beta \cdot \text{noise}))$$
*   **Noise Intensity ($\alpha, \beta$):** Percentage relative to the signal (e.g., 10%).
*   **Noise Distribution:** User-selectable (e.g., Uniform, Gaussian).
*   **Phase Noise:** Random phase shift in the range $[0, 2\pi]$.

### 2.3 Sampling & Constraints
*   **Duration:** strictly 10 seconds.
*   **Sampling Rate:** 1000Hz (Total of 10,000 samples per wave).
*   **Window Size:** strictly 10 consecutive samples.
*   **Combined Signal ($\Sigma_{\text{noise}}$):** $\sum_{i=1}^{4} S'_i(t)$.
*   **Data Exports:** The generator must provide Pure sine waves ($S_i$), individual noisy waves ($S'_i$), and the combined noisy signal ($\Sigma_{\text{noise}}$).

## 3. System Architecture & Configuration

### 3.1 Centralized Configuration (`config.py`)
All parameters MUST be managed from a single `config.py` file. Hardcoded "magic numbers" are strictly prohibited in the logic files.
*   **Data Parameters:** Frequencies, amplitude, noise intensity/distribution, sampling rates.
*   **Model Hyperparameters:** Number of layers, neurons per layer, activation functions, learning rate, batch size, and epochs.

### 3.2 Modularity and Coding Standards
*   **File Length:** Every file must be strictly under **150 lines** to ensure readability and support unit testing.
*   **Structure:** Logic must be partitioned into clear modules (e.g., `data_gen.py`, `models.py`, `trainer.py`, `utils.py`).

### 3.3 Unit Testing
A comprehensive testing suite (e.g., `pytest` or `unittest`) is mandatory.
*   **Core Modules:** Data generation, sequence window extraction, and model initialization/forward passes must have corresponding test cases.
*   **Verification:** Tests must verify signal shapes, one-hot encoding consistency, and that noise levels are applied correctly.

## 4. Training Example Structure
Training data must be structured to allow the model to learn the extraction of a specific target frequency based on a control input.

### 4.1 Target Selection ($C$)
A **One-Hot Encoded** vector of length 4.
*   Example: `[0, 0, 1, 0]` signals the model to extract the 3rd sine wave ($S_3$).

### 4.2 Input ($X_{\text{input}}$)
The input is a concatenation of:
1.  The One-Hot vector ($C$).
2.  A window of $N=10$ consecutive samples from the combined noisy signal ($\Sigma_{\text{noise}}$).

### 4.3 Target Output ($Y_{\text{true}}$)
A window of $N=10$ consecutive samples from the **pure** sine wave ($S_i$), corresponding to the exact same time frame as the input window.

## 5. Model Training Requirements
The system must support the implementation and training of three distinct architectures.

### 5.1 Architectures
*   **Fully Connected (FC):** Standard multi-layer perceptron.
*   **Simple RNN:** For processing the temporal sequence of the 10-sample window.
*   **LSTM:** To capture long-term dependencies within the signal window.

## 6. Deliverables & Evaluation (README)
A comprehensive `README.md` is required, including:

### 6.1 Performance Analysis
*   **Tabulated Comparison:** Error metrics (MSE/MAE) across all three architectures.
*   **Visualizations:** Comparative plots of "Pure" vs "Combined" vs "Reconstructed" signals for various frequencies.

### 6.2 Edge Cases & Failure Analysis
*   **Extreme Noise:** Document model behavior and reconstruction failure when noise intensity exceeds 80% of signal amplitude ($\alpha > 0.8$).
*   **Architectural Limitations:** A detailed explanation of why the FC network struggles with the 10-sample window compared to RNN/LSTM, focusing on the lack of internal state and temporal context processing.
*   **Noise vs. Error Correlation:** Analysis of how increasing $\alpha$ and $\beta$ impacts the reconstruction quality across different architectures.

### 6.3 Parameter Justification & Physical Context
* **Frequency Selection:** Explicit justification for the chosen frequencies ($f$) in relation to the sampling rate (1000Hz) and the highly constrained 10-sample window (10ms). The README must explain how the ratio between the wave's period and the window size affects the models' ability to extract temporal patterns.
* **Phase & Amplitude:** Explanation of why specific baseline values were chosen and how phase randomization impacts the training distribution.