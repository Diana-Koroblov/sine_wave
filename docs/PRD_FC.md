# Mechanism PRD: Fully Connected (FC) Architecture for Signal De-noising

## 1. Overview
The Fully Connected (FC) or Multi-Layer Perceptron (MLP) mechanism serves as the baseline architecture. It treats the 14-dimensional input as a static vector, ignoring the inherent sequential nature of the time-series data.

## 2. Model Specifications
*   **Architecture Type:** Feed-forward Neural Network.
*   **Input Dimension:** 14 (4-bit One-Hot + 10-sample signal window).
*   **Output Dimension:** 10 (Reconstructed pure signal window).
*   **Layer Structure:** Multiple dense layers with non-linear activation functions.

## 3. Training Requirements
*   **Loss Function:** Mean Squared Error (MSE).
*   **Regularization:** Dropout or L2 regularization may be used if overfitting occurs on the 10,000-sample dataset.

## 4. Evaluation Metrics
*   **Primary Metric:** MSE on test set.
*   **Performance Gap:** Comparison against RNN/LSTM to quantify the benefit of temporal modeling.

## 5. Parameter Justification
(To be completed in README.md)
*   Hidden Layer Size: [User to specify]
*   Number of Layers: [User to specify]
*   Activation: [User to specify]
