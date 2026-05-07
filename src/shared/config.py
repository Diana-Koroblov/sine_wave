"""
Centralized Configuration for Sine Wave Extraction Project.
This file contains all data generation parameters, noise settings,
and model hyperparameters to ensure consistency across the project.
"""

# --- Data Generation Parameters ---
SAMPLING_RATE = 1000  # Hz (Samples per second)
DURATION = 10  # Seconds
TOTAL_SAMPLES = SAMPLING_RATE * DURATION  # 10,000 samples

WINDOW_SIZE = 10  # Number of samples per input sequence
NUM_FREQUENCIES = 4  # Number of base sine waves
FREQUENCIES = [25, 50, 100, 150]  # Frequencies in Hz

# Distinct parameters for each wave to enhance signal realism
AMPLITUDES = [1.0, 1.2, 0.8, 1.5]
PHASES = [0.0, 0.5, 1.0, 1.5]  # In radians

# --- Noise Parameters ---
# alpha/beta are percentages relative to signal (0.1 = 10%)
# Scalar values are intentionally chosen for unified sensitivity analysis
NOISE_ALPHA = 0.1  # Amplitude noise intensity
NOISE_BETA = 0.1  # Phase noise intensity
NOISE_DISTRIBUTION = "gaussian"  # Type of noise ('gaussian' or 'uniform')

"""
--- RATIONALE FOR PARAMETER CHOICES (For README.md) ---
1. Unique Amplitudes & Phases:
   By assigning distinct amplitudes and phases to each of the four base sine waves, we
   ensure that the signals are mathematically unique and represent a more realistic
   scenario where different signal sources vary in intensity and starting time.

2. Global Scalar Noise (Alpha & Beta):
   We intentionally maintain NOISE_ALPHA and NOISE_BETA as global scalar values rather
   than per-wave parameters. This design choice enables a clear, unified sensitivity
   analysis across the entire combined signal. It allows us to observe and graph the
   correlation between total noise intensity and reconstruction error without the
   confounding complexity of varying noise levels across individual components.
"""

# --- Dataset Specification ---
DATASET_SIZE = 60000
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# --- Model & Training Hyperparameters ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 50
SENSITIVITY_DATASET_SIZE = 12000
SENSITIVITY_BATCH_SIZE = 256
SENSITIVITY_EPOCHS = 12

HIDDEN_SIZE = 64  # Number of units in hidden layers
NUM_LAYERS = 2  # Number of layers for RNN/LSTM/FC

# Input size: 4 (One-Hot) + WINDOW_SIZE (10) = 14
INPUT_SIZE = NUM_FREQUENCIES + WINDOW_SIZE
# Output size: WINDOW_SIZE (10)
OUTPUT_SIZE = WINDOW_SIZE

# --- File Paths ---
MODEL_SAVE_PATH = "models/"
ASSETS_PATH = "assets/v2_high_freq/"
