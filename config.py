"""
Centralized Configuration for Sine Wave Extraction Project.
This file contains all data generation parameters, noise settings, 
and model hyperparameters to ensure consistency across the project.
"""

# --- Data Generation Parameters ---
SAMPLING_RATE = 1000  # Hz (Samples per second)
DURATION = 10         # Seconds
TOTAL_SAMPLES = SAMPLING_RATE * DURATION  # 10,000 samples

WINDOW_SIZE = 10      # Number of samples per input sequence
NUM_FREQUENCIES = 4   # Number of base sine waves
FREQUENCIES = [5, 10, 15, 20]  # Frequencies in Hz

BASE_AMPLITUDE = 1.0
BASE_PHASE = 0.0

# --- Noise Parameters ---
# alpha/beta are percentages relative to signal (0.1 = 10%)
NOISE_ALPHA = 0.1     # Amplitude noise intensity
NOISE_BETA = 0.1      # Phase noise intensity
NOISE_DISTRIBUTION = 'gaussian'  # Type of noise ('gaussian' or 'uniform')

# --- Model & Training Hyperparameters ---
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 50

HIDDEN_SIZE = 64      # Number of units in hidden layers
NUM_LAYERS = 2        # Number of layers for RNN/LSTM/FC

# Input size: 4 (One-Hot) + WINDOW_SIZE (10) = 14
INPUT_SIZE = NUM_FREQUENCIES + WINDOW_SIZE 
# Output size: WINDOW_SIZE (10)
OUTPUT_SIZE = WINDOW_SIZE

# --- File Paths ---
MODEL_SAVE_PATH = "models/"
DATA_PATH = "data/"
