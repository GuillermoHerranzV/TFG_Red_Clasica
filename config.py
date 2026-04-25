from pathlib import Path

RANDOM_SEED = 12345

N_TRAIN = 700
N_TEST = 300

CLASSIFIER_EPOCHS = 12
CLASSIFIER_BATCH_SIZE = 32
CLASSIFIER_LEARNING_RATE = 1e-3

AUTOENCODER_EPOCHS = 10
AUTOENCODER_BATCH_SIZE = 32
AUTOENCODER_LEARNING_RATE = 1e-3

CONV_BLOCK_OPTIONS = [1, 2]

FILTER_CONFIGS = [
    [16],
    [32],
    [32, 64],
]

KERNEL_SIZES = [3, 5]

USE_REDUCTION_OPTIONS = [False, True]

LATENT_CHANNELS = [2, 4, 8]

OUT_DIR = Path("cnn_clasica_outputs")
CM_OUT_DIR = OUT_DIR / "matrices_confusion"
CSV_PATH = OUT_DIR / "metrics_cnn_clasica_mnist_binario.csv"

CSV_FIELDNAMES = [
    "timestamp",
    "use_reduction",
    "latent_channels",
    "conv_blocks",
    "filters",
    "kernel_size",
    "n_train",
    "n_test",
    "classifier_epochs",
    "classifier_batch_size",
    "autoencoder_epochs",
    "autoencoder_batch_size",
    "train_seconds",
    "accuracy",
    "balanced_accuracy",
    "kappa",
    "precision",
    "recall",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
]