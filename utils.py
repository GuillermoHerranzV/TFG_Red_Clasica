import os
import random
import re

import numpy as np
import tensorflow as tf


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_binary_labels(y: np.ndarray) -> np.ndarray:
    return (y >= 5).astype(np.int32)


def load_mnist_binary():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = (train_images / 255.0).astype(np.float32)
    test_images = (test_images / 255.0).astype(np.float32)

    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    train_labels = np.where(train_labels < 5, 0, 1)
    test_labels = np.where(test_labels < 5, 0, 1)

    return (train_images, train_labels), (test_images, test_labels)


def make_subsets(
    train_images,
    train_labels,
    test_images,
    test_labels,
    n_train: int,
    n_test: int,
    seed: int,
):
    rng = np.random.default_rng(seed)

    idx_train = rng.choice(len(train_images), size=n_train, replace=False)
    idx_test = rng.choice(len(test_images), size=n_test, replace=False)

    return {
        "X_train_small": train_images[idx_train],
        "y_train_small": train_labels[idx_train],
        "X_test_small": test_images[idx_test],
        "y_test_small": test_labels[idx_test],
    }


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))