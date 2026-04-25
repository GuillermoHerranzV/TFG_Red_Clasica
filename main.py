"""Pipeline clásico CNN para MNIST binario: <5 vs >=5."""

import csv
import os
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

import config as config
from utils import (
    load_mnist_binary,
    make_subsets,
    safe_name,
    set_global_seed,
)
from metrics import evaluate_binary_classifier, save_confusion_matrix
from model import build_cnn_classifier, build_conv_autoencoder


def run():
    set_global_seed(config.RANDOM_SEED)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CM_OUT_DIR.mkdir(parents=True, exist_ok=True)

    (train_images, train_labels), (test_images, test_labels) = load_mnist_binary()

    subsets = make_subsets(
        train_images=train_images,
        train_labels=train_labels,
        test_images=test_images,
        test_labels=test_labels,
        n_train=config.N_TRAIN,
        n_test=config.N_TEST,
        seed=config.RANDOM_SEED,
    )

    X_train_small = subsets["X_train_small"]
    y_train_small = subsets["y_train_small"]
    X_test_small = subsets["X_test_small"]
    y_test_small = subsets["y_test_small"]

    print("Subsets:", X_train_small.shape, X_test_small.shape)

    write_header = not config.CSV_PATH.exists()

    with config.CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=config.CSV_FIELDNAMES)

        if write_header:
            writer.writeheader()

        for use_reduction in config.USE_REDUCTION_OPTIONS:
            latent_channel_values = config.LATENT_CHANNELS if use_reduction else [None]

            for latent_channels in latent_channel_values:
                print("\n====================================================")
                print(f"use_reduction = {use_reduction} | latent_channels = {latent_channels}")
                print("====================================================")

                tf.keras.backend.clear_session()
                set_global_seed(config.RANDOM_SEED)

                if use_reduction:
                    autoencoder, encoder = build_conv_autoencoder(
                        input_shape=X_train_small.shape[1:],
                        latent_channels=latent_channels,
                        learning_rate=config.AUTOENCODER_LEARNING_RATE,
                    )

                    autoencoder.fit(
                        X_train_small,
                        X_train_small,
                        epochs=config.AUTOENCODER_EPOCHS,
                        batch_size=config.AUTOENCODER_BATCH_SIZE,
                        shuffle=True,
                        verbose=0,
                    )

                    X_train_in = encoder.predict(X_train_small, verbose=0)
                    X_test_in = encoder.predict(X_test_small, verbose=0)

                else:
                    X_train_in = X_train_small
                    X_test_in = X_test_small

                input_shape = X_train_in.shape[1:]

                print("Input shape clasificador:", input_shape)

                for conv_blocks in config.CONV_BLOCK_OPTIONS:
                    for filters in config.FILTER_CONFIGS:
                        if len(filters) < conv_blocks:
                            continue

                        for kernel_size in config.KERNEL_SIZES:
                            print(
                                f"\n--- CNN | reduction={use_reduction} | "
                                f"latent={latent_channels} | "
                                f"blocks={conv_blocks} | "
                                f"filters={filters[:conv_blocks]} | "
                                f"kernel={kernel_size}"
                            )

                            tf.keras.backend.clear_session()
                            set_global_seed(config.RANDOM_SEED)

                            classifier = build_cnn_classifier(
                                input_shape=input_shape,
                                conv_blocks=conv_blocks,
                                filters=filters,
                                kernel_size=kernel_size,
                                learning_rate=config.CLASSIFIER_LEARNING_RATE,
                            )

                            t0 = time.perf_counter()

                            classifier.fit(
                                X_train_in,
                                y_train_small,
                                epochs=config.CLASSIFIER_EPOCHS,
                                batch_size=config.CLASSIFIER_BATCH_SIZE,
                                shuffle=True,
                                verbose=0,
                            )

                            train_seconds = time.perf_counter() - t0

                            metrics = evaluate_binary_classifier(
                                model=classifier,
                                X_test=X_test_in,
                                y_test=y_test_small,
                            )

                            cm_base = (
                                f"cnn_red{int(use_reduction)}_"
                                f"latent{latent_channels}_"
                                f"blocks{conv_blocks}_"
                                f"filters{safe_name(filters[:conv_blocks])}_"
                                f"k{kernel_size}"
                            )

                            np.savetxt(
                                config.CM_OUT_DIR / f"{cm_base}.csv",
                                metrics["cm"],
                                delimiter=",",
                                fmt="%d",
                            )

                            save_confusion_matrix(
                                metrics["cm"],
                                title=(
                                    f"CNN | red={use_reduction} | "
                                    f"latent={latent_channels} | "
                                    f"blocks={conv_blocks} | "
                                    f"filters={filters[:conv_blocks]} | "
                                    f"k={kernel_size}"
                                ),
                                output_path=config.CM_OUT_DIR / f"{cm_base}.png",
                            )

                            print(
                                f"Accuracy: {metrics['accuracy']:.4f} | "
                                f"Balanced: {metrics['balanced_accuracy']:.4f} | "
                                f"Kappa: {metrics['kappa']:.4f}"
                            )
                            print(
                                f"Precision: {metrics['precision']:.4f} | "
                                f"Recall: {metrics['recall']:.4f} | "
                                f"F1: {metrics['f1']:.4f}"
                            )
                            print(f"Train seconds: {train_seconds:.2f}")

                            row = {
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "use_reduction": bool(use_reduction),
                                "latent_channels": latent_channels,
                                "conv_blocks": int(conv_blocks),
                                "filters": str(filters[:conv_blocks]),
                                "kernel_size": int(kernel_size),
                                "n_train": int(config.N_TRAIN),
                                "n_test": int(config.N_TEST),
                                "classifier_epochs": int(config.CLASSIFIER_EPOCHS),
                                "classifier_batch_size": int(config.CLASSIFIER_BATCH_SIZE),
                                "autoencoder_epochs": (
                                    int(config.AUTOENCODER_EPOCHS)
                                    if use_reduction
                                    else None
                                ),
                                "autoencoder_batch_size": (
                                    int(config.AUTOENCODER_BATCH_SIZE)
                                    if use_reduction
                                    else None
                                ),
                                "train_seconds": float(train_seconds),
                                "accuracy": metrics["accuracy"],
                                "balanced_accuracy": metrics["balanced_accuracy"],
                                "kappa": metrics["kappa"],
                                "precision": metrics["precision"],
                                "recall": metrics["recall"],
                                "f1": metrics["f1"],
                                "tn": metrics["tn"],
                                "fp": metrics["fp"],
                                "fn": metrics["fn"],
                                "tp": metrics["tp"],
                            }

                            writer.writerow(row)

    print(f"\nSweep completado. CSV: {os.path.abspath(config.CSV_PATH)}")


if __name__ == "__main__":
    run()