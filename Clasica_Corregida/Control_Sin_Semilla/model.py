import tensorflow as tf


def build_conv_autoencoder(
    input_shape=(28, 28, 1),
    target_hw=(2, 2),
    learning_rate: float = 1e-3,
):
    
    th, tw = target_hw

    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.ZeroPadding2D(2),  # 28 -> 32
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2, padding="same"),  # 16
            tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2, padding="same"),  # 8
            tf.keras.layers.Conv2D(1, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2, padding="same"),  # 4x4x1
            tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (th, tw)), name="latent_space"),
        ],
        name=f"conv_encoder_{th}x{tw}",
    )

    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(th, tw, 1)),
            tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (4, 4))),
            tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(size=2),  # 8
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(size=2),  # 16
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(size=2),  # 32
            tf.keras.layers.Cropping2D(2),  # 28
            tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same"),
        ],
        name=f"conv_decoder_{th}x{tw}",
    )

    encoder_input = tf.keras.Input(shape=input_shape, name="ae_input")
    autoencoder_output = decoder(encoder(encoder_input))

    autoencoder = tf.keras.Model(
        encoder_input,
        autoencoder_output,
        name=f"conv_autoencoder_{th}x{tw}",
    )

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
    )

    return autoencoder, encoder


def build_cnn_classifier(
    input_shape,
    conv_blocks: int,
    filters: list[int],
    kernel_size: int,
    learning_rate: float = 1e-3,
):
    if conv_blocks < 1:
        raise ValueError("conv_blocks debe ser >= 1")

    if len(filters) < conv_blocks:
        raise ValueError("filters debe tener al menos tantos valores como conv_blocks")

    inputs = tf.keras.Input(shape=input_shape, name="classifier_input")
    x = inputs

    for i in range(conv_blocks):
        x = tf.keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
            name=f"conv_{i + 1}",
        )(x)

        x = tf.keras.layers.MaxPooling2D(
            pool_size=2,
            padding="same",
            name=f"pool_{i + 1}",
        )(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="output",
    )(x)

    model = tf.keras.Model(inputs, outputs, name="cnn_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model