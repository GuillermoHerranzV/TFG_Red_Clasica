import tensorflow as tf


def build_conv_autoencoder(
    input_shape=(28, 28, 1),
    latent_channels: int = 8,
    learning_rate: float = 1e-3,
):
    encoder_input = tf.keras.Input(shape=input_shape, name="ae_input")

    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(encoder_input)
    x = tf.keras.layers.MaxPooling2D(pool_size=2, padding="same")(x)

    x = tf.keras.layers.Conv2D(latent_channels, 3, activation="relu", padding="same")(x)
    encoded = tf.keras.layers.MaxPooling2D(
        pool_size=2,
        padding="same",
        name="latent_space",
    )(x)

    encoder = tf.keras.Model(
        encoder_input,
        encoded,
        name=f"conv_encoder_{latent_channels}",
    )

    decoder_input = tf.keras.Input(shape=encoder.output_shape[1:], name="decoder_input")

    x = tf.keras.layers.Conv2D(latent_channels, 3, activation="relu", padding="same")(decoder_input)
    x = tf.keras.layers.UpSampling2D(size=2)(x)

    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.UpSampling2D(size=2)(x)

    decoded = tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")(x)

    decoder = tf.keras.Model(
        decoder_input,
        decoded,
        name=f"conv_decoder_{latent_channels}",
    )

    autoencoder_output = decoder(encoder(encoder_input))

    autoencoder = tf.keras.Model(
        encoder_input,
        autoencoder_output,
        name=f"conv_autoencoder_{latent_channels}",
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