from keras.regularizers import l2
from tensorflow import keras

def Classifier(
    n_timesteps, n_features, n_conv_layers=1, add_dense_layer=True, n_output=2
):
    # https://keras.io/examples/timeseries/timeseries_classification_from_scratch/

    input_shape = ( n_timesteps,n_features)

    inputs = keras.Input(shape=input_shape, dtype="float32")

    if add_dense_layer:
        x = keras.layers.Dense(128, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(inputs)
    else:
        x = inputs

    for i in range(n_conv_layers):
        x = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)

    x = keras.layers.MaxPooling1D(pool_size=2, padding="same")(x)
    x = keras.layers.Flatten()(x)

    if n_output >= 2:
        outputs = keras.layers.Dense(n_output,kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation="softmax")(x)
    else:
        outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    classifier = keras.models.Model(inputs=inputs, outputs=outputs)

    return classifier

def Autoencoder(n_timesteps, n_features, n_channels): 
    # n_channels = 64
    # Define encoder and decoder structure
    def Encoder(input):
        print(input.shape)
        x = keras.layers.Conv1D(
            filters=n_channels, kernel_size=3, activation="relu", padding="same"
        )(input)
        print(x.shape)
        x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
        print(x.shape)
        x = keras.layers.Conv1D(
            filters=n_channels/2, kernel_size=3, activation="relu", padding="same"
        )(x)
        print(x.shape)
        x = keras.layers.MaxPool1D(pool_size=2, padding="same")(x)
        print(x.shape)
        return x

    def Decoder(input):
        x = keras.layers.Conv1D(
            filters=n_channels/2, kernel_size=3, activation="relu", padding="same"
        )(input)
        print(x.shape)
        x = keras.layers.UpSampling1D(size=2)(x)
        print(x.shape)
        x = keras.layers.Conv1D(
            filters=n_channels, kernel_size=3, activation="relu", padding="same"
        )(x)
        print(x.shape)
        x = keras.layers.UpSampling1D(size=2)(x)
        print(x.shape)
        x = keras.layers.Conv1D(
            filters=n_features, kernel_size=3, activation="linear", padding="same"
        )(x)
        print(x.shape)
        return x

    # Define the AE model
    orig_input = keras.Input(shape=(n_timesteps, n_features))
    autoencoder = keras.Model(inputs=orig_input, outputs=Decoder(Encoder(orig_input)))

    return autoencoder
