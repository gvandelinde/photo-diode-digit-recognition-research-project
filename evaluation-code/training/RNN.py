from .config import *
import tensorflow as tf
import keras

class RNN:
    @staticmethod
    def get_model_framesize_5(units, lr = None):
        model = tf.keras.Sequential()

        # Flatten each frame 
        model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))

        # Add LSTM Layer
        model.add(keras.layers.SimpleRNN(units=units, input_shape=(40, 15)))
        model.add(keras.layers.Dropout(rate=0.5))

        # Into final classification layer
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(OUTPUT_LAYERS, activation='softmax'))

        # Use custom learning rate
        if lr is None:
            opt = 'adam'
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=lr)

        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['acc'],
        )

        return model