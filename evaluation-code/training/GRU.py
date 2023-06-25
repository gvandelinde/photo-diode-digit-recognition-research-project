from .config import *
import tensorflow as tf
import keras

class GRU:
    @staticmethod
    def get_model_framesize_5(units: int = 128, lr = None):
        model = tf.keras.Sequential()

        model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
        # Add GRU Layer
        model.add(
                keras.layers.GRU(
                    units=units,
                    input_shape=(40, 15)
            )
        )
        # Add dropout layer
        model.add(keras.layers.Dropout(rate=0.5))
        # model.add(keras.layers.Dense(units=units, activation='relu'))
        # Final layer
        model.add(keras.layers.Dense(OUTPUT_LAYERS, activation='softmax'))

        # Set optimizer
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