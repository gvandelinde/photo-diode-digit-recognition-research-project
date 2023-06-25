import tensorflow as tf
import keras
from .config import *

class CNN:
    @staticmethod
    def get_model_framesize_5(kernels: int = 128, lr = None):
        model = tf.keras.Sequential()

        model.add(keras.layers.TimeDistributed(keras.layers.Reshape((5, 3, 1))))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(kernels, 2, activation='relu', input_shape=(40, 5, 3, 1))))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(kernels, 2, activation='relu', input_shape=(40, 5, 3, 1))))
        model.add(keras.layers.Dropout(0.25))
        # Output?
        # Final dropout and dense layers are the same across all models.
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(OUTPUT_LAYERS, activation='softmax'))
        if lr is None:
            opt = 'adam'
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['acc'],
        )
        return model