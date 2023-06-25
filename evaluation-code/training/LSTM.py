from .config import *
import tensorflow as tf
import keras


class LSTM:
    @staticmethod
    def bidirectional_model(units: int = 128, input_shape: tuple = None, reshape_size: tuple = None):
        if input_shape is None or reshape_size is None:
            raise Exception("Breh, why don't you provide an input shape...")
        model = tf.keras.Sequential()
        # Kind of flatten each frame 
        model.add(keras.layers.Reshape((reshape_size)))
        #
        # Bidirectional LSTM model
        model.add(
            keras.layers.Bidirectional(
                keras.layers.LSTM(
                    units=units,
                    input_shape=input_shape
                )
            )
        )
        model.add(keras.layers.Dropout(rate=0.5))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        # Final layer
        model.add(keras.layers.Dense(OUTPUT_LAYERS, activation='softmax'))
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['acc']
        )
        return model

    @staticmethod
    def get_model(units: int = 128, lr = None):
        model = tf.keras.Sequential()
        # Flatten each frame 
        model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
        # Add LSTM Layer
        model.add(keras.layers.LSTM(units=units, input_shape=(40, 15)))
        # Add dropout layer
        model.add(keras.layers.Dropout(rate=0.5))
        # Does this do anything?
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

    @staticmethod
    def get_model_extra_dense_layer(units: int = 128, lr = None):
        model = tf.keras.Sequential()
        # Flatten each frame 
        model.add(keras.layers.TimeDistributed(keras.layers.Flatten()))
        # Add LSTM Layer
        model.add(keras.layers.LSTM(units=units, input_shape=(40, 15)))
        # Add dropout layer
        model.add(keras.layers.Dropout(rate=0.5))
        # Does this do anything?
        model.add(keras.layers.Dense(units=units, activation='relu'))
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

    @staticmethod
    def model16(input_shape, reshape_size):
        units = 16 
        return LSTM.get_model(units, input_shape, reshape_size)

    @staticmethod
    def model32(input_shape, reshape_size):
        units = 32
        return LSTM.get_model(units, input_shape, reshape_size)
     
    @staticmethod
    def model64(input_shape, reshape_size):
        units = 64
        return LSTM.get_model(units, input_shape, reshape_size)

    @staticmethod
    def model128(input_shape, reshape_size):
        units = 128
        return LSTM.get_model(units, input_shape, reshape_size)
        