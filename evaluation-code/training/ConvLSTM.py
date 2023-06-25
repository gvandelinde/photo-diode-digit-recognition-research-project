from .config import *
import tensorflow as tf
import keras

class ConvLSTM:
    @staticmethod
    def get_model_one_conv_framesize_5(units: int = 128, input_shape: tuple = None, reshape_size: tuple = None, lr = None):
        model = tf.keras.Sequential()

        model.add(keras.layers.TimeDistributed(keras.layers.Reshape((5, 3, 1))))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, 2, activation='relu', input_shape=(5, 3, 1))))
        model.add(keras.layers.Dropout(0.25))

        # Output?
        model.add(keras.layers.Reshape((40, 4 * 2 * 128)))
        model.add(keras.layers.LSTM(units))

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

    @staticmethod
    def get_model_two_conv_framesize_5(units: int = 128, lr = None):
        model = tf.keras.Sequential()

        model.add(keras.layers.TimeDistributed(keras.layers.Reshape((5, 3, 1))))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, 2, activation='relu', input_shape=(40, 5, 3, 1))))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, 2, activation='relu', input_shape=(40, 5, 3, 1))))
        model.add(keras.layers.Dropout(0.25))
        # Output?
        model.add(keras.layers.Reshape((40, 3 * 1 * 128)))
        model.add(keras.layers.LSTM(units))
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
    
    


    @staticmethod
    def get_model_with_pooling(units: int = 128, lr = None):
        model = tf.keras.Sequential()

        model.add(keras.layers.TimeDistributed(keras.layers.Reshape((5, 3, 1))))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, 2, activation='relu', input_shape=(5, 3, 1))))
        model.add(keras.layers.Dropout(0.25))

        # Take the MaxPooling of that
        model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D()))
        model.add(keras.layers.Dropout(0.25))


        # Output?
        model.add(keras.layers.Reshape((40, 2 * 1 * 128)))
        model.add(keras.layers.LSTM(units))
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
            # default: 0.001
            #TODO: Play around with this
            # learn_rate = 0.0001
        )
        return model

class ConvLSTM_horizontal:
    @staticmethod
    def get_model_one_layer_framesize_5(units: int = 128, input_shape: tuple = None, reshape_size: tuple = None, lr = None):
        model = tf.keras.Sequential()

        model.add(keras.layers.TimeDistributed(keras.layers.Reshape((5, 3, 1))))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, (3, 1), activation='relu', input_shape=(5, 3, 1))))
        model.add(keras.layers.Dropout(0.25))
        # Output shape = (40, 3, 3, 128)

        # LSTM
        model.add(keras.layers.Reshape((40, 3* 3* 128)))
        model.add(keras.layers.LSTM(units))

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

    @staticmethod
    def get_model_two_layers_framesize_5(units: int = 128, lr = None):
        model = tf.keras.Sequential()

        model.add(keras.layers.TimeDistributed(keras.layers.Reshape((5, 3, 1))))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, (3,1), activation='relu', input_shape=(5, 3, 1))))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, 2, activation='relu', input_shape=(40, 128, 3, 3))))
        # model.add(keras.layers.Dropout(0.25))
        # Output?
        model.add(keras.layers.Reshape((40, 2 * 2 * 128)))
        model.add(keras.layers.LSTM(units))
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

    @staticmethod
    def complex_model(units: int = 128, lr = None):
        model = tf.keras.Sequential()

        model.add(keras.layers.TimeDistributed(keras.layers.Reshape((5, 3, 1))))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, (3, 1), activation='relu', input_shape=(5, 3, 1))))
        model.add(keras.layers.Dropout(0.25))
        # Output shape = (40, 3, 3, 128)

        # Convoluiton
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, 2, activation='relu', input_shape=(3, 3, 128))))
        model.add(keras.layers.Dropout(0.25))
        # Output shape = (40, 2, 2, 128)

        # Take the MaxPooling of that
        model.add(keras.layers.TimeDistributed(keras.layers.MaxPooling2D()))
        model.add(keras.layers.Dropout(0.25))
        # Output shape = (40, 1, 1, 128)

        # Output?
        model.add(keras.layers.Reshape((40, 128)))
        model.add(keras.layers.LSTM(units))

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
        
    @staticmethod
    def get_model_two_layers_extra_dense_framesize_5(units, lr = None):
        model = tf.keras.Sequential()

        model.add(keras.layers.TimeDistributed(keras.layers.Reshape((5, 3, 1))))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, (3,1), activation='relu', input_shape=(5, 3, 1))))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.TimeDistributed(keras.layers.Conv2D(128, 2, activation='relu', input_shape=(40, 128, 3, 3))))
        model.add(keras.layers.Dropout(0.25))
        # Output?
        model.add(keras.layers.Reshape((40, 2 * 2 * 128)))
        model.add(keras.layers.LSTM(units))
        # Final dropout and dense layers are the same across all models.
        model.add(keras.layers.Dropout(0.5))

        # Extra dense layer!
        model.add(keras.layers.Dense(units))
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

