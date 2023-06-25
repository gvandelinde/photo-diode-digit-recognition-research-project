import tensorflow as tf
import numpy as np 
import time
import os
from .utils import get_timestr

# Documentation used: https://www.tensorflow.org/lite/performance/post_training_quantization

def convert_and_quantize(model, dataset, model_name = "undefined_model"):
    # Convert the model to the TensorFlow Lite format without quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8, # enable TensorFlow Lite ops.
        # tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops, which will be used as a fallback for unsupported operations.
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter._experimental_lower_tensor_list_ops = False

    # Define a generator function that provides our test data's x values
    # as a representative dataset, and tell the converter to use it
    def representative_dataset_generator():
        for value in dataset:
            # Each scalar value must be inside of a 2D array that is wrapped in a list
            yield [np.array(value, dtype=np.float32, ndmin=3)]
    converter.representative_dataset = representative_dataset_generator

    # converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    file_name = f"./output/models/tflite/{model_name}_{get_timestr()}" 

    # Save the model to disk
    open(f"{file_name}.tflite", "wb").write(tflite_model)
    
    # Print size in KB.
    print(f"{len(tflite_model) / 1000} KB")

    # Save results to a file   
    with open(f"{file_name}.txt", 'w') as file:
        file.write(f"Model name: {model_name}\n")
        file.write(f"Size: {len(tflite_model) / 1000} KB\n")
    return file_name