#! /usr/bin/env python3

import onnx
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model = onnx.load('crestereo.onnx')

# Convert the ONNX model to TensorFlow
tf_rep = prepare(onnx_model)

# Export the TensorFlow model
tf_rep.export_graph('example_tf')
