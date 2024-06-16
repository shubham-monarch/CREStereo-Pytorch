#! /usr/bin/env python

import coloredlogs, logging
import tensorrt as trt
import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import cv2
# ssimport pycuda.driver as cuda


# TO-DO
# - check tensort version compatibility
# - using polygraphy for quick onnx model checking
# - refer official nvidia docs for best practices
# - explore onnx-graphsurgeon and polygraphy 
# - checck pip install trt-cu11
# - tinker with cudnn version 
# - change cuda version to 12.1 
# - read onnxsim => removes unsupported operations -> both python / cli apis
# - how to specify precision
# - intergrate onnxsimp api to code


# (H, W)
DIMS = (480, 640)
H,W = DIMS
ZED_IMAGE_DIR = "zed_input/images"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
runtime = trt.Runtime(TRT_LOGGER)


def build_engine(onnx_file_path):

	with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
		config = builder.create_builder_config()
		config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
		with open(onnx_file_path, 'rb') as model:
			if not parser.parse(model.read()):
				print ('ERROR: Failed to parse the ONNX file.')
				for error in range(parser.num_errors):
					print (parser.get_error(error))

		serialized_engine = builder.build_serialized_network(network, config)
		
		logging.error(f"serialized_engine_is_null: {serialized_engine is None}")
		logging.error(f"config is null: {config is None}")
		logging.error(f"network is null: {network is None}")

		engine = runtime.deserialize_cuda_engine(serialized_engine)
		return engine

	
def main():
	# engine = build_engine("models/crestereo_without_flow.onnx")
	engine = build_engine("models/crestereo_without_flow_simp.onnx")
	# engine = build_engine("models/crestereo_dynamic.onnx")	
	# serialize_engine_to_file(engine, args.savepth)



if __name__ == '__main__':
	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	main()