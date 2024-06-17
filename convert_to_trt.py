#! /usr/bin/env python

import coloredlogs, logging
import tensorrt as trt
import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import cv2
import os
import trt_utils
import pycuda.driver as cuda 

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
# - add engine serialization and deserialization 
# - explore FPS() class -> YOLO -> https://github.com/sithu31296/PyTorch-ONNX-TRT/blob/master/examples/yolov4/yolov4/scripts/infer.py
# - package structure from YOLO 
# - add trt_info functions -> https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer#workflow
# - add more attributes to the engine configuration
# - add pre / post processing
# - add trt engine class
# - make onxx_without_flow work without onnx-simplifier
# - addded batch_inference


# (H, W)
DIMS = (480, 640)
H,W = DIMS
ZED_IMAGE_DIR = "zed_input/images"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
runtime = trt.Runtime(TRT_LOGGER)


# def do_inference_v2(context, bindings, inputs, outputs, stream):
# 	# Transfer input data to the GPU.
# 	[cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
# 	# Run inference.
# 	context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
# 	# Transfer predictions back from the GPU.
# 	[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
# 	# Synchronize the stream
# 	stream.synchronize()
# 	# Return only the host outputs.
# 	return [out.host for out in outputs]

def load_engine(model_path: str):
	if os.path.exists(model_path) and model_path.endswith('trt'):
		print(f"Reading engine from file {model_path}")
		with open(model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
			return runtime.deserialize_cuda_engine(f.read())
	else:
		print(f"FILE: {model_path} not found or extension not supported.")




def generate_engine_from_onnx(onnx_file_path: str):
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
		
		engine_file_path = onnx_file_path.replace(".onnx", ".trt")
		with open(engine_file_path, "wb") as f:
			f.write(engine.serialize())

		return engine

def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v3(stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
	

def main():
	# engine = build_engine("models/crestereo_without_flow.onnx")
	logging.debug(f"TensortRT version: {trt.__version__}")
	# onnx_model = "models/crestereo_without_flow.onnx"
	# onnx_model = "models/crestereo_without_flow_simp.onnx"
	onnx_model = "models/crestereo.onnx"
	# engine = generate_engine_from_onnx(onnx_model)
	trt.init_libnvinfer_plugins(None, "")
	engine = load_engine("models/crestereo.trt")
	
	# trt_utils.allocate_buffers(engine)
	# context = engine.create_execution_context()
	# inputs, outputs, bindings, stream = trt_utils.allocate_buffers(engine)
	
	# logging.debug(f"type(inputs[0]): {type(inputs[0])}")
	# # logging.debug(f"dir(inputs[0]): {dir(inputs[0])}")
	# logging.debug(f"inputs[0].host: {inputs[0].host} inputs[0].device: {inputs[0].device}")
	# logging.debug(f"type(outputs): {type(outputs)}")
	# logging.debug(f"type(bindings): {type(bindings)}")
	# logging.debug(f"type(stream): {type(stream)}")

	# # do_inference_v2(context, inputs, outputs, bindings, stream, batch_size=1)
	# do_inference_v2(context, inputs, outputs, bindings, stream)


	# # load the inputs
	# # Assuming left_img and right_img are your input images
	# left_img = cv2.imread(f"{ZED_IMAGE_DIR}/left_18.png")
	# right_img = cv2.imread(f"{ZED_IMAGE_DIR}/right_18.png")

	# (w ,h) = (W, H)
	# imgL_dw2 = cv2.resize(left_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
	# imgR_dw2 = cv2.resize(right_img, (w//2, h//2),  interpolation=cv2.INTER_LINEAR)
	# imgL_dw2 = np.ascontiguousarray(imgL_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32) 
	# imgR_dw2 = np.ascontiguousarray(imgR_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)

	# inputs[0].host = imgL_dw2
	# inputs[1].host = imgR_dw2
	
	# # inputs[0].host = np.random.random_sample(inputs[0].host.shape).astype(np.float32)
	# # inputs[1].host = np.random.random_sample(inputs[1].host.shape).astype(np.float32)
	
	# output = do_inference_v2(context, bindings, inputs, outputs, stream)



if __name__ == '__main__':
	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	main()