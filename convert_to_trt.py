#! /usr/bin/env python

import coloredlogs, logging
import tensorrt as trt
import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import cv2

# TO-DO
# - check tensort version compatibility
# - refer official nvidia docs for best practices
# - explore onnx-graphsurgeon and polygraphy 
# - checck pip install trt-cu11
# - tinker with cudnn version 
# - change cuda version to 12.1 


# (H, W)
DIMS = (480, 640)
H,W = DIMS
ZED_IMAGE_DIR = "zed_input/images"

if __name__ == "__main__":
	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger

	# PREPARING INPUT DATA
	# left_img = cv2.imread("zed_input/images/left_18.png")	
	# right_img = cv2.imread("zed_input/images/right_18.png")

	# left = cv2.resize(left_img, (W, H), interpolation=cv2.INTER_LINEAR)
	# right = cv2.resize(right_img, (W, H), interpolation=cv2.INTER_LINEAR)

	# (h,w) = (H,W)

	# # Transpose the dimensions and add a batch dimension
	# imgL = cv2.resize(left_img, (w, h), interpolation=cv2.INTER_LINEAR)
	# imgR = cv2.resize(right_img, (w, h), interpolation=cv2.INTER_LINEAR)
	# imgL = np.ascontiguousarray(imgL.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	# imgR = np.ascontiguousarray(imgR.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)

	# imgL_dw2 = cv2.resize(left_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
	# imgR_dw2 = cv2.resize(right_img, (w//2, h//2),  interpolation=cv2.INTER_LINEAR)
	# imgL_dw2 = np.ascontiguousarray(imgL_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32) 
	# imgR_dw2 = np.ascontiguousarray(imgR_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	
	# input_data = [imgL_dw2, imgR_dw2]
	
	TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

	builder = trt.Builder(TRT_LOGGER)
	

	



	config = builder.create_builder_config()
	config.max_workspace_size = 1 << 50
	# Set the optimization precision to float16
	config.flags |= 1 << int(trt.BuilderFlag.FP16) 
	# Set the default device type to GPU
	config.default_device_type = trt.DeviceType.GPU

	# Enable explicit batch mode for dynamic input shapes
	EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH) #we have enabled the explicit Batch
	# Create an empty network definition with explicit batch mode
	network = builder.create_network(EXPLICIT_BATCH)

	# ONNX_FILE_PATH = "models/crestereo_dynamic.onnx"
	# ONNX_FILE_PATH = "models/crestereo_without_flow_dynamic.onnx"
	ONNX_FILE_PATH = "models/crestereo.onnx"
	# ONNX_FILE_PATH = "models/crestereo_without_flow.onnx"

	parser = trt.OnnxParser(network, TRT_LOGGER)
	with open(ONNX_FILE_PATH, "rb") as f:
	# Parse the ONNX model into the network definition
		if not parser.parse(f.read()):
		# Log any parsing errors
			print('ERROR: Failed to parse the ONNX file.')
			logging.debug(f"num_errors: {parser.num_errors}")
			for error in range(parser.num_errors):
				logging.error(parser.get_error(error))