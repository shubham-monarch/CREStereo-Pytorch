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


# (H, W)
DIMS = (480, 640)
H,W = DIMS
ZED_IMAGE_DIR = "zed_input/images"

if __name__ == "__main__":
	coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger

	model = onnx.load("models/crestereo.onnx")
	engine = backend.prepare(model, device='CUDA')
	
	left_img = cv2.imread("zed_input/images/left_18.png")	
	right_img = cv2.imread("zed_input/images/right_18.png")

	left = cv2.resize(left_img, (W, H), interpolation=cv2.INTER_LINEAR)
	right = cv2.resize(right_img, (W, H), interpolation=cv2.INTER_LINEAR)

	(h,w) = (H,W)

	# Transpose the dimensions and add a batch dimension
	imgL = cv2.resize(left_img, (w, h), interpolation=cv2.INTER_LINEAR)
	imgR = cv2.resize(right_img, (w, h), interpolation=cv2.INTER_LINEAR)
	imgL = np.ascontiguousarray(imgL.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	imgR = np.ascontiguousarray(imgR.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)

	imgL_dw2 = cv2.resize(left_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
	imgR_dw2 = cv2.resize(right_img, (w//2, h//2),  interpolation=cv2.INTER_LINEAR)
	imgL_dw2 = np.ascontiguousarray(imgL_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32) 
	imgR_dw2 = np.ascontiguousarray(imgR_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	
	
	# input_data = np.random.random(size=(32, 3, 224, 224)).astype(np.float32)
	input_data = [imgL, imgR]
	
	
	output_data = engine.run(input_data)[0]
	print(output_data)
	print(output_data.shape)
