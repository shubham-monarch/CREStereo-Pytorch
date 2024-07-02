#! /usr/bin/env python3

import numpy as np
import onnxruntime as ort
import coloredlogs, logging
import matplotlib.pyplot as plt
import cv2
import utils
import os
import random
import re
import time
from tqdm import tqdm
import disparity2pcl
import open3d as o3d

# custom imports
import zed_inference
import utils_matplotlib
import trt_inference


# TO-DO -> 
# - calculate frame rate -> pytorch vs onnx vs tensorrt
# - using gpu for inference in pytorch / onnx / tensorrt
# - move data to gpu before inference
# - role of batch_size while exporting the onnx model
# - add tensorrt executionprovider => https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html

# (H, W)
# DIMS = (480, 640)
# H,W = DIMS
ZED_IMAGE_DIR = zed_inference.ZED_IMG_DIR 
ONNX_VS_PYTORCH_DIR = "onnx_vs_pytorch"
ONNX_VS_TRT_DIR = "onnx_vs_trt"


# ONNX_DISPARITY_DIR = f"{ONNX_VS_PYTORCH_DIR}/onnx_disparity"
ONNX_DISPARITY_DIR = f"{trt_inference.ONNX_VS_TRT_DIR}/onnx_disparity"
ONNX_DEPTH_MAP_DIR = f"{trt_inference.ONNX_VS_TRT_DIR}/onnx_depth_map"

FOLDERS_TO_CREATE = [ONNX_DISPARITY_DIR, ONNX_DEPTH_MAP_DIR]

# storing onxx init_flow histograms
onnx_init_flow_plts = []

def inference_with_flow(left_img, right_img, model, model_no_flow, pred_flow_dw2, img_shape=(480, 640)):	
	input1_name = model.get_inputs()[0].name
	input2_name = model.get_inputs()[1].name
	input3_name = model.get_inputs()[2].name
	output_name = model.get_outputs()[0].name
	
	(h,w) = img_shape 

	imgL = cv2.resize(left_img, (w, h), interpolation=cv2.INTER_LINEAR)
	imgR = cv2.resize(right_img, (w, h), interpolation=cv2.INTER_LINEAR)
	imgL = np.ascontiguousarray(imgL.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	imgR = np.ascontiguousarray(imgR.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	
	pred_disp = model.run([output_name], {
						  input1_name: imgL, input2_name: imgR, input3_name: pred_flow_dw2})[0]
	
	return np.squeeze(pred_disp[:, 0, :, :])

def inference_no_flow(left_img, right_img, model, model_no_flow, img_shape=(480, 640)):	
	
	input1_name = model.get_inputs()[0].name
	input2_name = model.get_inputs()[1].name
	input3_name = model.get_inputs()[2].name
	output_name = model.get_outputs()[0].name
	
	(h,w) = img_shape 

	imgL_dw2 = cv2.resize(left_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
	imgR_dw2 = cv2.resize(right_img, (w//2, h//2),  interpolation=cv2.INTER_LINEAR)
	imgL_dw2 = np.ascontiguousarray(imgL_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32) 
	imgR_dw2 = np.ascontiguousarray(imgR_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	
	pred_flow_dw2 = model_no_flow.run(
		[output_name], {input1_name: imgL_dw2, input2_name: imgR_dw2})[0]
	
	return pred_flow_dw2

def inference(left_img, right_img, model, model_no_flow, img_shape=(480, 640)):	
	
	# Get onnx model layer names (see convert_to_onnx.py for what these are)
	input1_name = model.get_inputs()[0].name
	input2_name = model.get_inputs()[1].name
	input3_name = model.get_inputs()[2].name
	output_name = model.get_outputs()[0].name
	
	(h,w) = img_shape 

	# Transpose the dimensions and add a batch dimension
	imgL = cv2.resize(left_img, (w, h), interpolation=cv2.INTER_LINEAR)
	imgR = cv2.resize(right_img, (w, h), interpolation=cv2.INTER_LINEAR)
	imgL = np.ascontiguousarray(imgL.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	imgR = np.ascontiguousarray(imgR.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	
	imgL_dw2 = cv2.resize(left_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
	imgR_dw2 = cv2.resize(right_img, (w//2, h//2),  interpolation=cv2.INTER_LINEAR)
	imgL_dw2 = np.ascontiguousarray(imgL_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32) 
	imgR_dw2 = np.ascontiguousarray(imgR_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	
	# logging.debug("Running no-flow model!")
	pred_flow_dw2 = model_no_flow.run(
		[output_name], {input1_name: imgL_dw2, input2_name: imgR_dw2})[0]
	
	# logging.warning(f"pred_flow_dw2.shape: {pred_flow_dw2.shape} pred_flow_dw2.dtype: {pred_flow_dw2.dtype}")
	# onnx_init_flow_plts.append(utils.PLT(data=pred_flow_dw2[0], 
	# 									title='onnx_init_flow',
	# 									bins=100,
	# 									range=(pred_flow_dw2.min(), pred_flow_dw2.max())))
	
	# logging.warning(f"pred_flow_dw2.shape: {pred_flow_dw2.shape} pred_flow_dw2.dtype: {pred_flow_dw2.dtype}")
	# logging.warning(f"pred_flow_dw2[0].shape: {pred_flow_dw2[0].shape}")

	# logging.debug("Running with flow model!")
	pred_disp = model.run([output_name], {
						  input1_name: imgL, input2_name: imgR, input3_name: pred_flow_dw2})[0]
	
	# logging.warning(f"output shape after inference => {pred_disp.shape}")
	return np.squeeze(pred_disp[:, 0, :, :])

def main(num_frames, H,  W):
	# cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
	# cv2.resizeWindow("TEST", (2 * W, H))

	utils.delete_folders(FOLDERS_TO_CREATE)
	utils.create_folders(FOLDERS_TO_CREATE)

	sess_crestereo = ort.InferenceSession('models/crestereo.onnx')
	sess_crestereo_no_flow = ort.InferenceSession('models/crestereo_without_flow.onnx')

	image_files_left = [os.path.join(ZED_IMAGE_DIR, f) for f in os.listdir(ZED_IMAGE_DIR) if f.startswith('left_') and f.endswith('.png')]
	image_files_right = [os.path.join(ZED_IMAGE_DIR, f) for f in os.listdir(ZED_IMAGE_DIR) if f.startswith('right_') and f.endswith('.png')]
	
	image_files_left.sort()
	image_files_right.sort()

	assert(len(image_files_left) == len(image_files_right)), "Number of left and right images should be equal"
	assert(len(image_files_left) >= num_frames), "Number of frames should be less than total number of images"

	frame_rates = []
	for i in tqdm(range(num_frames)):
		left_img = cv2.imread(image_files_left[i])
		right_img = cv2.imread(image_files_right[i])

		left = cv2.resize(left_img, (W, H), interpolation=cv2.INTER_LINEAR)
		right = cv2.resize(right_img, (W, H), interpolation=cv2.INTER_LINEAR)
	
		start_time = time.time()

		model_inference = inference(left_img , right_img, sess_crestereo, sess_crestereo_no_flow, img_shape=(480, 640))   
		depth = utils.get_depth_data(model_inference, trt_inference.BASELINE, trt_inference.FOCAL_LENGTH)

		# extracting image name from the input left image
		img_name = os.path.basename(image_files_left[i])
		npy_name = img_name.replace('.png', '.npy')
		np.save(f"{ONNX_DISPARITY_DIR}/{npy_name}", model_inference)
		np.save(f"{ONNX_DEPTH_MAP_DIR}/{npy_name}", depth)

	
	# utils_matplotlib.plot_histograms(onnx_init_flow_plts, save_path=f"{ONNX_INIT_FLOW_DIR}/onnx_init_flow.png", visualize=False)

if __name__ == "__main__": 
	
	coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger
	logging.warning("[onnx_inference.py] Starting inference ...")
	num_frames = 20
	(H,W) = (480, 640)
	main(num_frames, H, W)	