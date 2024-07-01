#! /usr/bin/env python3

# TO-DO=>
# - tune onxx conversion params -> opset_version, do_constant_folding
# - analyze onxx conversion warnings


# standard imports
import coloredlogs, logging
import onnxruntime as ort
import os
from tqdm import tqdm
import cv2
import numpy as np


# custom imports
import zed_inference
import onnx_inference
import trt_inference as trt_inf
import utils, utils_matplotlib
import trt_engine


ZED_IMAGE_DIR = zed_inference.ZED_IMG_DIR
ONNX_VS_TRT_DIR = "onnx_vs_trt"
FOLDERS_TO_CREATE = []

(W,H) = (640, 480)

def onnx_inference_no_flow(left_img, right_img, sess_crestereo, sess_crestereo_no_flow,  H, W):
	onnx_inference_no_flow = onnx_inference.inference_no_flow(left_img , right_img, 
														sess_crestereo, sess_crestereo_no_flow, 
														(H, W))   
	
	onnx_inference_no_flow_reshaped = np.squeeze(onnx_inference_no_flow[:, 0, :, :]) 
	logging.warning(f"onnx_inference_no_flow.shape: {onnx_inference_no_flow.shape}") # (1, 2, 240, 320)
	return onnx_inference_no_flow_reshaped



def main(num_frames):
	
	utils.delete_folders(FOLDERS_TO_CREATE)
	utils.create_folders(FOLDERS_TO_CREATE)

	image_files_left = [os.path.join(ZED_IMAGE_DIR, f) for f in os.listdir(ZED_IMAGE_DIR) if f.startswith('left_') and f.endswith('.png')]
	image_files_right = [os.path.join(ZED_IMAGE_DIR, f) for f in os.listdir(ZED_IMAGE_DIR) if f.startswith('right_') and f.endswith('.png')]
	
	image_files_left.sort()
	image_files_right.sort()

	assert(len(image_files_left) == len(image_files_right)), "Number of left and right images should be equal"
	assert(len(image_files_left) >= num_frames), "Number of frames should be less than total number of images"

	# loading the onnx models
	sess_crestereo = ort.InferenceSession('models/crestereo.onnx')
	sess_crestereo_no_flow = ort.InferenceSession('models/crestereo_without_flow.onnx')

	# loading the trt engines
	# trt_engine_no_flow = trt_inf.TRTEngine("models/crestereo_without_flow.trt")
	# trt_engine_with_flow = trt_inf.TRTEngine("models/crestereo.trt")
	
	# trt_engine_no_flow = trt_engine.trt_inference("models/simp_crestereo_without_flow.trt")
	


	for i in tqdm(range(num_frames)):
		left_img = cv2.imread(image_files_left[i])
		right_img = cv2.imread(image_files_right[i])

		onnx_inf_no_flow = onnx_inference_no_flow(left_img, right_img, sess_crestereo, sess_crestereo_no_flow, H, W)

		trt_inf = trt_engine.trt_inference("models/simp_crestereo_without_flow.trt", left_img, right_img, H, W)
		trt_inf_ = np.squeeze(trt_inf[:, 0, :, :])

		utils_matplotlib.plot_histograms(onnx_inf_no_flow, trt_inf_)
		# utils_matplotlib.plot_histograms(onnx_inf_no_flow)
		# utils_matplotlib.plot_histograms(trt_inf_)
	# 	left = cv2.resize(left_img, (W, H), interpolation=cv2.INTER_LINEAR)
	# 	right = cv2.resize(right_img, (W, H), interpolation=cv2.INTER_LINEAR)

	# 	# ONNX INFERENCE -->
	# 	onnx_inference_no_flow = onnx_inference.inference_no_flow(left_img , right_img, 
	# 														sess_crestereo, sess_crestereo_no_flow, 
	# 														img_shape=(480, 640))   
		
	# 	onnx_inference_no_flow_reshaped = np.squeeze(onnx_inference_no_flow[:, 0, :, :]) 
	# 	logging.warning(f"onnx_inference_no_flow.shape: {onnx_inference_no_flow.shape}") # (1, 2, 240, 320)
	
	# 	onnx_inference_with_flow = onnx_inference.inference_with_flow(left_img , right_img, 
	# 														  sess_crestereo, sess_crestereo_no_flow,
	# 														  onnx_inference_no_flow, 
	# 														  img_shape=(480, 640))   
		
	# 	# logging.warning(f"onnx_inference_with_flow.shape: {onnx_inference_with_flow.shape}")
		
	# 	# TRT INFERENCE --> 
		
	# 	# TRT ENGINE => NO FLOW
	# 	ppi_trt_no_flow = trt_engine_no_flow.pre_process_input([left_img, right_img], 
	# 													 [(H // 2, W // 2), (H // 2, W // 2)])
	# 	trt_engine_no_flow.load_input(ppi_trt_no_flow)
	# 	trt_inference_outputs_no_flow =  trt_engine_no_flow.run_trt_inference()
	# 	trt_inference_output_no_flow = trt_inference_outputs_no_flow[0].reshape(1, 2, H // 2, W // 2)
	# 	trt_inference_output_no_flow_reshaped = np.squeeze(trt_inference_output_no_flow[:, 0, :, :])
	# 	zeros_arr = np.zeros(trt_inference_output_no_flow.shape)
		
		
	# 	utils_matplotlib.plot_histograms(onnx_inference_no_flow_reshaped, 
	# 										   	trt_inference_output_no_flow_reshaped)  
		

if __name__ == "__main__":
	coloredlogs.install(level="INFO", force=True)  # install a handler on the root logger
	num_frames = 1 
	main(num_frames)