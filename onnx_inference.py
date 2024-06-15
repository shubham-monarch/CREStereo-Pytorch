#! /usr/bin/env python3

import numpy as np
import onnxruntime as ort
import coloredlogs, logging
import matplotlib.pyplot as plt
import cv2
import utils

# TO-DO -> 
# - use np.squeeze()
# - calculate frame rate




def inference(left_img, right_img, model, model_no_flow, img_dims=(480, 640)):
	
	(h,w) = img_dims
	
	# Get onnx model layer names (see convert_to_onnx.py for what these are)
	input1_name = model.get_inputs()[0].name
	input2_name = model.get_inputs()[1].name
	input3_name = model.get_inputs()[2].name
	output_name = model.get_outputs()[0].name

	
	# Transpose the dimensions and add a batch dimension
	imgL = cv2.resize(left_img, (w, h), interpolation=cv2.INTER_LINEAR)
	imgR = cv2.resize(right_img, (w, h), interpolation=cv2.INTER_LINEAR)
	imgL = np.ascontiguousarray(imgL.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	imgR = np.ascontiguousarray(imgR.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)

	logging.warn(f"imgL.shape: {imgL.shape} imgR.shape: {imgR.shape}")

	imgL_dw2 = cv2.resize(left_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
	imgR_dw2 = cv2.resize(right_img, (w//2, h//2),  interpolation=cv2.INTER_LINEAR)
	imgL_dw2 = np.ascontiguousarray(imgL_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32) 
	imgR_dw2 = np.ascontiguousarray(imgR_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)

	# # Perform inference with the first model
	# pred_flow_dw2 = sess_cstrereo_without_flow.run(None, {'left': imgL_dw2, 'right': imgR_dw2})
	# input_names_without_flow = [input.name for input in sess_cstrereo_without_flow.get_inputs()]
	# input_feed_without_flow = {name: value for name, value in zip(input_names_without_flow, [imgL_dw2, imgR_dw2])}
	
	# without flow model inference
	# pred_flow_dw2 = sess_cstrereo_without_flow.run(None, input_feed_without_flow)
	# First pass it just to get the flow
	pred_flow_dw2 = model_no_flow.run(
		[output_name], {input1_name: imgL_dw2, input2_name: imgR_dw2})[0]
	# Second pass gets us the disparity
	pred_disp = model.run([output_name], {
						  input1_name: imgL, input2_name: imgR, input3_name: pred_flow_dw2})[0]

	return np.squeeze(pred_disp[:, 0, :, :])


	# input_names_with_flow = [input.name for input in sess_crestereo.get_inputs()]
	# input_feed_with_flow = {name: value for name, value in zip(input_names_with_flow, [imgL, imgR, pred_flow_dw2[0]])}    
	
	# logging.debug(f"input_names_with_flow: {input_names_with_flow}")
	# # pred_flow = sess_crestereo.run(None, input_feed)

	# pred_flow = sess_crestereo.run(None, input_feed_with_flow)    


# (H, W)
DIMS = (480, 640)
H,W = DIMS

if __name__ == "__main__": 
	
	coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger
	
	# setting up cv2 window
	cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("TEST", 2 * W, H)

	# Load the ONNX models
	sess_crestereo = ort.InferenceSession('models/crestereo.onnx')
	sess_crestereo_no_flow = ort.InferenceSession('models/crestereo_without_flow.onnx')
	
	# Load and preprocess your images
	left_img = cv2.imread('zed_input/images/left_20.png')
	right_img = cv2.imread('zed_input/images/right_20.png')

	left = cv2.resize(left_img, (W, H), interpolation=cv2.INTER_LINEAR)
	right = cv2.resize(right_img, (W, H), interpolation=cv2.INTER_LINEAR)

	model_inference = inference(left_img , right_img, sess_crestereo, sess_crestereo_no_flow, img_dims=(480, 640))   
	logging.warning(f"model_inference.shape: {model_inference.shape} model_inference.dtype: {model_inference.dtype}") 
	model_inference_depth_map_mono = utils.uint8_normalization(model_inference)
	logging.warning(f"model_inference_depth_map_mono.shape: {model_inference_depth_map_mono.shape} model_inference.dtype: {model_inference_depth_map_mono.dtype}") 
	model_infereence_depth_map_unit8 = cv2.cvtColor(model_inference_depth_map_mono, cv2.COLOR_GRAY2BGR)


	# visualizing the results
	cv2.imshow("TEST", left)
	cv2.waitKey(0)
	cv2.imshow("TEST", model_infereence_depth_map_unit8)
	cv2.waitKey(0)
	cv2.imshow("TEST", cv2.hconcat([left, model_infereence_depth_map_unit8]))
	cv2.waitKey(0)


	cv2.destroyAllWindows()

	