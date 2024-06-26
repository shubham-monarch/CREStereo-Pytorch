#! /usr/bin/env python3

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
import utils



ZED_IMAGE_DIR = zed_inference.ZED_IMG_DIR
ONNX_VS_TRT_DIR = "onnx_vs_trt"
FOLDERS_TO_CREATE = []

(W,H) = (640, 480)

def main(num_frames):
	
	utils.delete_folders(FOLDERS_TO_CREATE)
	utils.create_folders(FOLDERS_TO_CREATE)

	image_files_left = [os.path.join(ZED_IMAGE_DIR, f) for f in os.listdir(ZED_IMAGE_DIR) if f.startswith('left_') and f.endswith('.png')]
	image_files_right = [os.path.join(ZED_IMAGE_DIR, f) for f in os.listdir(ZED_IMAGE_DIR) if f.startswith('right_') and f.endswith('.png')]
	
	image_files_left.sort()
	image_files_right.sort()

	assert(len(image_files_left) == len(image_files_right)), "Number of left and right images should be equal"
	assert(len(image_files_left) >= num_frames), "Number of frames should be less than total number of images"

	sess_crestereo = ort.InferenceSession('models/crestereo.onnx')
	sess_crestereo_no_flow = ort.InferenceSession('models/crestereo_without_flow.onnx')

	for i in tqdm(range(num_frames)):
		left_img = cv2.imread(image_files_left[i])
		right_img = cv2.imread(image_files_right[i])

		left = cv2.resize(left_img, (W, H), interpolation=cv2.INTER_LINEAR)
		right = cv2.resize(right_img, (W, H), interpolation=cv2.INTER_LINEAR)
	
		model_inference_onnx = onnx_inference.inference(left_img , right_img, sess_crestereo, sess_crestereo_no_flow, img_shape=(480, 640))   
		# extracting image name from the input left image
		logging.warning(f"model_inference_onnx.shape: {model_inference_onnx.shape} model_inference_onnx.dtype: {model_inference_onnx.dtype}")
		img_name = os.path.basename(image_files_left[i])
		npy_name = img_name.replace('.png', '.npy')
		# np.save(f"{ONNX_DISPARITY_DIR}/{npy_name}", model_inference)
	

if __name__ == "__main__":
	coloredlogs.install(level="INFO", force=True)  # install a handler on the root logger
	num_frames = 30
	main(num_frames=30)