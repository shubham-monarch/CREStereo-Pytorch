#! /usr/bin/env python3

import sys
import pyzed.sl as sl
import numpy as np
import cv2
import os
import shutil
import coloredlogs, logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils
import random
import test_model

# camera parameters
BASELINE = 0.13
FOCAL_LENGTH = 1093.5

# zed input folders => input to the model pipeline
ZED_INPUT_DIR = "zed_input"
ZED_INPUT_IMAGES_DIR = f"{ZED_INPUT_DIR}/images"


# model output folders => output of the model pipeline
MODEL_OUTPUT_DIR = "outputs/model_output"
MODEL_DEPTH_MAPS_DIR = f"{MODEL_OUTPUT_DIR}/depth_maps"

# zed output folders => output of the zed pipeline
ZED_OUTPUT_DIR = "outputs/zed_output"
ZED_DEPTH_MAPS_DIR = f"{ZED_OUTPUT_DIR}/depth_maps"


# comparison folders
ZED_VS_MODEL_DIR = "outputs/zed_vs_model"
IMG_ZED_MODEL_ERROR_DIR = f"{ZED_VS_MODEL_DIR}/img-zed-model-error"
MEAN_VARIANCE_HIST_DIR = f"{ZED_VS_MODEL_DIR}/mean_variance_hist"
ZED_VS_MODEL_HEATMAP_DIR = f"{ZED_VS_MODEL_DIR}/depth_error_heatmaps"

PIPELINE_FOLDERS = [ZED_INPUT_IMAGES_DIR , # zed-pipeline inputs
				IMG_ZED_MODEL_ERROR_DIR, MEAN_VARIANCE_HIST_DIR, ZED_VS_MODEL_HEATMAP_DIR, # zed-vs-model 
				MODEL_DEPTH_MAPS_DIR, # model outputs
				ZED_DEPTH_MAPS_DIR] # zed outputs


def run_zed_pipeline(svo_file, num_frames=5): 	
		
	# deleting the old folders
	utils.delete_folders(PIPELINE_FOLDERS)
	# creating the new folders
	utils.create_folders(PIPELINE_FOLDERS)
	
	# ZED PROCESSING
	input_type = sl.InputType()
	input_type.set_from_svo_file(svo_file)
	
	init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
	init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Use ULTRA depth mode
	init_params.coordinate_units = sl.UNIT.METER # Use millimeter units (for depth measurements)
	
	zed = sl.Camera()
	status = zed.open(init_params)
	
	image_l = sl.Mat()
	image_r = sl.Mat()
	depth_map = sl.Mat()

	mean_depth_errors = []
	variance_depth_errors= []
			
	runtime_parameters = sl.RuntimeParameters()
	runtime_parameters.enable_fill_mode	= True
	
	total_svo_frames = zed.get_svo_number_of_frames()
	# logging.debug(f"Total number of frames in the svo file: {total_svo_frames}")	

	# setting cv2 window
	cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("TEST", 1000, 1000)
	
	assert num_frames <= total_svo_frames, "num_frames should be less than total_svo_frames"
	
	random_frames = random.sample(range(0, total_svo_frames), num_frames)
	for i in tqdm(random_frames):
		if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
			
			# logging.debug(f"Processing {i}th frame!s")
			zed.set_svo_position(i)	
			zed.retrieve_image(image_l, sl.VIEW.LEFT) # Retrieve left image
			zed.retrieve_image(image_r, sl.VIEW.RIGHT) # Retrieve left image
			image_l.write( os.path.join(ZED_INPUT_IMAGES_DIR , f'left_{i}.png') )
			image_r.write( os.path.join(ZED_INPUT_IMAGES_DIR , f'right_{i}.png') )

			left_img_path = os.path.join(ZED_INPUT_IMAGES_DIR , f"left_{i}.png")
			right_img_path = os.path.join(ZED_INPUT_IMAGES_DIR , f"right_{i}.png")
			left_img = cv2.imread(left_img_path)	
			right_img = cv2.imread(right_img_path)
			
			# retrieve absolute depth map
			zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # Retrieve depth
			# contains depth data in meters
			svo_depth_map_data = depth_map.get_data()
			
			# depth data in uint8 format normalized between 0-255	
			svo_depth_map_unit8 = utils.uint8_normalization(svo_depth_map_data)

			# cv2.imshow("ZED", depth_map_unit8)	
			# cv2.waitKey(0)

			in_h, in_w = left_img.shape[:2]

			# Resize image in case the GPU memory overflows
			eval_h, eval_w = (in_h,in_w)
			
			assert eval_h%8 == 0, "input height should be divisible by 8"
			assert eval_w%8 == 0, "input width should be divisible by 8"
			
			imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
			imgR = cv2.resize(right_img	, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)

			# converting imgL_mono to 3-channel grayscale image	
			imgL_mono = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
			imgL_mono = cv2.cvtColor(imgL_mono, cv2.COLOR_GRAY2BGR)

			# model prediction
			pred = test_model.INFERENCE(imgL, imgR)
			
			t = float(in_w) / float(eval_w)
			
			# [MODEL] Depth Calculations
			# disparity data in meters
			model_disp_data = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t	
			# logging.debug(f"model_disp_data.shape: {model_disp_data.shape} model_disp_data.dtype: {model_disp_data.dtype}")
			
			model_depth_data = utils.get_depth_data(model_disp_data, BASELINE, FOCAL_LENGTH)
			model_depth_map_mono = utils.uint8_normalization(model_depth_data)
			# converting model_depth_map_mono to 3-channel image
			model_depth_map_mono = cv2.cvtColor(model_depth_map_mono, cv2.COLOR_GRAY2BGR)
			model_depth_map_rgb = cv2.applyColorMap(model_depth_map_mono, cv2.COLORMAP_INFERNO)

			# cv2.imshow("MODEL", cv2.hconcat([model_depth_map_rgb, model_depth_map_mono]))
			# cv2.waitKey(0)

			# [ZED] Depth Calculations
			zed_depth_data = svo_depth_map_data
			zed_depth_map_mono = utils.uint8_normalization(zed_depth_data)
			# converting zed_depth_map_mono to 3-channel image
			zed_depth_map_mono = cv2.cvtColor(zed_depth_map_mono, cv2.COLOR_GRAY2BGR)
			zed_depth_map_rgb = cv2.applyColorMap(zed_depth_map_mono, cv2.COLORMAP_INFERNO)
			
			# cv2.imshow("ZED", cv2.hconcat([zed_depth_map_rgb, zed_depth_map_mono]))
			# cv2.waitKey(0)
			
			# [ZED vs MODEL] Depth Calculations
			# filtering inf values 
			model_depth_data_filtered = utils.inf_filtering(model_depth_data)
			zed_depth_data_filtered = utils.inf_filtering(zed_depth_data)
			utils.write_legend_plot(model_depth_data_filtered, f"{MODEL_DEPTH_MAPS_DIR}/frame_{i}.png")
			utils.write_legend_plot(zed_depth_data_filtered, f"{ZED_DEPTH_MAPS_DIR}/frame_{i}.png")


			depth_error_data = cv2.absdiff(model_depth_data_filtered, zed_depth_data_filtered)
			
			utils.write_legend_plot(depth_error_data, f"{ZED_VS_MODEL_HEATMAP_DIR}/frame_{i}.png")		


			mean_depth_errors.append(np.mean(depth_error_data))
			variance_depth_errors.append(np.var(depth_error_data))

			depth_error_map_mono = utils.uint8_normalization(depth_error_data)
			# converting depth_error_map_mono to 3-channel image
			depth_error_map_mono = cv2.cvtColor(depth_error_map_mono, cv2.COLOR_GRAY2BGR)
			depth_error_map_rgb = cv2.applyColorMap(depth_error_map_mono, cv2.COLORMAP_INFERNO)

			# [ZED vs MODEL] writing [img  +  zed_depth +  model_depth + depth_error] to disk
			concat_img_zed_model_error_mono = cv2.hconcat([imgL_mono, zed_depth_map_mono, model_depth_map_mono, depth_error_map_mono])
			concat_img_zed_model_error_rgb = cv2.hconcat([imgL, zed_depth_map_rgb, model_depth_map_rgb, depth_error_map_rgb])
			concat_img_zed_model_error = cv2.vconcat([concat_img_zed_model_error_mono, concat_img_zed_model_error_rgb])
			
			# cv2.imshow("TEST", concat_img_zed_model_error)
			# cv2.waitKey()
			cv2.imwrite(f"{IMG_ZED_MODEL_ERROR_DIR}/frame_{i}.png", concat_img_zed_model_error)
			
			
			# a = cv2.hconcat([model_depth_map_mono, zed_depth_map_mono, depth_error_map_mono])
			# b = cv2.hconcat([model_depth_map_rgb, zed_depth_map_rgb, depth_error_map_rgb])
			# cv2.imshow("TEST", cv2.vconcat([a, b]))
			# cv2.waitKey(0)
			
	zed.close()

	logging.info("Finished processing all the frames.")

	plt.figure(figsize=(12, 6))

	plt.subplot(1, 2, 1)
	plt.hist(mean_depth_errors, bins=200, color='blue', edgecolor='black')
	plt.title('Histogram of Mean Errors')

	plt.subplot(1, 2, 2)
	plt.hist(variance_depth_errors, bins=200, color='red', edgecolor='black')
	plt.title('Histogram of Variance Errors')
	plt.savefig(f"{MEAN_VARIANCE_HIST_DIR}/mean_variance_hist.png")
	plt.close()


if __name__ == '__main__':

	# coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger
	# logging.getLogger('matplotlib').setLevel(logging.WARNING)
	svo_file = "svo-files/front_2024-05-15-18-59-18.svo"
	num_frames = 100
	logging.info(f"Running ZED pipeline for {num_frames} frames.")
	run_zed_pipeline(svo_file, num_frames)
	