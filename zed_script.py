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
zed_input_dir = "zed_input"
zed_input_images= f"{zed_input_dir}/images"
# # zed_input_disp_maps = f"{zed_input_dir}/disparity_maps"
# zed_input_depth_maps = f"{zed_input_dir}/depth_maps"

# comparison folders
zed_vs_model_dir = "zed_vs_model"
img_zed_model_error_dir = f"{zed_vs_model_dir}/img-zed-model-error"
mean_variance_hist_dir = f"{zed_vs_model_dir}/mean_variance_hist"
# zed_vs_model_depth_map = f"{zed_vs_model_dir}/depth_maps"
# zed_vs_model_heatmap_dir = f"{zed_vs_model_dir}/depth_error_heatmaps"
# zed_vs_model_disp_dir = f"{zed_vs_model_dir}/disparity"
# zed_vs_model_depth_dir = f"{zed_vs_model_dir}/depth"

def run_zed_pipeline(svo_file, num_frames=5): 	
	# logging.info(f"Running ZED pipeline for {num_frames} frames.")
	
	# deleting the old files
	for folder_path in [zed_vs_model_dir, img_zed_model_error_dir, zed_input_images,mean_variance_hist_dir]:
		logging.debug(f"Deleting the old files in {folder_path}")
		if os.path.exists(folder_path):
			try: 
				shutil.rmtree(folder_path)
			except OSError:
				logging.error(f"Error while deleting {folder_path}. Retrying...")
				# time.sleep(1)  # wait for 1 second before retrying
		else:
			print(f"The folder {folder_path} does not exist.")
	logging.info("Deleted the old files.")
	

	# creating the new folders
	for path in [zed_vs_model_dir, img_zed_model_error_dir, zed_input_images,mean_variance_hist_dir]:
		os.makedirs(path, exist_ok=True)
		logging.info(f"Created the {path} folder!")

	input_type = sl.InputType()
	input_type.set_from_svo_file(svo_file)
	
	init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
	init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Use ULTRA depth mode
	init_params.coordinate_units = sl.UNIT.METER # Use millimeter units (for depth measurements)
	
	zed = sl.Camera()
	status = zed.open(init_params)
	
	# logging.debug(f"Total number of frames in the svo file: {zed.get_svo_number_of_frames()}")
	# logging.debug(f"Running the pipeline for {num_frames} frames.")

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
			image_l.write( os.path.join(zed_input_images, f'left_{i}.png') )
			image_r.write( os.path.join(zed_input_images, f'right_{i}.png') )

			left_img_path = os.path.join(zed_input_images, f"left_{i}.png")
			right_img_path = os.path.join(zed_input_images, f"right_{i}.png")
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

			depth_error_data = cv2.absdiff(model_depth_data_filtered, zed_depth_data_filtered)
			mean_depth_errors.append(np.mean(depth_error_data))
			variance_depth_errors.append(np.var(depth_error_data))

			depth_error_map_mono = utils.uint8_normalization(depth_error_data)
			# converting depth_error_map_mono to 3-channel image
			depth_error_map_mono = cv2.cvtColor(depth_error_map_mono, cv2.COLOR_GRAY2BGR)
			depth_error_map_rgb = cv2.applyColorMap(depth_error_map_mono, cv2.COLORMAP_INFERNO)

			# [ZED vs MODEL] writing [img  +  zed_depth +  model_depth + depth_error] to disk
			concat_img_zed_model_error_mono = cv2.hconcat([imgL, zed_depth_map_mono, model_depth_map_mono, depth_error_map_mono])
			concat_img_zed_model_error_rgb = cv2.hconcat([imgL_mono, zed_depth_map_rgb, model_depth_map_rgb, depth_error_map_rgb])
			concat_img_zed_model_error = cv2.vconcat([concat_img_zed_model_error_mono, concat_img_zed_model_error_rgb])
			
			# cv2.imshow("TEST", concat_img_zed_model_error)
			# cv2.waitKey()
			cv2.imwrite(f"{img_zed_model_error_dir}/frame_{i}.png", concat_img_zed_model_error)
			
			
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
	plt.savefig(f"{mean_variance_hist_dir}/mean_variance_hist.png")
	plt.show()


if __name__ == '__main__':

	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	svo_file = "svo-files/front_2024-05-15-18-59-18.svo"
	num_frames = 100
	logging.info(f"Running ZED pipeline for {num_frames} frames.")
	run_zed_pipeline(svo_file, num_frames)
	