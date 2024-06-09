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

import test_model

# camera parameters
BASELINE = 0.13
FOCAL_LENGTH = 1093.5

# zed input folders => input to the model pipeline
zed_input_dir = "zed_input"
# zed_input_disp_maps = f"{zed_input_dir}/disparity_maps"
zed_input_depth_maps = f"{zed_input_dir}/depth_maps"
zed_input_images= f"{zed_input_dir}/images"


def run_zed_pipeline(svo_file, num_frames=5): 	
	# logging.info(f"Running ZED pipeline for {num_frames} frames.")
	# deleting the old files
	for folder_path in [zed_input_depth_maps, zed_input_images]:
		logging.debug(f"Deleting the old files in {folder_path}")
		if os.path.exists(folder_path):
			try: 
				shutil.rmtree(folder_path)
			except OSError:
				logging.error(f"Error while deleting {folder_path}. Retrying...")
				# time.sleep(1)  # wait for 1 second before retrying
		else:
			print(f"The folder {folder_path} does not exist.")
	logging.debug("Deleted the old files.")
	
	# creating the new folders
	for path in [zed_input_depth_maps, zed_input_images]:
		os.makedirs(path, exist_ok=True)

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
			
	runtime_parameters = sl.RuntimeParameters()
	runtime_parameters.enable_fill_mode	= True
	
	total_svo_frames = zed.get_svo_number_of_frames()
	# logging.debug(f"Total number of frames in the svo file: {total_svo_frames}")	

	# setting cv2 window
	cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("ZED", 1000, 1000)

	cv2.namedWindow("MODEL", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("MODEL", 1000, 1000)

	for i in tqdm(range(0, num_frames, 30)):
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
			
			# model prediction
			pred = test_model.INFERENCE(imgL, imgR)
			
			t = float(in_w) / float(eval_w)
			
			# [MODEL] Depth Calculations
			# disparity data in meters
			model_disp_data = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t	
			logging.debug(f"model_disp_data.shape: {model_disp_data.shape} model_disp_data.dtype: {model_disp_data.dtype}")
			
			model_depth_data = utils.get_depth_data(model_disp_data, BASELINE, FOCAL_LENGTH)
			model_depth_map_mono = utils.uint8_normalization(model_depth_data)
			# converting model_depth_map_mono to 3-channel image
			model_depth_map_mono = cv2.cvtColor(model_depth_map_mono, cv2.COLOR_GRAY2BGR)
			model_depth_map_rgb = cv2.applyColorMap(model_depth_map_mono, cv2.COLORMAP_INFERNO)

			# cv2.imshow("MODEL", cv2.hconcat([model_depth_map_rgb, model_depth_map_mono]))
			# cv2.waitKey(0)

			# [ZED] Depth Calculations
			zed_depth_map_mono = svo_depth_map_unit8
			# converting zed_depth_map_mono to 3-channel image
			zed_depth_map_mono = cv2.cvtColor(zed_depth_map_mono, cv2.COLOR_GRAY2BGR)
			zed_depth_map_rgb = cv2.applyColorMap(zed_depth_map_mono, cv2.COLORMAP_INFERNO)
			
			cv2.imshow("ZED", cv2.hconcat([zed_depth_map_rgb, zed_depth_map_mono]))
			cv2.waitKey(0)
			
			# # # [ZED vs MODEL] Depth Calculations
			# # left_img_bgr = left_img
			# # left_img_mono = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
			# # left_img_mono = cv2.cvtColor(left_img_mono, cv2.COLOR_GRAY2BGR)

			# # concat_depth_mono = cv2.hconcat([left_img_mono, zed_depth_mono, model_depth_mono])
			# # concat_depth_bgr = cv2.hconcat([left_img_bgr, zed_depth_rgb, model_depth_rgb])
			# # concat_depth = cv2.vconcat([concat_depth_bgr, concat_depth_mono])
			# # cv2.imwrite(f"{zed_vs_model_dir}/frame_{frame_id}.png",concat_depth)	
			# # # cv2.imshow("TEST", concat_images)
			# # # cv2.waitKey(0)
			
			# # # # [ZED vs MODEL] Heatmap Calculations
			# # # # error_map = utils.create_depth_error_heatmap(model_depth_rgb, zed_depth_rgb, zed_vs_model_dir, frame_id)
			# # # depth_error_map_mono = utils.get_error_heatmap(model_depth_mono, zed_depth_mono)
			# # # depth_error_map_bgr = cv2.applyColorMap(depth_error_map_mono, cv2.COLORMAP_INFERNO)
			# # # depth_eror_map_concat_mono = cv2.hconcat([left_img_mono, zed_depth_mono, model_depth_mono, depth_error_map_mono])
			# # # depth_error_map_concat_bgr = cv2.hconcat([left_img_bgr, zed_depth_rgb, model_depth_rgb, depth_error_map_bgr])
			# # # # concat_error_depth = cv2.hconcat([left_img_bgr, zed_depth_rgb, model_depth_rgb, error_map])
			# # # # concat_error_depth = cv2.vconcat([depth_error_map_concat_bgr, depth_eror_map_concat_mono])
			# # # cv2.imshow("TEST", cv2.vconcat([depth_error_map_concat_bgr, depth_eror_map_concat_mono]))
			# # # cv2.imwrite(f"{zed_vs_model_heatmap_dir}/frame_{frame_id}.png",cv2.vconcat([depth_error_map_concat_bgr, depth_eror_map_concat_mono]))	
			# # # cv2.waitKey(0)
			
			# # # [ZED vs MODEL] Absolute Error Heatmap
			# # model_depth_data = 	(BASELINE * FOCAL_LENGTH) / (model_disp + 1e-6)
			# # logging.debug(f"model_depth_data: {model_depth_data[:5][:5]}")




	
	
	
	zed.close()



if __name__ == '__main__':

	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	svo_file = "svo-files/front_2024-05-15-18-59-18.svo"
	num_frames = 60
	run_zed_pipeline(svo_file, num_frames)
	