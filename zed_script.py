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
	
	logging.debug(f"Total number of frames in the svo file: {zed.get_svo_number_of_frames()}")
	logging.debug(f"Running the pipeline for {num_frames} frames.")

	image_l = sl.Mat()
	image_r = sl.Mat()
	depth_map = sl.Mat()
			
	runtime_parameters = sl.RuntimeParameters()
	runtime_parameters.enable_fill_mode	= True
	
	total_svo_frames = zed.get_svo_number_of_frames()
	logging.debug(f"Total number of frames in the svo file: {total_svo_frames}")	

	# setting cv2 window
	cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("ZED", 1000, 1000)

	for i in tqdm(range(0, num_frames, 30)):
		if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
			# logging.debug(f"Processing {i}th frame!s")
			zed.set_svo_position(i)	
			zed.retrieve_image(image_l, sl.VIEW.LEFT) # Retrieve left image
			zed.retrieve_image(image_r, sl.VIEW.RIGHT) # Retrieve left image
			image_l.write( os.path.join(zed_input_images, f'left_{i}.png') )
			image_r.write( os.path.join(zed_input_images, f'right_{i}.png') )
			
			# retrieve absolute depth map
			zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # Retrieve depth
			depth_map_data = depth_map.get_data()	
			depth_map_unit8 = utils.uint8_normalization(depth_map_data)

			cv2.imshow("ZED", depth_map_unit8)	
			cv2.waitKey(0)



if __name__ == '__main__':

	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	svo_file = "svo-files/front_2024-05-15-18-59-18.svo"
	num_frames = 60
	run_zed_pipeline(svo_file, num_frames)
	