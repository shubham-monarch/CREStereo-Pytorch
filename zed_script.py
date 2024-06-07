#! /usr/bin/env python3

import sys
import pyzed.sl as sl
import numpy as np
import cv2
import os
import shutil
import coloredlogs, logging


# zed input folders => input to the model pipeline
zed_input_dir = "zed_input"
zed_input_disp_maps = f"{zed_input_dir}/disparity_maps"
zed_input_images= f"{zed_input_dir}/images"

def to_depth_map(pred):
	
	in_h, in_w = pred.shape[:2]
	eval_h, eval_w = (in_h,in_w)
	
	t = float(in_w) / float(eval_w)
	disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
	
	disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
	disp_vis = disp_vis.astype("uint8")
	# disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
	return disp_vis


def run_zed_pipeline(svo_file, num_frames=5): 
	
	# deleting the old files
	for folder_path in [zed_input_dir]:
		if os.path.exists(folder_path):
			shutil.rmtree(folder_path)
		else:
			print(f"The folder {folder_path} does not exist.")
	
	# creating the new folders
	for path in [zed_input_disp_maps, zed_input_images]:
		os.makedirs(path, exist_ok=True)

	input_type = sl.InputType()
	input_type.set_from_svo_file(svo_file)
	
	init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
	init_params.depth_mode = sl.DEPTH_MODE.ULTRA # Use ULTRA depth mode
	init_params.coordinate_units = sl.UNIT.METER # Use millimeter units (for depth measurements)

	zed = sl.Camera()
	status = zed.open(init_params)
	
	image_l = sl.Mat()
	image_r = sl.Mat()
	disp_map = sl.Mat()
			
	runtime_parameters = sl.RuntimeParameters()
	runtime_parameters.enable_fill_mode	= True
	
	i = 0
	while zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS and i < num_frames:

			logging.info(f"Processing {i}th frame!")			
			# retrieve and write stereo images
			zed.retrieve_image(image_l, sl.VIEW.LEFT) # Retrieve left image
			zed.retrieve_image(image_r, sl.VIEW.RIGHT) # Retrieve left image
			image_l.write( os.path.join(zed_input_images, f'left_{i}.png') )
			image_r.write( os.path.join(zed_input_images, f'right_{i}.png') )
			
			# retrieve and write disparity map
			zed.retrieve_image(disp_map, sl.VIEW.DEPTH)
			depth_map_colorized =to_depth_map(disp_map.get_data()[: , : , :3])
			cv2.imwrite( os.path.join(zed_input_disp_maps, f'frame_{i}.png'), to_depth_map(disp_map.get_data()[: , : , :3]))

			i += 1

if __name__ == '__main__':

	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	svo_file = "svo-files/front_2024-05-15-18-59-18.svo"
	run_zed_pipeline(svo_file)
