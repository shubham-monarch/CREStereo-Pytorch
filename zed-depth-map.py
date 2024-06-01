#! /usr/bin/env python3

import sys
import pyzed.sl as sl
import numpy as np
import cv2
import os
import shutil

def colorize_depth_map(pred):
	
	in_h, in_w = pred.shape[:2]
	eval_h, eval_w = (in_h,in_w)
	
	t = float(in_w) / float(eval_w)
	disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
	
	disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
	disp_vis = disp_vis.astype("uint8")
	disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)
	return disp_vis


if __name__ == '__main__':


	svo_file = "svo-files/front_2023-11-03-10-46-17.svo"
	i = 0 


	input_type = sl.InputType()
	input_type.set_from_svo_file(svo_file)
	
	init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
	init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Use ULTRA depth mode
	init_params.coordinate_units = sl.UNIT.METER # Use millimeter units (for depth measurements)

	zed = sl.Camera()
	status = zed.open(init_params)
	
	image_l = sl.Mat()
	image_r = sl.Mat()
	
	depth_map = sl.Mat()
	depth_for_display = sl.Mat()
			
	runtime_parameters = sl.RuntimeParameters()
	runtime_parameters.enable_fill_mode	= True
	
	zed_output_dir = "zed-output"
	zed_depth_map_dir = "zed-depth-maps"
	
	# delete old directories
	for path in [zed_output_dir, zed_depth_map_dir]:
		try:
			shutil.rmtree(path)
			print(f"Directory '{path}' has been removed successfully.")
		except OSError as e:
			print(f"Error: {e.strerror}")

	
	os.makedirs( zed_output_dir, exist_ok=True)
	os.makedirs( zed_depth_map_dir, exist_ok=True)

	while True:

		# SVO PROCESSING 
		if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
			
			print(f"Processing {i}th frame!")
			# retrieve stereo images
			zed.retrieve_image(image_l, sl.VIEW.LEFT) # Retrieve left image
			zed.retrieve_image(image_r, sl.VIEW.RIGHT) # Retrieve left image
			
			# retrieve depth map image
			zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)

			image_l.write( os.path.join(zed_output_dir, f'left_{i}.png') )
			image_r.write( os.path.join(zed_output_dir, f'right_{i}.png') )
			
			depth_map_colorized = colorize_depth_map(depth_for_display.get_data()[: , : , :3])
			cv2.imwrite( os.path.join(zed_depth_map_dir, f'frame_{i}.png'), depth_map_colorized)	

			i = i + 1
			#combined_img = np.hstack((pred, disp_vis))
			#cv2.namedWindow("output", cv2.WINDOW_NORMAL)
			#cv2.imshow("output", depth_map_colorized)	
			#cv2.imwrite("output/output.jpg", disp_vis)
			#cv2.waitKey()

			if i > 5: 
				break

		else:
			break

