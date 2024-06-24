#! /usr/bin/env python3
import coloredlogs, logging
import utils
import pyzed.sl as sl
import random
from tqdm import tqdm
import os
import numpy as np
import cv2

ZED_VS_PT_DIR = "zed_vs_pt"

ZED_IMG_DIR = f"{ZED_VS_PT_DIR}/zed_images"	
ZED_PCL_DIR = f"{ZED_VS_PT_DIR}/zed_pcl"

FOLDERS_TO_CREATE = [ZED_PCL_DIR, ZED_IMG_DIR]


def main(svo_file : str, num_frames : int) -> None: 	
		
	utils.delete_folders(FOLDERS_TO_CREATE)
	utils.create_folders(FOLDERS_TO_CREATE)
	
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
	zed_pcl = sl.Mat()

	mean_depth_errors = []
	variance_depth_errors= []
			
	runtime_parameters = sl.RuntimeParameters()
	runtime_parameters.enable_fill_mode	= True
	
	total_svo_frames = zed.get_svo_number_of_frames()
	
	assert num_frames <= total_svo_frames, "num_frames should be less than total_svo_frames"
	
	random_frames = random.sample(range(0, total_svo_frames), num_frames)
	for i in tqdm(random_frames):
		if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
			zed.set_svo_position(i)	
			zed.retrieve_image(image_l, sl.VIEW.LEFT) # Retrieve left image
			zed.retrieve_image(image_r, sl.VIEW.RIGHT) # Retrieve left image
			# writing images
			image_l.write( os.path.join(ZED_IMG_DIR , f'left_{i}.png') )
			image_r.write( os.path.join(ZED_IMG_DIR , f'right_{i}.png') )
			# writing pcl
			zed.retrieve_measure(zed_pcl, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
			# saving as PLY
			zed_pcl.write(os.path.join(ZED_PCL_DIR, f"pcl_{i}.ply"))
			zed_pcl_arr = zed_pcl.get_data()
			# saving as numpy array	
			np.save(os.path.join(ZED_PCL_DIR, f"pcl_{i}.npy"), zed_pcl_arr)
	zed.close()


if __name__ == "__main__":
	coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger
	svo_file = "svo-files/front_2024-05-15-18-59-18.svo"
	num_frames = 10
	logging.warning(f"Running ZED inference for {num_frames} frames.")
	main(svo_file, num_frames)
	