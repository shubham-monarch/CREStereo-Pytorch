#! /usr/bin/env python3 

import coloredlogs, logging
import zed_inference
import pt_inference
import utils 
import os
import numpy as np
from tqdm import tqdm

ZED_PCL_DIR = zed_inference.ZED_PCL_DIR
PT_DEPTH_MAP_DIR = pt_inference.PT_DEPTH_MAP_DIR

ZED_VS_PT_DIR = "zed_vs_pt"
ZED_VS_PT_PCL_DIR = f"{ZED_VS_PT_DIR}/depth_error"

FOLDERS_TO_CREATE = [ZED_VS_PT_PCL_DIR]

def main(): 	
	
	utils.delete_folders(FOLDERS_TO_CREATE)
	utils.create_folders(FOLDERS_TO_CREATE)
	
	zed_files = [] 
	pt_files = [] 

	zed_files = [os.path.join(ZED_PCL_DIR, f) for f in os.listdir(ZED_PCL_DIR) if f.endswith('.npy')]
	pt_files = [os.path.join(PT_DEPTH_MAP_DIR, f) for f in os.listdir(PT_DEPTH_MAP_DIR) if f.endswith('.npy')]

	zed_files = sorted(zed_files)
	pt_files = sorted(pt_files)

	assert len(zed_files) == len(pt_files), f"Number of files in ZED({len(zed_files)}) and PT directories({len(pt_files)}) should be the same."

	for zed_file, pt_file in tqdm(zip(zed_files, pt_files), total = len(zed_files)):
		zed_depth = np.load(zed_file)
		pt_depth = np.load(pt_file)
		# error = zed_depth - pt_depth
		error = pt_depth - zed_depth
		npy_filename = os.path.basename(zed_file)
		png_filename = npy_filename.replace('.npy', '.png')	
		utils.write_legend_plot(error, f"{ZED_VS_PT_PCL_DIR}/{png_filename}")

if __name__ == "__main__":
	coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger
	main()