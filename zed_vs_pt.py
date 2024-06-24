#! /usr/bin/env python3 

import coloredlogs, logging
import zed_inference
import pt_inference
import utils 
import os
import numpy as np

ZED_PCL_DIR = zed_inference.ZED_PCL_DIR
PT_PCL_DIR = pt_inference.PT_PCL_DIR

ZED_VS_PT_DIR = "zed_vs_pt"
ZED_VS_PT_PCL_DIR = f"{ZED_VS_PT_DIR}/pcl_error"

FOLDERS_TO_CREATE = [ZED_VS_PT_PCL_DIR]

def main(): 	
	
	utils.delete_folders(FOLDERS_TO_CREATE)
	utils.create_folders(FOLDERS_TO_CREATE)
	
	zed_files = [] 
	pt_files = [] 

	zed_files = [os.path.join(ZED_PCL_DIR, f) for f in os.listdir(ZED_PCL_DIR) if f.endswith('.npy')]
	pt_files = [os.path.join(PT_PCL_DIR, f) for f in os.listdir(PT_PCL_DIR) if f.endswith('.npy')]

	logging.warning(f"len(zed_files): {len(zed_files)} \nzed_files: {zed_files}")
	logging.warning(f"len(pt_files): {len(pt_files)} \npt_files: {pt_files}")
	
	

	zed_files = sorted(zed_files)
	pt_files = sorted(pt_files)

	assert len(zed_files) == len(pt_files), f"Number of files in ZED({len(zed_files)}) and PT directories({len(pt_files)}) should be the same."

	for zed_file, pt_file in zip(zed_files, pt_files):
		zed_pcl = np.load(zed_file)
		pt_pcl = np.load(pt_file)
		error = zed_pcl - pt_pcl
		filename = os.path.basename(zed_file)
		utils.write_legend_plot(error, f"{ZED_VS_PT_PCL_DIR}/{filename}.png")

if __name__ == "__main__":
	coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger
	main()