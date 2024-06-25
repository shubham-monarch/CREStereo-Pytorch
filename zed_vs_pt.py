#! /usr/bin/env python3 

import coloredlogs, logging
import zed_inference
import pt_inference
import utils 
import os
import numpy as np
from tqdm import tqdm

# ZED_DEP = zed_inference.ZED_PCL_DIR
# PT_DEPTH_MAP_DIR = pt_inference.PT_DEPTH_MAP_DIR

ZED_VS_PT_DIR = "zed_vs_pt"
ZED_VS_PT_PCL_DIR = f"{ZED_VS_PT_DIR}/depth_error"

FOLDERS_TO_CREATE = [ZED_VS_PT_PCL_DIR]

def main(): 	
	
	utils.delete_folders(FOLDERS_TO_CREATE)
	utils.create_folders(FOLDERS_TO_CREATE)

	zed_files = [] 
	pt_files = [] 

	zed_files = [os.path.join(zed_inference.ZED_DEPTH_MAP_DIR, f) for f in os.listdir(zed_inference.ZED_DEPTH_MAP_DIR) if f.endswith('.npy')]
	pt_files = [os.path.join(pt_inference.PT_DEPTH_MAP_DIR, f) for f in os.listdir(pt_inference.PT_DEPTH_MAP_DIR) if f.endswith('.npy')]

	zed_files = sorted(zed_files)
	pt_files = sorted(pt_files)

	assert len(zed_files) == len(pt_files), f"Number of files in ZED({len(zed_files)}) and PT directories({len(pt_files)}) should be the same."

	for zed_file, pt_file in tqdm(zip(zed_files, pt_files), total = len(zed_files)):
		zed_depth = np.load(zed_file)
		pt_depth = np.load(pt_file)

		zed_depth[zed_depth > 10] = np.nan
		pt_depth[pt_depth > 10] = np.nan
		
		# plts = []
		# plts.append(utils.PLT(data=zed_depth, 
		# 				title='ZED DEPTH MAP',
		# 				bins=100, 
		# 				range=(zed_depth.min(), zed_depth.max())))
		# plts.append(utils.PLT(data=pt_depth, 
		# 				title='PT DEPTH MAP',
		# 				bins=100, 
		# 				range=(pt_depth.min(), pt_depth.max())))		
		
		error = pt_depth - zed_depth
		filename_npy = os.path.basename(zed_file)
		filename_png = filename_npy.replace('.npy', '.png')	
		utils.write_legend_plot(error, f"{ZED_VS_PT_PCL_DIR}/{filename_png}")

if __name__ == "__main__":
	coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger
	main()