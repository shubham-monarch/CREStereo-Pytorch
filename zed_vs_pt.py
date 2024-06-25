#! /usr/bin/env python3 

import coloredlogs, logging
import zed_inference
import pt_inference
import utils 
import os
import numpy as np
from tqdm import tqdm
import cv2

# ZED_DEP = zed_inference.ZED_PCL_DIR
# PT_DEPTH_MAP_DIR = pt_inference.PT_DEPTH_MAP_DIR

ZED_VS_PT_DIR = "zed_vs_pt"
ZED_VS_PT_DEPTH_ERROR_DIR = f"{ZED_VS_PT_DIR}/depth_error"
ZED_VS_PT_DEPTH_ERROR_HIST = f"{ZED_VS_PT_DEPTH_ERROR_DIR}/depth_error_histograms"
ZED_VS_PT_DEPTH_ERROR_HEATMAP = f"{ZED_VS_PT_DEPTH_ERROR_DIR}/depth_error_heatmaps"

FOLDERS_TO_CREATE = [ZED_VS_PT_DEPTH_ERROR_HIST, ZED_VS_PT_DEPTH_ERROR_HEATMAP]

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

		zed_depth = utils.inf_filtering(zed_depth)
		pt_depth = utils.inf_filtering(pt_depth)

		zed_depth[np.isinf(zed_depth)] = np.nan
		zed_depth[zed_depth > 10] = np.nan
		# logging.warning(f"Number of inf values in zed_depth: {np.sum(np.isinf(zed_depth))}")
		
		pt_depth[np.isinf(pt_depth)] = np.nan
		pt_depth[pt_depth > 10] = np.nan
		# logging.warning(f"Number of inf values in pt_depth: {np.sum(np.isinf(pt_depth))}")
		
		error = pt_depth - zed_depth
		# logging.warning(f"Number of inf values in error: {np.sum(np.isinf(error))}")
		# logging.warning(f"error.max: {np.max(error)} error.min: {np.min(error)}")
		# logging.warning(f"np.nanmax(error): {np.nanmax(error)} np.nanmin(error): {np.nanmin(error)}")
		
		filename_npy = os.path.basename(zed_file)
		filename_png = filename_npy.replace('.npy', '.png')	
		utils.write_legend_plot(error, f"{ZED_VS_PT_DEPTH_ERROR_HEATMAP}/{filename_png}")
		utils.plot_histograms([utils.PLT(data=error,
								    title='Depth Error Histogram', 
									xlabel='Depth Error (meters)',
									bins=100, range=(np.nanmin(error), np.nanmax(error)))], 
									save_path=f"{ZED_VS_PT_DEPTH_ERROR_HIST}/{filename_png}", 
									visualize= False)



if __name__ == "__main__":
	coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger
	main()