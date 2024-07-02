#! /usr/bin/env python3 

import coloredlogs, logging
import zed_inference
import pt_inference
import utils 
import os
import numpy as np
from tqdm import tqdm
import cv2

# custom imports
import trt_inference
import onnx_inference
import utils_matplotlib


# ZED_DEP = zed_inference.ZED_PCL_DIR
# PT_DEPTH_MAP_DIR = pt_inference.PT_DEPTH_MAP_DIR

# ZED_VS_PT_DIR = "zed_vs_pt"
# ZED_VS_PT_DEPTH_ERROR_DIR = f"{ZED_VS_PT_DIR}/depth_error"
# ZED_VS_PT_DEPTH_ERROR_HIST = f"{ZED_VS_PT_DEPTH_ERROR_DIR}/depth_error_histograms"
# ZED_VS_PT_DEPTH_ERROR_HEATMAP = f"{ZED_VS_PT_DEPTH_ERROR_DIR}/depth_error_heatmaps"

ONNX_VS_TRT_DEPTH_ERROR_DIR = f"{trt_inference.ONNX_VS_TRT_DIR}/depth_error"
ONNX_VS_TRT_DEPTH_ERROR_HIST = f"{ONNX_VS_TRT_DEPTH_ERROR_DIR}/depth_error_histograms"
ONNX_VS_TRT_DEPTH_ERROR_HEATMAP = f"{ONNX_VS_TRT_DEPTH_ERROR_DIR}/depth_error_heatmaps"

FOLDERS_TO_CREATE = [ONNX_VS_TRT_DEPTH_ERROR_HIST, ONNX_VS_TRT_DEPTH_ERROR_HEATMAP]

def main(): 	
	
	utils.delete_folders(FOLDERS_TO_CREATE)
	utils.create_folders(FOLDERS_TO_CREATE)

	onnx_files = [] 
	trt_files = [] 

	onnx_files = [os.path.join(onnx_inference.ONNX_DEPTH_MAP_DIR, f) for f in os.listdir(onnx_inference.ONNX_DEPTH_MAP_DIR) if f.endswith('.npy')]
	trt_files = [os.path.join(trt_inference.TRT_DEPTH_DIR, f) for f in os.listdir(trt_inference.TRT_DEPTH_DIR) if f.endswith('.npy')]

	onnx_files = sorted(onnx_files)
	trt_files = sorted(trt_files)

	assert len(onnx_files) == len(trt_files), f"Number of files in ONNX({len(onnx_files)}) and TRT directories({len(trt_files)}) should be the same."

	for onnx_file, trt_file in tqdm(zip(onnx_files, trt_files), total = len(onnx_files)):
		onnx_depth = np.load(onnx_file)
		trt_depth = np.load(trt_file)

		onnx_depth = utils.inf_filtering(onnx_depth)
		trt_depth = utils.inf_filtering(trt_depth)

		onnx_depth[np.isinf(onnx_depth)] = np.nan
		onnx_depth[onnx_depth > 10] = np.nan
		# logging.warning(f"Number of inf values in zed_depth: {np.sum(np.isinf(zed_depth))}")
		
		trt_depth[np.isinf(trt_depth)] = np.nan
		trt_depth[trt_depth > 10] = np.nan
		# logging.warning(f"Number of inf values in pt_depth: {np.sum(np.isinf(pt_depth))}")
		
		error = trt_depth - onnx_depth
		# logging.warning(f"Number of inf values in error: {np.sum(np.isinf(error))}")
		# logging.warning(f"error.max: {np.max(error)} error.min: {np.min(error)}")
		# logging.warning(f"np.nanmax(error): {np.nanmax(error)} np.nanmin(error): {np.nanmin(error)}")
		
		filename_npy = os.path.basename(onnx_file)
		filename_png = filename_npy.replace('.npy', '.png')	
		utils.write_legend_plot(error, f"{ONNX_VS_TRT_DEPTH_ERROR_HEATMAP}/{filename_png}")
		utils_matplotlib.plot_histograms([utils_matplotlib.PLT(data=error,
								    title='Depth Error Histogram', 
									xlabel='Depth Error (meters)',
									bins=100, range=(np.nanmin(error), np.nanmax(error)))], 
									save_path=f"{ONNX_VS_TRT_DEPTH_ERROR_HIST}/{filename_png}", 
									visualize= False)


if __name__ == "__main__":
	coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger
	main()