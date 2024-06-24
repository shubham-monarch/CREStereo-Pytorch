#! /usr/bin/env python3 

import coloredlogs, logging
import zed_inference
import pt_inference
import utils 

ZED_PCL_DIR = zed_inference.ZED_PCL_DIR
PT_PCL_DIR = pt_inference.PT_PCL_DIR

ZED_VS_PT_DIR = "zed_vs_pt"
ZED_VS_PT_PCL_DIR = f"{ZED_VS_PT_DIR}/pcl_error"

FOLDERS_TO_CREATE = [ZED_VS_PT_PCL_DIR]

def main(): 	
	
	utils.delete_folders(FOLDERS_TO_CREATE)
	utils.create_folders(FOLDERS_TO_CREATE)
	
	




if __name__ == "__main__":
	# svo_file = "svo-files/front_2024-05-15-18-59-18.svo"
	# num_frames = 5
	# logging.info(f"Running zed_vs_pt for {num_frames} frames.")
	# main(svo_file, num_frames)

	main()