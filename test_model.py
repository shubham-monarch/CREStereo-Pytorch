#! /usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from imread_from_url import imread_from_url

from nets import Model
import os
import shutil
from tqdm import tqdm
import utils
import coloredlogs, logging
import sys

device = 'cuda'

# camera parameters
BASELINE = 0.13
FOCAL_LENGTH = 1093.5

# comparison folders
disp_comparison_dir = "comparison/disparity"
depth_comparison_dir = "comparison/depth"

# model output folders	
model_directory = "model_output"
model_disp_maps = f"{model_directory}/disparity_maps"
model_depth_maps = f"{model_directory}/depth_maps"
model_disp_vs_depth_maps = f"{model_directory}/disp_vs_depth"	

# zed input folders
zed_input_dir = "zed_input"
zed_input_disp_maps = f"{zed_input_dir}/disparity_maps"
zed_input_depth_maps = f"{zed_input_dir}/depth_maps"
zed_input_images= f"{zed_input_dir}/images"



# zed output folders	
zed_output_dir = "zed_output"
zed_disp_maps = f"{zed_output_dir}/disparity_maps"
zed_depth_maps = f"{zed_output_dir}/depth_maps"
zed_disp_vs_depth_maps = f"{zed_output_dir}/disp_vs_depth"	


#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, n_iter=20):

	torch.cuda.empty_cache()

	# print("Model Forwarding...")
	imgL = left.transpose(2, 0, 1)
	imgR = right.transpose(2, 0, 1)
	imgL = np.ascontiguousarray(imgL[None, :, :, :])
	imgR = np.ascontiguousarray(imgR[None, :, :, :])

	imgL = torch.tensor(imgL.astype("float32")).to(device)
	imgR = torch.tensor(imgR.astype("float32")).to(device)



	imgL_dw2 = F.interpolate(
		imgL,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	imgR_dw2 = F.interpolate(
		imgR,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	
	# print(imgR_dw2.shape)
	with torch.inference_mode():
		pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
		pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
	pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

	return pred_disp

def run_model_pipeline():
	for path in [	
					# model_disp_maps, model_depth_maps, model_disp_vs_depth_maps, \
					zed_disp_maps, zed_depth_maps
					]:
		try:
			shutil.rmtree(path)
			print(f"Directory '{path}' has been removed successsfully.")
		except OSError as e:
			logging.error(f"Error deleting folder: {e.strerror}")

	for path in [disp_comparison_dir, depth_comparison_dir, \
					model_disp_maps, model_depth_maps, model_disp_vs_depth_maps, \
					zed_disp_maps, zed_depth_maps]:
		try:
			os.makedirs(path, exist_ok=True)
		except OSError as e:
			logging.error(f"Error creating folder: {e}")
	
	
	files = sorted(os.listdir(zed_input_images))

	left_images = [f for f in files if f.startswith("left")]
	right_images = [f for f in files if f.startswith("right")]

	assert len(left_images) == len(right_images), "The number of left and right images should be equal"
	
	progress_bar = tqdm(total=len(left_images))
	for idx, (left_image, right_image) in enumerate(tqdm(zip(left_images, right_images)), start=0):
		if idx  > 2 :
			break
		
		logging.debug(f"Processing {idx}th frame!")
		progress_bar.update(1)

		# print(f"Processing {idx}th frame!")
		left_path = os.path.join(zed_input_images, left_image)
		right_path = os.path.join(zed_input_images, right_image)

		left_img = cv2.imread(left_path)	
		right_img = cv2.imread(right_path)
		

		in_h, in_w = left_img.shape[:2]

		print(f"(in_h, in_w): {(in_h, in_w)}")
		# Resize image in case the GPU memory overflows
		# eval_h, eval_w = (in_h,in_w)
		eval_h, eval_w = (520,928)
		assert eval_h%8 == 0, "input height should be divisible by 8"
		assert eval_w%8 == 0, "input width should be divisible by 8"
		
		imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
		imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
		
		model_path = "models/crestereo_eth3d.pth"

		model = Model(max_disp=256, mixed_precision=False, test_mode=True)
		model.load_state_dict(torch.load(model_path), strict=True)
		model.to(device)
		model.eval()

		pred = inference(imgL, imgR, model, n_iter=5)
		t = float(in_w) / float(eval_w)
		
		# [MODEL] Disparity Calculations
		model_disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t	
		model_disp_vis = (model_disp - model_disp.min()) / (model_disp.max() - model_disp.min()) * 255.0
		
		model_disp_mono = model_disp_vis.astype("uint8")	
		model_disp_rgb = cv2.applyColorMap(model_disp_mono, cv2.COLORMAP_INFERNO)
		model_disp_mono_vs_rgb = cv2.hconcat([np.tile(np.expand_dims(model_disp_mono,axis=-1), (1,1,3)), model_disp_rgb])

		# [MODEL] writing grayscale + colored disparity map
		cv2.imwrite(f"{model_disp_maps}/frame_{idx}.png", model_disp_mono_vs_rgb)	

	
		# [MODEL] Depth Calculations
		model_depth_mono = utils.get_mono_depth(model_disp, BASELINE, FOCAL_LENGTH, t)
		model_depth_rgb = utils.get_rgb_depth(model_disp, BASELINE, FOCAL_LENGTH, t)
		model_depth_mono_vs_rgb = cv2.hconcat([np.tile(np.expand_dims(model_depth_mono, axis=-1), (1,1,3)), model_depth_rgb])

		# [MODEL] writing grayscale + colored depth map
		cv2.imwrite(f"{model_depth_maps}/frame_{idx}.png", model_depth_mono_vs_rgb)

		# [MODEL] writing disparity vs [MODEL] depth
		cv2.imwrite(f"{model_disp_vs_depth_maps}/frame_{idx}.png", cv2.vconcat([model_disp_mono_vs_rgb, model_depth_mono_vs_rgb]))


		# # cv2 window parameters
		# cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
		# cv2.resizeWindow("Disparity", 600, 600)
		# cv2.imshow("Disparity", model_disp_rgb)
		# cv2.waitKey(5000)


		# [ZED] Depth Calculations
		zed_depth = cv2.imread(f"{zed_input_depth_maps}/frame_{idx}.png")
		zed_depth = cv2.resize(zed_depth, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t			
		zed_depth_vis = (zed_depth - zed_depth.min()) / (zed_depth.max() - zed_depth.min()) * 255.0

		zed_depth_mono = zed_depth_vis.astype("uint8")	
		zed_depth_rgb = cv2.applyColorMap(zed_depth_mono, cv2.COLORMAP_INFERNO)
		zed_depth_mono_vs_rgb = cv2.hconcat([zed_depth_mono, zed_depth_rgb])

		# [ZED] saving grayscale + colored depth map
		cv2.imwrite(f"{zed_depth_maps}/frame_{idx}.png", zed_depth_mono_vs_rgb)

		# # # [ZED] Disparity Calculations
		# zed_disp = cv2.imread(f"{zed_input_depth_maps}/frame_{idx}.png")
		# zed_disp = cv2.resize(zed_disp, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t	
		
		# # # print(f"type(zed_disp): {type(zed_disp)} zed_disp.shape: {zed_disp.shape}")
		
		# zed_disp_vis = (zed_disp - zed_disp.min()) / (zed_disp.max() - zed_disp.min()) * 255.0
		
		# zed_disp_mono = zed_disp_vis.astype("uint8")	
		# zed_disp_rgb = cv2.applyColorMap(zed_disp_mono, cv2.COLORMAP_INFERNO)
		# logging.debug(f"zed_disp_mono.shape: {np.expand_dims(zed_disp_mono, axis=-1).shape} zed_disp_rgb.shape: {zed_disp_rgb.shape}")	
		# zed_disp_mono_vs_rgb = cv2.hconcat([zed_disp_mono, zed_disp_rgb])

		# # [ZED] saving grayscale + colored disparity map
		# cv2.imwrite(f"{zed_disparity_maps}/frame_{idx}.png", zed_disp_rgb)

		
		
		# # saving [ZED] vs [MODEL] Disparity	
		# # logging.debug(f"zed_disp_mono.shape: {zed_disp_mono.shape} model_disp_mono.shape: {model_disp_mono.shape}")
		# zed_vs_model_disp_mono = cv2.hconcat([zed_disp_mono[:, :, ], model_disp_mono])
		# zed_vs_model_disp_rgb = cv2.hconcat([zed_disp_rgb, model_disp_rgb])
		# zed_vs_model_disp = cv2.vconcat([zed_vs_model_disp_mono, zed_vs_model_disp_rgb])
		# cv2.imwrite(f"{disparity_comparison_dir}/frame_{idx}.png", zed_vs_model_disp)

		# # saving [ZED] vs [MODEL] Depth
		# zed_vs_model_depth_mono = cv2.hconcat([zed_depth_mono, model_depth_mono])
		# zed_vs_model_depth_rgb = cv2.hconcat([zed_depth_rgb, model_depth_rgb])
		# zed_vs_model_depth = cv2.vconcat([zed_vs_model_depth_mono, zed_vs_model_depth_rgb])
		# cv2.imwrite(f"{depth_comparison_dir}/frame_{idx}.png", zed_vs_model_depth)


		
		# # cv2 window parameters
		# # cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
		# # cv2.resizeWindow("Disparity", 600, 600)
		# # cv2.imshow("Disparity", zed_disp_rgb)
		# # cv2.waitKey(5000)	

		# # # [DISPARITY] ZED vs Model 
		# # disp_comp = cv2.hconcat([zed_disp_rgb, model_disp_rgb])
		# # # disp_depth_concat = cv2.hconcat([disp_vis_three_channel, depth_rgb])
		# # cv2.namedWindow("[Disparity] ZED vs Model", cv2.WINDOW_NORMAL)
		# # cv2.resizeWindow("[Disparity] ZED vs Model", 600, 600)
		# # cv2.imshow("[Disparity] ZED vs Model", disp_comp)	
		# # cv2.waitKey(5000)

		# # [DEPTH] ZED vs Model
		# # depth_comp = cv2.hconcat([zed_depth_rgb, model_depth_rgb])
		# depth_comp = cv2.hconcat([zed_depth_mono, model_depth_mono])
		# cv2.namedWindow("[Depth] ZED vs Model", cv2.WINDOW_NORMAL)
		# cv2.resizeWindow("[Depth] ZED vs Model", 600, 600)
		# cv2.imshow("[Depth] ZED vs Model", depth_comp)
		# cv2.waitKey(0)


		# # cv2.namedWindow("Disparity => 3 Channel", cv2.WINDOW_NORMAL)
		# # cv2.resizeWindow("Disparity => 3 Channel", 600, 600)
		# # cv2.imshow("Disparity => 3 Channel", disp_vis_three_channel)
		# # cv2.waitKey(5000)

		
		# # depth_mono = utils.get_mono_depth(disp, baseline, focal_length, t)
		# # depth_rgb = utils.get_rgb_depth(disp, baseline, focal_length, t)	

		# # print(f"depth_mono.shape: {depth_mono.shape} disp_vis_single_channel.shape: {disp_vis_single_channel.shape}")

		# # # disp_depth_concat = cv2.hconcat([disp_vis_single_channel, depth_mono])
		# # disp_depth_concat = cv2.hconcat([disp_vis_three_channel, depth_rgb])
		# # cv2.namedWindow("Disparity vs Depth", cv2.WINDOW_NORMAL)
		# # cv2.resizeWindow("Disparity vs Depth", 600, 600)
		# # cv2.imshow("Disparity vs Depth", disp_depth_concat)

		# # cv2.waitKey(0)

		# # zed_disp_map = cv2.imread(f"{zed_disparity_maps}/frame_{idx}.png")
		# # cv2.imshow("zed_depth_map", zed_depth_map)
		# # cv2.waitKey(0)
		# # print(f"type()")
		# # print(f"zed_depth_map.shape: {zed_depth_map.shape} disp_vis.shape: {disp_vis.shape}")

		# # combined_img = np.hstack((zed_disp_map, disp_vis))
		# # #cv2.namedWindow("output", cv2.WINDOW_NORMAL)
		# # #cv2.imshow("output", combined_img)
		# # cv2.imwrite(f"{model_disparity_maps}/frame_{idx}.png", disp_vis)
		# # cv2.imwrite(f"{comparison_folder}/frame_{idx}.jpg", combined_img)
		# # #cv2.waitKey()
		
		cv2.destroyAllWindows()

	progress_bar.close()


if __name__ == '__main__':

	#left_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png")
	#right_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png")

	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	run_model_pipeline()



