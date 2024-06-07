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


device = 'cuda'

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



if __name__ == '__main__':

	#left_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png")
	#right_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png")

	comparison_folder ="comparison"
	disparity_comparison = "disparity_comparison"
	
	model_disparity_maps = "model_disparity_maps"
	
	zed_files = "zed_output"
	zed_disparity_maps = "zed_disparity_maps"

	for path in [comparison_folder, model_disparity_maps]:
		try:
			shutil.rmtree(path)
			print(f"Directory '{path}' has been removed successfully.")
		except OSError as e:
			print(f"Error: {e.strerror}")

	os.makedirs( comparison_folder, exist_ok=True)
	os.makedirs( model_disparity_maps, exist_ok=True)

	files = sorted(os.listdir(zed_files))

	left_images = [f for f in files if f.startswith("left")]
	right_images = [f for f in files if f.startswith("right")]

	assert len(left_images) == len(right_images), "The number of left and right images should be equal"
	
	progress_bar = tqdm(total=len(left_images))
	for idx, (left_image, right_image) in enumerate(tqdm(zip(left_images, right_images)), start=0):
		if idx  > 0 :
			break
		
		progress_bar.update(1)

		# print(f"Processing {idx}th frame!")
		left_path = os.path.join(zed_files, left_image)
		right_path = os.path.join(zed_files, right_image)

		left_img = cv2.imread(left_path)	
		right_img = cv2.imread(right_path)
		

		in_h, in_w = left_img.shape[:2]

		# Resize image in case the GPU memory overflows
		eval_h, eval_w = (in_h,in_w)
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

		# Disparity Calculations
		disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t	
		disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
		disp_vis_single_channel = disp_vis.astype("uint8")	
		disp_vis_three_channel = cv2.applyColorMap(disp_vis_single_channel, cv2.COLORMAP_INFERNO)

		
		# cv2 window parameters
		# cv2.namedWindow("Disparity => 1 Channel", cv2.WINDOW_NORMAL)
		# cv2.resizeWindow("Disparity => 1 Channel", 600, 600)
		# cv2.imshow("Disparity => 1 Channel", disp_vis_single_channel)
		# cv2.waitKey(5000)

		# cv2.namedWindow("Disparity => 3 Channel", cv2.WINDOW_NORMAL)
		# cv2.resizeWindow("Disparity => 3 Channel", 600, 600)
		# cv2.imshow("Disparity => 3 Channel", disp_vis_three_channel)
		# cv2.waitKey(5000)

		baseline = 0.13
		focal_length = 1093.5
		
		# Depth Calculations
		# depth_ = (baseline * focal_length) / (disp + 1e-6)
		# depth = cv2.resize(depth_, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t	
		# depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
		# depth_vis_single_channel = depth_vis.astype("uint8")	
		# depth_vis_three_channel = cv2.applyColorMap(depth_vis_single_channel, cv2.COLORMAP_INFERNO)

		# disp - depth Calculations
		# diff = disp - depth
		# diff_vis = (diff - diff.min()) / (diff.max() - diff.min()) * 255.0
		# diff_vis_single_channel = diff_vis.astype("uint8")	
		
		# # cv2 window parameters
		# cv2.namedWindow("Disp - Depth", cv2.WINDOW_NORMAL)
		# cv2.resizeWindow("Disp - Depth", 600, 600)
		# cv2.imshow("Disp - Depth", diff_vis_single_channel)
		# cv2.waitKey(0)

		
		# cv2 window parameters
		# cv2.namedWindow("Depth => 1 Channel", cv2.WINDOW_NORMAL)
		# cv2.resizeWindow("Depth => 1 Channel", 600, 600)
		# cv2.imshow("Depth => 1 Channel", depth_vis_single_channel)
		# cv2.waitKey(5000)

		# print(f"\ndisp_vis_single_channel.shape: {disp_vis_single_channel.shape}")
		# print(f"depth_vis_single_channel.shape: {depth_vis_single_channel.shape}")

		depth_mono = utils.get_mono_depth(disp, baseline, focal_length, t)

		print(f"depth_mono.shape: {depth_mono.shape} disp_vis_single_channel.shape: {disp_vis_single_channel.shape}")

		disp_depth_concat = cv2.hconcat([disp_vis_single_channel, depth_mono])
		# disp_depth_concat = cv2.hconcat([disp_vis_three_channel, depth_vis_three_channel])
		cv2.namedWindow("Disparity vs Depth", cv2.WINDOW_NORMAL)
		cv2.resizeWindow("Disparity vs Depth", 600, 600)
		cv2.imshow("Disparity vs Depth", disp_depth_concat)

		cv2.waitKey(0)

		# zed_disp_map = cv2.imread(f"{zed_disparity_maps}/frame_{idx}.png")
		# cv2.imshow("zed_depth_map", zed_depth_map)
		# cv2.waitKey(0)
		# print(f"type()")
		# print(f"zed_depth_map.shape: {zed_depth_map.shape} disp_vis.shape: {disp_vis.shape}")

		# combined_img = np.hstack((zed_disp_map, disp_vis))
		# #cv2.namedWindow("output", cv2.WINDOW_NORMAL)
		# #cv2.imshow("output", combined_img)
		# cv2.imwrite(f"{model_disparity_maps}/frame_{idx}.png", disp_vis)
		# cv2.imwrite(f"{comparison_folder}/frame_{idx}.jpg", combined_img)
		# #cv2.waitKey()
		
		cv2.destroyAllWindows()

	progress_bar.close()



