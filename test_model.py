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
import testing

device = 'cuda'

# camera parameters
BASELINE = 0.13
FOCAL_LENGTH = 1093.5

# comparison folders
zed_vs_model_dir = "zed_vs_model"
# zed_vs_model_disp_dir = f"{zed_vs_model_dir}/disparity"
# zed_vs_model_depth_dir = f"{zed_vs_model_dir}/depth"

# model output folders	
# model_directory = "model_output"
# model_disp_maps = f"{model_directory}/disparity_maps"
# model_depth_maps = f"{model_directory}/depth_maps"
# model_disp_vs_depth_maps = f"{model_directory}/disp_vs_depth"	

# zed input folders
zed_input_dir = "zed_input"
zed_input_disp_maps = f"{zed_input_dir}/disparity_maps"
zed_input_depth_maps = f"{zed_input_dir}/depth_maps"
zed_input_images= f"{zed_input_dir}/images"

# zed output folders	
# zed_output_dir = "zed_output"
# zed_disp_maps = f"{zed_output_dir}/disparity_maps"
# zed_depth_maps = f"{zed_output_dir}/depth_maps"
# zed_disp_vs_depth_maps = f"{zed_output_dir}/disp_vs_depth"	


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
	for path in [zed_vs_model_dir]:	
		try:
			shutil.rmtree(path)
			print(f"Directory '{path}' has been removed successsfully.")
		except OSError as e:
			logging.error(f"Error deleting folder: {e.strerror}")

	for path in [zed_vs_model_dir]:
		try:
			os.makedirs(path, exist_ok=True)
		except OSError as e:
			logging.error(f"Error creating folder: {e}")
	
	files = sorted(os.listdir(zed_input_images))

	left_images = [f for f in files if f.startswith("left")]
	right_images = [f for f in files if f.startswith("right")]

	assert len(left_images) == len(right_images), "The number of left and right images should be equal"

	# setting cv2 window	
	cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("TEST", 1000, 1000)
			
	progress_bar = tqdm(total=len(left_images))
	for idx, (left_image, right_image) in enumerate(tqdm(zip(left_images, right_images)), start=0):
		frame_id = int(left_image.split(".")[0].split("_")[1])
		progress_bar.update(1)

		left_path = os.path.join(zed_input_images, left_image)
		right_path = os.path.join(zed_input_images, right_image)

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
		
		# [MODEL] Depth Calculations
		model_disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t	
		
		model_depth_mono = utils.get_mono_depth(model_disp, BASELINE, FOCAL_LENGTH, t)
		model_depth_mono = cv2.cvtColor(model_depth_mono, cv2.COLOR_GRAY2BGR) 
		model_depth_rgb = utils.get_rgb_depth(model_disp, BASELINE, FOCAL_LENGTH, t)
		
		# [ZED] Depth Calculations
		zed_depth = cv2.imread(f"{zed_input_depth_maps}/frame_{frame_id}.png", cv2.IMREAD_GRAYSCALE)
		zed_depth = cv2.cvtColor(zed_depth, cv2.COLOR_GRAY2BGR)
		zed_depth = cv2.resize(zed_depth, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
		zed_depth = (zed_depth - zed_depth.min()) / (zed_depth.max() - zed_depth.min()) * 255.0
		zed_depth = zed_depth.astype(np.uint8)
		
		zed_depth_mono = zed_depth
		zed_depth_rgb = cv2.applyColorMap(zed_depth_mono, cv2.COLORMAP_INFERNO)	
		
		# [ZED vs MODEL] Calculations
		left_img_bgr = left_img
		left_img_mono = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
		left_img_mono = cv2.cvtColor(left_img_mono, cv2.COLOR_GRAY2BGR)

		concat_images_mono = cv2.hconcat([left_img_mono, zed_depth_mono, model_depth_mono])
		concat_images_bgr = cv2.hconcat([left_img_bgr, zed_depth_rgb, model_depth_rgb])
		concat_images = cv2.vconcat([concat_images_bgr, concat_images_mono])
		cv2.imwrite(f"{zed_vs_model_dir}/frame_{frame_id}.png",concat_images)	
		
		# cv2.imshow("TEST", concat_images)
		# cv2.waitKey(0)

		# cv2 cleanup
		cv2.destroyAllWindows()

	progress_bar.close()


if __name__ == '__main__':

	#left_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/left.png")
	#right_img = imread_from_url("https://raw.githubusercontent.com/megvii-research/CREStereo/master/img/test/right.png")

	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	run_model_pipeline()



