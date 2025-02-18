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
import disparity2pcl
import zed_inference

device = 'cuda'


BASELINE = 0.13
FOCAL_LENGTH = 1093.5

ZED_IMAGE_DIR = zed_inference.ZED_IMG_DIR
ONNX_VS_PYTORCH_DIR = "onnx_vs_pytorch"
ZED_VS_PT_DIR = "zed_vs_pt"

PT_DISPARITY_DIR = f"{ONNX_VS_PYTORCH_DIR}/pt_disparity"
PT_DEPTH_MAP_DIR = f"{ZED_VS_PT_DIR}/pt_depth_map"
PT_PCL_DIR = f"{ZED_VS_PT_DIR}/pt_pcl"	

FOLDERS_TO_CREATE = [PT_DISPARITY_DIR, PT_DEPTH_MAP_DIR,PT_PCL_DIR]


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


def INFERENCE(imgL, imgR):
	model_path = "models/crestereo_eth3d.pth"
	model = Model(max_disp=256, mixed_precision=False, test_mode=True)
	model.load_state_dict(torch.load(model_path), strict=True)
	model.to(device)
	model.eval()
	pred = inference(imgL, imgR, model, n_iter=5)
	return pred


def main(num_frames, H, W):
	utils.delete_folders(FOLDERS_TO_CREATE)
	utils.create_folders(FOLDERS_TO_CREATE)

	image_files_left = [os.path.join(ZED_IMAGE_DIR, f) for f in os.listdir(ZED_IMAGE_DIR) if f.startswith('left_') and f.endswith('.png')]
	image_files_right = [os.path.join(ZED_IMAGE_DIR, f) for f in os.listdir(ZED_IMAGE_DIR) if f.startswith('right_') and f.endswith('.png')]
	
	image_files_left.sort()
	image_files_right.sort()

	assert(len(image_files_left) == len(image_files_right)), "Number of left and right images should be equal"
	assert(len(image_files_left) >= num_frames), "Number of frames should be less than total number of images"
	
	for i in tqdm(range(num_frames)):
		left_img = cv2.imread(image_files_left[i])
		right_img = cv2.imread(image_files_right[i])

		# imgL = left_img
		# imgR = right_img

		imgL = cv2.resize(left_img, (W, H), interpolation=cv2.INTER_LINEAR)	
		imgR = cv2.resize(right_img, (W, H), interpolation=cv2.INTER_LINEAR)	
		
		pred = INFERENCE(imgL, imgR)
		img_name = os.path.basename(image_files_left[i])
		npy_name = img_name.replace('.png', '.npy')
		np.save(f"{PT_DISPARITY_DIR}/{npy_name}", pred)

		# logging.warning(f"left.shape: {imgL.shape} pred.shape: {pred.shape}")

		depth = utils.get_depth_data(pred, BASELINE, FOCAL_LENGTH)
		np.save(f"{PT_DEPTH_MAP_DIR}/{npy_name}", depth)

		points, colors = disparity2pcl.main(imgL, imgR, pred)	
		# logging.warning(f"[pt_inference.py] colors.dtype: {colors.dtype} colors.shape: {colors.shape}")
		# logging.warning(f"[pt_inference.py] colors[:30]: \n{colors[:30]}")
		filename_ply = img_name.replace('.png', '.ply')
		utils.save_npy_as_ply(f"{PT_PCL_DIR}/{filename_ply}",  points, colors)


if __name__ == '__main__':

	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	main(num_frames=10, H=480, W=640)	


