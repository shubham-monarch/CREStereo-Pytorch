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



# comparison folders
zed_vs_model_dir = "zed_vs_model"
zed_vs_model_heatmap_dir = f"{zed_vs_model_dir}/heatmap"
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


def INFERENCE(imgL, imgR):
	model_path = "models/crestereo_eth3d.pth"
	model = Model(max_disp=256, mixed_precision=False, test_mode=True)
	model.load_state_dict(torch.load(model_path), strict=True)
	model.to(device)
	model.eval()
	pred = inference(imgL, imgR, model, n_iter=5)
	return pred

if __name__ == '__main__':

	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	


