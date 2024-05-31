#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from imread_from_url import imread_from_url

from nets import Model

import sys
import pyzed.sl as sl
import numpy as np
import cv2
import os

device = 'cuda'

#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, n_iter=20):

	print("Model Forwarding...")
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


	svo_file = "svo-files/front_2023-11-03-10-46-17.svo"
	i = 0 


	input_type = sl.InputType()
	input_type.set_from_svo_file(svo_file)
	
	init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
	init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Use ULTRA depth mode
	init_params.coordinate_units = sl.UNIT.METER # Use millimeter units (for depth measurements)

	zed = sl.Camera()
	status = zed.open(init_params)
	
	image_l = sl.Mat()
	image_r = sl.Mat()
	
	depth_map = sl.Mat()
	depth_for_display = sl.Mat()
			
	runtime_parameters = sl.RuntimeParameters()
	runtime_parameters.enable_fill_mode	= True
	
	dir_path = "zed-output"


	while True:

		# SVO PROCESSING 
		print("Doing {}".format(i))
		
		output_dir = os.path.join(dir_path, "frame_{}/images".format(i) )
		os.makedirs( output_dir, exist_ok=True )
		
		if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS :
			
			# retrieve stereo images
			zed.retrieve_image(image_l, sl.VIEW.LEFT) # Retrieve left image
			zed.retrieve_image(image_r, sl.VIEW.LEFT) # Retrieve left image
			
			# retrieve depth map image
			zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH)

			image_l.write( os.path.join(output_dir, 'left_image.png') )
			image_r.write( os.path.join(output_dir, 'right_image.png') )
			
			i = i + 1
			
			#colorize_depth(depth_for_display.get_data()[: , : , :3])
			print(f"i: {i}")
			if i > 10: 
				break

		# MODEL INFERENCING
		left_img = cv2.imread(os.path.join(output_dir, 'left_image.png'))
		right_img = cv2.imread(os.path.join(output_dir, 'right_image.png'))	
		print("Images loaded successfully!")

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

		pred = inference(imgL, imgR, model, n_iter=20)

		t = float(in_w) / float(eval_w)
		disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t

		disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
		disp_vis = disp_vis.astype("uint8")
		disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

		combined_img = np.hstack((left_img, disp_vis))
		cv2.namedWindow("output", cv2.WINDOW_NORMAL)
		cv2.imshow("output", combined_img)
		cv2.imwrite("output/output.jpg", disp_vis)
		cv2.waitKey(0)
