#! /usr/bin/env python3

import cv2
import numpy as np

# TODO: add assert for disp.shape
def get_mono_depth(disp, baseline, focal_length, gpu_t):
	assert(disp.ndim == 2)
	depth_ = (baseline * focal_length) / (disp + 1e-6)
	depth = cv2.resize(depth_, disp.shape[::-1], interpolation=cv2.INTER_LINEAR) * gpu_t	
	depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
	return depth_vis.astype("uint8")

def get_rgb_depth(disp, baseline, focal_length, gpu_t):
	mono_depth = get_mono_depth(disp, baseline, focal_length, gpu_t)
	return cv2.applyColorMap(mono_depth, cv2.COLORMAP_INFERNO)

def get_mono_disparity(depth, baseline, focal_length, gpu_t):
	assert(depth.ndim == 2)
	depth = depth.astype(np.float64)
	disp_ = (baseline * focal_length) / (depth + 1e-6)
	disp = cv2.resize(disp_, depth.shape[::-1], interpolation=cv2.INTER_LINEAR) * gpu_t	
	disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
	return disp_vis.astype("uint8")

def add_label(image, label):
	position = (50, 50)  # (x, y)
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 1
	color = (255, 255, 255)  # white
	thickness = 2
	cv2.putText(image, label, position, font, scale, color, thickness)
	return image

