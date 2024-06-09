#! /usr/bin/env python3

import cv2
import numpy as np
import logging


def get_mono_depth(disp, baseline, focal_length, gpu_t):
	assert(disp.ndim == 2)
	depth_ = (baseline * focal_length) / (disp + 1e-6)
	depth = cv2.resize(depth_, disp.shape[::-1], interpolation=cv2.INTER_LINEAR) * gpu_t	
	depth_vis = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	# depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
	return depth_vis.astype("uint8")

def get_rgb_depth(disp, baseline, focal_length, gpu_t):
	mono_depth = get_mono_depth(disp, baseline, focal_length, gpu_t)
	# return cv2.applyColorMap(mono_depth, cv2.COLORMAP_INFERNO)
	return cv2.applyColorMap(mono_depth, cv2.COLORMAP_JET)

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

# crops the upper height percent of the image
def crop_image(image, height_percent, width_percent):
	height, width = image.shape[:2]
	new_width = int(width * width_percent)
	new_height = int(height * height_percent)
	logging.debug(f"(new_height, new_width): ({new_height}, {new_width}")
	cropped_image = image[new_height:height, 0:new_width]
	logging.debug(f"cropped_image.shape: {cropped_image.shape}")
	return cropped_image


def is_grayscale(image):
	grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	grayscale_image_bgr = cv2.cvtColor(grayscale_image, cv2.COLOR_GRAY2BGR)
	return np.array_equal(image, grayscale_image_bgr)


def get_error_heatmap(model_depth_map, zed_depth_map):
	depth_error = cv2.absdiff(model_depth_map, zed_depth_map)
	depth_error = cv2.normalize(depth_error, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	return depth_error	

def percentage_infinite_points(image):
	total_points = image.size
	infinite_points = np.sum(np.isinf(image))
	percentage = (infinite_points / total_points) * 100
	return percentage

def uint8_normalization(depth_map):
	max_depth = np.max(depth_map[np.isfinite(depth_map)])
	depth_map_finite = np.where(np.isinf(depth_map), max_depth, depth_map)
	depth_map_8U = cv2.normalize(depth_map_finite, depth_map_finite, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	return depth_map_8U