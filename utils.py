#! /usr/bin/env python3

import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt

# TO-DO=>
# - add histograms
# - capitalize dir params
# - jsonize config files



# input -> np.float32 disp_data
def get_depth_data(disp_data, baseline, focal_length): 
	assert disp_data.dtype == np.float32
	depth_data = (baseline * focal_length) / (disp_data + 1e-6)
	return depth_data 	

# removes inf values and normalizes to 255
def uint8_normalization(depth_map):
	assert depth_map.dtype == np.float32
	max_depth = np.max(depth_map[np.isfinite(depth_map)])
	depth_map_finite = np.where(np.isinf(depth_map), max_depth, depth_map)
	depth_map_uint8 = cv2.normalize(depth_map_finite, depth_map_finite, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	return depth_map_uint8

def plot_histogram(image, title):
	plt.hist(image, bins='auto')
	plt.title(title)
	# plt.draw()
	# plt.waitforbuttonpress(0)
	# plt.close('all')
	plt.show() 
	
def inf_filtering(depth_map):
	max_depth = np.max(depth_map[np.isfinite(depth_map)])
	depth_map_finite = np.where(np.isinf(depth_map), max_depth, depth_map)
	return depth_map_finite

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

def percentage_infinite_points(image):
	total_points = image.size
	infinite_points = np.sum(np.isinf(image))
	percentage = (infinite_points / total_points) * 100
	return percentage

