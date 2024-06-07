#! /usr/bin/env python3

import cv2

def get_mono_depth(disp, baseline, focal_length, gpu_t):
    depth_ = (baseline * focal_length) / (disp + 1e-6)
    depth = cv2.resize(depth_, disp.shape[::-1], interpolation=cv2.INTER_LINEAR) * gpu_t	
    depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    return depth_vis.astype("uint8")

def get_rgb_depth(disp, baseline, focal_length, gpu_t):
    mono_depth = get_mono_depth(disp, baseline, focal_length, gpu_t)
    return cv2.applyColorMap(mono_depth, cv2.COLORMAP_INFERNO)


