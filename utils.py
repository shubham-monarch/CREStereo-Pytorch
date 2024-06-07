#! /usr/bin/env python3

import cv2
import numpy as np


import numpy as np
import cv2

def get_mono_depth(disp, baseline, focal_length, gpu_t):
    """
    Calculate and visualize depth from disparity map.

    Args:
        disp (numpy.ndarray): Disparity map (single or multi-channel).
        baseline (float): Baseline distance between cameras.
        focal_length (float): Focal length of the camera.
        gpu_t (float): GPU time or scale factor.

    Returns:
        numpy.ndarray: Depth visualization with same shape as input disparity.
    """
    # Ensure disp is a numpy array
    disp = np.asarray(disp)

    # Handle multi-channel arrays
    if disp.ndim == 3:
        # Assuming the last dimension is the channel
        depth_channels = []
        for channel in range(disp.shape[2]):
            depth_channel = (baseline * focal_length) / (disp[..., channel] + 1e-6)
            depth_channels.append(depth_channel)
        depth = np.stack(depth_channels, axis=-1)
    else:
        # Single channel
        depth = (baseline * focal_length) / (disp + 1e-6)

    # Scale depth by gpu_t (assuming it's a scale factor)
    depth *= gpu_t

    # Normalize depth for visualization (channel-wise if multi-channel)
    if depth.ndim == 3:
        depth_vis = np.zeros_like(depth, dtype=np.uint8)
        for channel in range(depth.shape[2]):
            channel_data = depth[..., channel]
            depth_vis[..., channel] = ((channel_data - channel_data.min()) / 
                                      (channel_data.max() - channel_data.min() + 1e-6) * 255.0)
    else:
        depth_vis = ((depth - depth.min()) / 
                    (depth.max() - depth.min() + 1e-6) * 255.0).astype(np.uint8)

    return depth_vis

# # TODO: add assert for disp.shape
# def get_mono_depth(disp, baseline, focal_length, gpu_t):
#     # Compute the depth map from the first channel of the input array
#     depth_ = (baseline * focal_length) / (disp[..., 0] + 1e-6)

#     # If the input array is multi-channel, create a new array with the same shape and number of channels
#     if disp.ndim > 2:
#         depth = np.zeros_like(disp)
#         depth[..., 0] = depth_
#     else:
#         depth = depth_

#     # Resize the depth map to match the shape of the input disparity map
#     depth = cv2.resize(depth, disp.shape[::-1], interpolation=cv2.INTER_LINEAR) * gpu_t

#     # Normalize the depth map to the range [0, 255]
#     depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

#     # Return the normalized depth map with the same dimensions as the input array
#     return depth_vis.astype("uint8")


def get_rgb_depth(disp, baseline, focal_length, gpu_t):
    mono_depth = get_mono_depth(disp, baseline, focal_length, gpu_t)
    return cv2.applyColorMap(mono_depth, cv2.COLORMAP_INFERNO)


