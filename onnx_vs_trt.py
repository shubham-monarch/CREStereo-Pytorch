#! /usr/bin/env python3

# standard imports
import coloredlogs, logging
import onnxruntime as ort
import os
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt


# custom imports
import zed_inference
import onnx_inference
import trt_inference as trt_inf
import utils



ZED_IMAGE_DIR = zed_inference.ZED_IMG_DIR
ONNX_VS_TRT_DIR = "onnx_vs_trt"
FOLDERS_TO_CREATE = []

(W,H) = (640, 480)

def visualize_inference_and_histogram(left, onnx_inference_no_flow):
    """
    Visualizes the left image, ONNX inference without flow, and the histogram of the ONNX inference in a single plot.
    Exits on key press.

    Parameters:
    - left: The left image as a numpy array.
    - onnx_inference_no_flow: The ONNX inference result without flow as a numpy array.
    """
    def on_key(event):
        if event.key == 'q':
            plt.close()  # Close the plot window

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display the left image
    axs[0].imshow(left, cmap = "plasma")
    axs[0].set_title('Left Image')
    axs[0].axis('off')  # Hide the axis

    # Display the onnx_inference_no_flow image
    axs[1].imshow(onnx_inference_no_flow)
    axs[1].set_title('ONNX Inference No Flow')
    axs[1].axis('off')  # Hide the axis

    # Calculate and display the histogram of onnx_inference_no_flow
    onnx_inference_no_flow_flat = onnx_inference_no_flow.flatten()
    axs[2].hist(onnx_inference_no_flow_flat, bins=50, color='blue', alpha=0.7)
    axs[2].set_title('Histogram of ONNX Inference No Flow')

    plt.tight_layout()  # Adjust the layout

    fig.canvas.mpl_connect('key_press_event', on_key)  # Connect the key press event to the on_key function

    plt.show()  # Show the plot
def main(num_frames):
	
	utils.delete_folders(FOLDERS_TO_CREATE)
	utils.create_folders(FOLDERS_TO_CREATE)

	image_files_left = [os.path.join(ZED_IMAGE_DIR, f) for f in os.listdir(ZED_IMAGE_DIR) if f.startswith('left_') and f.endswith('.png')]
	image_files_right = [os.path.join(ZED_IMAGE_DIR, f) for f in os.listdir(ZED_IMAGE_DIR) if f.startswith('right_') and f.endswith('.png')]
	
	image_files_left.sort()
	image_files_right.sort()

	assert(len(image_files_left) == len(image_files_right)), "Number of left and right images should be equal"
	assert(len(image_files_left) >= num_frames), "Number of frames should be less than total number of images"

	# loading the onnx models
	sess_crestereo = ort.InferenceSession('models/crestereo.onnx')
	sess_crestereo_no_flow = ort.InferenceSession('models/crestereo_without_flow.onnx')

	# loading the trt engines
	# trt_engine_no_flow = trt_inf.TRTEngine("models/crestereo_without_flow.trt")
	# trt_engine_with_flow = trt_inf.TRTEngine("models/crestereo.trt")
	


	for i in tqdm(range(num_frames)):
		left_img = cv2.imread(image_files_left[i])
		right_img = cv2.imread(image_files_right[i])

		left = cv2.resize(left_img, (W, H), interpolation=cv2.INTER_LINEAR)
		right = cv2.resize(right_img, (W, H), interpolation=cv2.INTER_LINEAR)
	
		# model_inference_onnx = onnx_inference.inference(left_img , right_img, sess_crestereo, sess_crestereo_no_flow, img_shape=(480, 640))   
		onnx_inference_no_flow = onnx_inference.inference_no_flow(left_img , right_img, 
															sess_crestereo, sess_crestereo_no_flow, 
															img_shape=(480, 640))   
		
		onnx_inference_no_flow_reshaped = np.squeeze(onnx_inference_no_flow[:, 0, :, :]) # (H * W)
		logging.warn(f"onnx_inference_no_flow.shape: {onnx_inference_no_flow.shape}")
	
		onnx_inference_with_flow = onnx_inference.inference_with_flow(left_img , right_img, 
															  sess_crestereo, sess_crestereo_no_flow,
															  onnx_inference_no_flow, 
															  img_shape=(480, 640))   
		
		# logging.warning(f"onnx_inference_with_flow.shape: {onnx_inference_with_flow.shape}")
		
		# plt.imshow(onnx_inference_with_flow, cmap='plasma')  # Display the image in grayscale
		# plt.gcf().canvas.mpl_connect('key_press_event', on_key)  # Connect the key press event to the on_key function
		# plt.show()  # Show the plot

		visualize_inference_and_histogram(left, onnx_inference_no_flow_reshaped)

		# # TRT ENGINE => NO FLOW
		# ppi_trt_no_flow = trt_engine_no_flow.pre_process_input([left_img, right_img], 
		# 												 [(H // 2, W // 2), (H // 2, W // 2)])
		# trt_engine_no_flow.load_input(ppi_trt_no_flow)
		# trt_inference_outputs_no_flow =  trt_engine_no_flow.run_trt_inference()
		# trt_inference_output_no_flow = trt_inference_outputs_no_flow[0].reshape(1, 2, H // 2, W // 2)
		# trt_inference_no_flow = trt_inference_output_no_flow

		# # TRT ENGINE => WITH FLOW	
		# ppi_trt = trt_engine_with_flow.pre_process_input([left_img, right_img, trt_inference_no_flow], 
		# 									[(H, W), (H, W), None])
		# trt_engine_with_flow.load_input(ppi_trt)
		# trt_inference_outputs =  trt_engine_with_flow.run_trt_inference()
		# trt_output = trt_inference_outputs[0].reshape(1, 2, H, W)
		# trt_inference = np.squeeze(trt_output[:, 0, :, :]) # (H * W)
		
		

if __name__ == "__main__":
	coloredlogs.install(level="INFO", force=True)  # install a handler on the root logger
	num_frames = 30
	main(num_frames=30)