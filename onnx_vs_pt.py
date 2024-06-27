#! /usr/bin/env python3


import coloredlogs, logging
import os
import numpy as np 
import matplotlib.pyplot as plt
import utils
import coloredlogs, logging
import os
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import onnx_inference
import pt_inference


ONNX_VS_PYTORCH_PLOTS = f"{onnx_inference.ONNX_VS_PYTORCH_DIR}/plots_onnx_vs_pt"
FOLDERS_TO_CREATE = [ONNX_VS_PYTORCH_PLOTS]

def visualize_and_save_disparity_comparisons(pt_file_path, onnx_file_path, save_path):
    # Load the disparity maps
    pt_disparity = np.load(pt_file_path)
    onnx_disparity = np.load(onnx_file_path)
    
    # logging.warning(f"pt_disparity.shape: {pt_disparity.shape} onnx_disparity.shape: {onnx_disparity.shape}")   
    # Compute the difference
    disparity_difference = (pt_disparity - onnx_disparity)
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot PT disparity
    im0 = axs[0].imshow(pt_disparity, cmap='plasma')
    axs[0].set_title('PT Disparity')
    axs[0].axis('off')  # Hide axes for better visualization
    fig.colorbar(im0, ax=axs[0], orientation='vertical', fraction=0.046, pad=0.04).set_label('Disparity Value')
    
    # Plot ONNX disparity
    im1 = axs[1].imshow(onnx_disparity, cmap='plasma')
    axs[1].set_title('ONNX Disparity')
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs[1], orientation='vertical', fraction=0.046, pad=0.04).set_label('Disparity Value')
    
    # Plot Disparity Difference
    im2 = axs[2].imshow(disparity_difference, cmap='plasma')
    axs[2].set_title('Disparity Difference')
    axs[2].axis('off')
    cbar = fig.colorbar(im2, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Difference Value')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(save_path)
    
    # Optionally display the plot
    # plt.show()
    
    # Close the plot to free up memory
    plt.close()

def main():

    pt_files = [] 
    onnx_files = []
    
    try:
        pt_files = [os.path.join(pt_inference.PT_DISPARITY_DIR, f) for f in os.listdir(pt_inference.PT_DISPARITY_DIR)]
    except Exception as e:
        logging.error(f"Error accessing PT_DISPARITY_DIR: {e}")
        
    try:
        onnx_files = [os.path.join(onnx_inference.ONNX_DISPARITY_DIR, f) for f in os.listdir(onnx_inference.ONNX_DISPARITY_DIR)]
    except Exception as e:
        logging.error(f"Error accessing ONNX_DISPARITY_DIR: {e}")

    assert(len(pt_files) > 0), "No disparity maps found in PT_DISPARITY_DIR"
    assert(len(onnx_files) > 0), "No disparity maps found in ONNX_DISPARITY_DIR"
    assert(len(pt_files) == len(onnx_files)), f"Number of PT({len(pt_files)}) and ONNX disparity maps({len(onnx_files)}) should be equal"
    
    utils.delete_folders(FOLDERS_TO_CREATE)
    utils.create_folders(FOLDERS_TO_CREATE)
    
    for i in tqdm(range(len(pt_files))):
        npy_filename = os.path.basename(pt_files[i])
        png_filename = npy_filename.replace('.npy', '.png')
        # logging.warn(f"file_name: {npy_filename} png_filename: {png_filename}")    
        visualize_and_save_disparity_comparisons(pt_files[i], onnx_files[i], f"{ONNX_VS_PYTORCH_PLOTS}/{png_filename}")
    
if __name__ == "__main__": 
    coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger
    main()