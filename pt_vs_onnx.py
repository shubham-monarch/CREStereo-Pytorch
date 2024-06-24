#! /usr/bin/env python3


import coloredlogs, logging
import os
import numpy as np 
import matplotlib.pyplot as plt
import utils

ONNX_VS_PYTORCH_DIR = "onnx_vs_pytorch"
ONNX_VS_PYTORCH_PLOTS = f"{ONNX_VS_PYTORCH_DIR}/plots"
PT_INFERENCES_DIR = f"{ONNX_VS_PYTORCH_DIR}/pt_inferences"
ONNX_INFERENCES_DIR = f"{ONNX_VS_PYTORCH_DIR}/onnx_inferences"
FOLDERS_TO_CREATE = [ONNX_VS_PYTORCH_PLOTS]


import coloredlogs, logging
import os
import numpy as np 
import matplotlib.pyplot as plt


def visualize_and_save_disparity_comparisons(pt_file_path, onnx_file_path, save_path):
    # Load the disparity maps
    pt_disparity = np.load(pt_file_path)
    onnx_disparity = np.load(onnx_file_path)
    
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
    plt.show()
    
    # Close the plot to free up memory
    plt.close()

def main():
    pt_files = [os.path.join(PT_INFERENCES_DIR, f) for f in os.listdir(PT_INFERENCES_DIR)]
    onnx_files = [os.path.join(ONNX_INFERENCES_DIR, f) for f in os.listdir(ONNX_INFERENCES_DIR)]
    
    # utils.delete_folders([ONNX_VS_PYTORCH_PLOTS])
    # utils.create_folders([ONNX_VS_PYTORCH_PLOTS])

    utils.delete_folders(FOLDERS_TO_CREATE)
    utils.create_folders(FOLDERS_TO_CREATE)
    
    for i in range(len(pt_files)):
        visualize_and_save_disparity_comparisons(pt_files[i], onnx_files[i], f"{ONNX_VS_PYTORCH_PLOTS}/frame_{i}.png")
    
if __name__ == "__main__": 
    coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger
    main()