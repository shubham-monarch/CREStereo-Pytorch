#! /usr/bin/env python3

import matplotlib.pyplot as plt
import math
from collections import namedtuple  
import numpy as np
import coloredlogs, logging

''' matplotlib based helpers '''



def plot_arrays(*arrays, cmap='viridis', exit_on_q=True):
    """
    Plots each of the input numpy arrays with the specified colormap.
    
    Parameters:
    - arrays: Variable number of numpy arrays to be plotted.
    - cmap: Colormap to be used for plotting the arrays.
    - exit_on_q: If True, allows exiting the plot by pressing 'q'.
    """
    # Set up the plot figure and adjust subplots
    fig, axs = plt.subplots(1, len(arrays), figsize=(5 * len(arrays), 5))
    if len(arrays) == 1:  # Adjust if there's only one array to avoid indexing issues
        axs = [axs]
    
    # Plot each array
    for ax, array in zip(axs, arrays):
        im = ax.imshow(array, cmap=cmap)
        fig.colorbar(im, ax=ax)
        ax.grid(False)  # Disable grid to not obscure the image
    
    # Adjust layout
    plt.tight_layout()
    
    # Enable exit on 'q' press if required
    if exit_on_q:
        def on_key(event):
            if event.key == 'q':
                plt.close(fig)
        fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()


def plot_histograms(*arrays):
    # Determine the number of rows needed (max 2 histograms per row)
    num_arrays = len(arrays)
    num_rows = (num_arrays + 1) // 2  # Calculate rows needed, round up if odd number of arrays
    
    fig, axs = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))  # Adjust figure size as needed
    axs = axs.flatten()  # Flatten the array of axes for easy iteration
    
    for i, array in enumerate(arrays):
        # Plot histogram on the ith subplot
        axs[i].hist(array.flatten(), bins=20, alpha=0.75)  # Adjust bins and alpha as needed
        axs[i].set_title(f'Array {i+1} Histogram')
    
    # Hide any unused subplots if the number of arrays is odd
    if num_arrays % 2 != 0:
        axs[-1].axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close()


# PLT = namedtuple('PLT', ['data', 'title', 'xlabel', 'bins', 'range'])
# def plot_histograms(datasets, save_path=None, visualize=True):

# 	# plts.append(utils.PLT(data=clipped_disp_data_uint8, 
# 	# 					title='clipped uint8',
# 	# 					bins=100,
# 	# 					range=(0, 255)))
# 	plt.figure(figsize=(10, 10))
	
# 	rows = math.ceil(len(datasets) / 2)

# 	for i, dataset in enumerate(datasets):
# 		plt.subplot(rows, 2, i+1)
# 		plt.hist(x=dataset.data.flatten(),  bins=dataset.bins, range=dataset.range)
# 		plt.title(dataset.title)
# 		plt.xlabel(dataset.xlabel)
# 		plt.ylabel('Frequency')

# 	plt.tight_layout()
# 	if save_path:
# 		plt.savefig(save_path)
# 	if visualize:
# 		plt.show()
# 	plt.close()


def plot_histograms_and_arrays(histogram_arrs, plot_arrs):
    # Assert the number of arrays is the same for both inputs
    assert len(histogram_arrs) == len(plot_arrs), "The number of histogram arrays and plot arrays must be the same."
    # logging.warning(f"[plot_histograms_and_arrays]")
    # Ensure axs is always a 2D array
    fig, axs = plt.subplots(2, 4, figsize=(10, 10))
    # if len(histogram_arrs) == 1:
    #     axs = axs.reshape(1, -1)  # Reshape axs to 2D array if only one row
    # logging.warning(f"generated fig,axs")
    # return
    # Plot histograms and arrays
    for i, (hist_arr, plot_arr) in enumerate(zip(histogram_arrs, plot_arrs)):
        # Plot histogram on the first column
        # logging.warning(f"i: {i}")
        # if i % 2: 
            axs[0 , i].hist(hist_arr.flatten(), bins=50, alpha=0.5, label='Init Flow')
            axs[1, i].imshow(plot_arr, cmap="plasma", alpha=0.5, label='Init Flow Map')
        # else: 
        #     axs[1 , i].hist(hist_arr.flatten(), bins=50, alpha=0.5, label='Disp')
        #     axs[0, i].imshow(plot_arr, cmap="plasma", alpha=0.5, label='Disp Map')
             
        # axs[i, 0].legend()

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()