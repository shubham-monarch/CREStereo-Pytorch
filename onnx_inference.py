#! /usr/bin/env python3

import numpy as np
import onnxruntime as rt
import coloredlogs, logging
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__": 
    
    coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger

    # Load the ONNX model
    sess = rt.InferenceSession("crestereo.onnx")

    # Create a dummy input for the model. Replace this with your actual data.
    # Assuming your model takes two images of shape (3, 224, 224) and a flow_init of shape (2, 224, 224)
    # left = np.random.random((1, 3, 480, 640)).astype(np.float32)
    # right = np.random.random((1, 3, 480, 640)).astype(np.float32)
    flow_init = np.random.random((1, 2, 240, 320)).astype(np.float32)

    left = cv2.imread(f"zed_input/images/left_20.png")
    right = cv2.imread(f"zed_input/images/right_20.png")

    dim = (640,480)
    left = cv2.resize(left, dim, interpolation = cv2.INTER_LINEAR).astype(np.float32)
    right = cv2.resize(right, dim, interpolation = cv2.INTER_LINEAR).astype(np.float32)

    # Transpose the dimensions from (height, width, channels) to (channels, height, width)
    left = np.transpose(left, (2, 0, 1))
    right = np.transpose(right, (2, 0, 1))

    # Add an extra dimension for batch size
    left = left[np.newaxis, :]
    right = right[np.newaxis, :]
    
    logging.debug(f"type(left): {type(left)} left.dtype: {left.dtype} left.shape: {left.shape}")

    # scv2.imshow("left", left)    
    # cv2.imshow("right", right)    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Run the model (replace 'left', 'right', 'flow_init' with the names of your actual inputs)
    input_names = [input.name for input in sess.get_inputs()]
    # input_name = sess.get_inputs()[0].name
    # output_name = sess.get_outputs()[0].name
    
    input_feed = {name: value for name, value in zip(input_names, [left, right, flow_init])}

    # logging.debug(f"input_feed: {input_feed} type: {type(input_feed)}")

    # logging.debug(f"input_name: {input_name} type: {type(input_name)}")
    # logging.debug(f"input_names: {input_names} type: {type(input_names)}")
    # logging.debug(f"output_name: {output_name} type: {type(output_name)}")

    

    result = sess.run(None, input_feed)
    logging.debug(f"type(result): {type(result[0])}")
    logging.info(f"len(result): {len(result)}")
    result_elem = result[0]

    logging.info(f"result_elem.shape: {result_elem.shape}")
    
    # logging.debug("result: ", result)

    # Print the result
    # print(result)

    # result_elem.shape: (1, 2, 480, 640)
        # Assuming result_elem is your disparity map
    disparity_map = result_elem[0, 0, :, :]  # Select the first disparity map in the batch

    plt.imshow(disparity_map, cmap='plasma')
    plt.colorbar()
    plt.show()