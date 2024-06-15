#! /usr/bin/env python3

import numpy as np
import onnxruntime as ort
import coloredlogs, logging
import matplotlib.pyplot as plt
import cv2


if __name__ == "__main__": 
    
    coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger

    # Load the ONNX models
    sess_crestereo = ort.InferenceSession('models/crestereo.onnx')
    sess_cstrereo_without_flow = ort.InferenceSession('models/crestereo_without_flow.onnx')
   
    # Load and preprocess your images
    left = cv2.imread('zed_input/images/left_20.png')
    right = cv2.imread('zed_input/images/right_20.png')

    # (h,w) = left.shape[:2]
    (h,w) = (480,640)

    # Transpose the dimensions and add a batch dimension
    imgL = cv2.resize(left, (w, h), interpolation=cv2.INTER_LINEAR)
    imgR = cv2.resize(right, (w, h), interpolation=cv2.INTER_LINEAR)
    imgL = np.ascontiguousarray(imgL.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
    imgR = np.ascontiguousarray(imgR.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)

    logging.warn(f"imgL.shape: {imgL.shape} imgR.shape: {imgR.shape}")

    imgL_dw2 = cv2.resize(left, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    imgR_dw2 = cv2.resize(right, (w//2, h//2),  interpolation=cv2.INTER_LINEAR)
    imgL_dw2 = np.ascontiguousarray(imgL_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32) 
    imgR_dw2 = np.ascontiguousarray(imgR_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)

    # # Perform inference with the first model
    # pred_flow_dw2 = sess_cstrereo_without_flow.run(None, {'left': imgL_dw2, 'right': imgR_dw2})
    input_names_without_flow = [input.name for input in sess_cstrereo_without_flow.get_inputs()]
    input_feed_without_flow = {name: value for name, value in zip(input_names_without_flow, [imgL_dw2, imgR_dw2])}
    
    # without flow model inference
    pred_flow_dw2 = sess_cstrereo_without_flow.run(None, input_feed_without_flow)


    input_names_with_flow = [input.name for input in sess_crestereo.get_inputs()]
    input_feed_with_flow = {name: value for name, value in zip(input_names_with_flow, [imgL, imgR, pred_flow_dw2[0]])}    
    
    logging.debug(f"input_names_with_flow: {input_names_with_flow}")
    # pred_flow = sess_crestereo.run(None, input_feed)

    pred_flow = sess_crestereo.run(None, input_feed_with_flow)

    # Extract the disparity map
    # pred_disp = pred_flow_dw2[0][0, 0, :, :]
    pred_disp = pred_flow[0][0, 0, :, :]


    plt.imshow(pred_disp, cmap='plasma')
    plt.colorbar()
    plt.show()