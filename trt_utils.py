#! /usr/bin/env python3

import time
import pycuda.driver as cuda 
import tensorrt as trt
import os
import coloredlogs, logging
import numpy as np
from tensorrt import TensorIOMode
import utils

# input.shape => (3, 480, 640)
def reshape_input_image(input, shape =(3, 480//2, 640//2), batch_size=1):
    '''
    Reshape the input image(1D) to the desired shape(1D -> 4D -> 2D)
    '''
    logging.debug(f"input.shape: {input.shape}")
    new_shape = (batch_size,) + shape
    logging.debug(f"new_shape: {new_shape}")
    img = np.reshape(input, new_shape)
    img = np.squeeze(img[:, 0, :, :])
    assert img.ndim == 2
    img_mono_channel = utils.uint8_normalization(img)
    return img_mono_channel

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class FPS:
    def __init__(self):
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"

    def start(self):
        self.prev_time = time.time()

    def stop(self):
        self.curr_time = time.time()
        exec_time = self.curr_time - self.prev_time
        self.prev_time = self.curr_time
        self.accum_time += exec_time

    def get_fps(self):
        self.curr_fps += 1
        if self.accum_time > 1:
            self.accum_time -= 1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0
        return self.fps




# def generate_engine_from_onnx(onnx_file_path: str):
# 	with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
# 		config = builder.create_builder_config()
# 		config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
# 		with open(onnx_file_path, 'rb') as model:
# 			if not parser.parse(model.read()):
# 				print ('ERROR: Failed to parse the ONNX file.')
# 				for error in range(parser.num_errors):
# 					print (parser.get_error(error))
# 				return None

# 		serialized_engine = builder.build_serialized_network(network, config)

# 		logging.error(f"serialized_engine_is_null: {serialized_engine is None}")
# 		logging.error(f"config is null: {config is None}")
# 		logging.error(f"network is null: {network is None}")

# 		engine = runtime.deserialize_cuda_engine(serialized_engine)
		
# 		engine_file_path = onnx_file_path.replace(".onnx", ".trt")
# 		with open(engine_file_path, "wb") as f:
# 			f.write(engine.serialize())

# 		return engine

    # self.inputs, self.outputs, self.bindings = [], [], []
    #     self.stream = cuda.Stream()
    #     for binding in engine:
    #         size = trt.volume(engine.get_binding_shape(binding))
    #         dtype = trt.nptype(engine.get_binding_dtype(binding))
    #         host_mem = cuda.pagelocked_empty(size, dtype)
    #         device_mem = cuda.mem_alloc(host_mem.nbytes)
    #         self.bindings.append(int(device_mem))
    #         if engine.binding_is_input(binding):
    #             self.inputs.append({'host': host_mem, 'device': device_mem})
    #         else:
    #             self.outputs.append({'host': host_mem, 'device': device_mem})