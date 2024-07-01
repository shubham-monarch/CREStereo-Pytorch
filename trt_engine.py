#! /usr/bin/env python

import coloredlogs, logging
import tensorrt as trt
import onnx
import onnx_tensorrt.backend as backend
import numpy as np
import cv2
import os
import trt_utils
import pycuda.driver as cuda
import pycuda.autoinit
import utils
import matplotlib.pyplot as plt
import fnmatch
from tqdm import tqdm
import random
import time
import ctypes

# custom imports
import zed_inference
import utils_matplotlib


# ssimport pycuda.driver as cuda


# TO-DO
# - using polygraphy for quick onnx model checking
# - refer official nvidia docs for best practices
# - explore onnx-graphsurgeon / polygraphy 
# - read onnxsim => removes unsupported operations -> both python / cli apis
# - intergrate onnxsimp api to code
# - explore FPS() class -> YOLO -> https://github.com/sithu31296/PyTorch-ONNX-TRT/blob/master/examples/yolov4/yolov4/scripts/infer.py
# - package structure from YOLO 
# - add trt_info functions -> https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer#workflow
# - add pre / post processing
# - addded batch_inference
# - refactor code based on official nvidia examples
# - add cuda loading in .ppth to .onnx conversion script
# - reload output to torch for further processing
# - fix negative values in flow_init **
# - improve output normalization **
# - refactor TRTEngine
# - explore dynamic shapess
# - altenative to using two trt engines
# - asynchronous vs synchronous inference
# - check trt model inference accuracy
# - config->setMemoryPoolLimit(MemoryPoolType::kTACTIC_SHARED_MEMORY, 48 << 10);
# - try onxx vs pytorch inference comparsion 
# - using fp32 instead of tf32
# - simplification using polygraphy instead of onnx-simp
# - better normalization 
# - check onnx inference distribtion vs trt inference output distribution 




# trt-inference constants
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
runtime = trt.Runtime(TRT_LOGGER)



def trt_inference(engine_path, left_img, right_img, H, W):
	with open(engine_path , "rb") as f, \
			trt.Runtime(TRT_LOGGER) as runtime, \
			runtime.deserialize_cuda_engine(f.read()) as engine, \
			engine.create_execution_context() as context:
		
		d_inputs = []
		
		for idx, binding in enumerate(engine):
			logging.error(f"binding: {binding}")
			shape = engine.get_tensor_shape(binding)
			logging.warning(f"shape: {shape}")
			size = trt.volume(shape)
			logging.warning(f"size: {size}")
			size_nbytes = size * trt.int32.itemsize
			logging.warning(f"size_nbytes: {size_nbytes}")
			dtype = trt.nptype(engine.get_tensor_dtype(binding))
			logging.warning(f"dtype: {dtype}")
			if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
				context.set_input_shape(binding, shape)
				d_inputs.append(cuda.mem_alloc(size_nbytes))

		stream = cuda.Stream()
		assert context.all_binding_shapes_specified

		h_output = cuda.pagelocked_empty(tuple(context.get_tensor_shape(engine.get_tensor_name(2))), dtype=np.float32)
		d_output = cuda.mem_alloc(h_output.nbytes)

		# pre-process input
		left = cv2.resize(left_img, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
		left = np.ascontiguousarray(left.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)		
		left = cuda.register_host_memory(np.ascontiguousarray(left.ravel()))
		
		right = cv2.resize(right_img, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
		right = np.ascontiguousarray(right.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
		right = cuda.register_host_memory(np.ascontiguousarray(right.ravel()))

		cuda.memcpy_htod_async(d_inputs[0], left, stream)
		cuda.memcpy_htod_async(d_inputs[1], right, stream)

		bindings = [int(d_inputs[i]) for i in range(2)] + [int(d_output)]
		logging.warning(f"len(bindings): {len(bindings)}")

		for i in range(engine.num_io_tensors):
			# logging.warning(f"i: {i}")
			context.set_tensor_address(engine.get_tensor_name(i), bindings[i])

		context.execute_async_v3(stream_handle=stream.handle)
		stream.synchronize()
		cuda.memcpy_dtoh_async(h_output, d_output, stream)
		stream.synchronize()

		logging.warning(f"output shape: {h_output.shape}")
		logging.warning(np.squeeze(h_output[:, 0, :, :]))
		return h_output
		# for binding in engine:
			

# class TRTEngine:
# 	def __init__(self, engine_path):

			   
# 		# self.logger = trt.logger(trt.Logger.INFO)
# 		# self.logger = trt.Logger(trt.Logger.INFO)
# 		# with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
# 		# 	assert runtime
# 		# 	self.engine = runtime.deserialize_cuda_engine(f.read())
# 		# assert self.engine
# 		# self.context = self.engine.create_execution_context()
# 		# assert self.context
		
# 		# self.inputs = []
# 		# self.outputs = []
# 		# self.bindings = []
# 		# self.stream = cuda.Stream()
		
		
		
		
# 		# self.allocate_buffers()
# 		# assert len(self.inputs) > 0
# 		# assert len(self.outputs) > 0
# 		# assert len(self.bindings) > 0
# 		# logging.warning(f"type(self.engine): {type(self.engine)}")
# 		# logging.warning(f"(engine.num_layers): {(self.engine.num_layers)}")	

# 	def load_input(self, inputs):
# 		for idx, input in enumerate(inputs): 
# 			self.inputs[idx]['host'] = input
# 			# self.inputs[idx]['host'] = np.ravel(input)

# 	def pre_process_input(self,inputs, dims):
# 		pre_processed_inputs = [] 
# 		for (input, dim) in zip (inputs, dims):
# 			if dim is not None: 
# 				input = cv2.resize(input, (dim[1], dim[0]), interpolation=cv2.INTER_LINEAR)
# 				input = np.ascontiguousarray(input.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)		
# 			pre_processed_inputs.append(input)
# 		return pre_processed_inputs

# 	def allocate_buffers(self):
# 		self.inputs = []
# 		self.outputs = []
# 		self.bindings = []
# 		self.stream = cuda.Stream()
# 		with self.engine.create_execution_context() as context:
# 			for binding in self.engine:
# 				shape = self.engine.get_tensor_shape(binding)	
# 				size = trt.volume(shape)
# 				dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
# 				logging.warning(f"shape: {shape} size: {size} dtype: {dtype}")
# 				host_mem = cuda.pagelocked_empty(size, dtype)
# 				device_mem = cuda.mem_alloc(host_mem.nbytes)
# 				self.bindings.append(int(device_mem))
# 				if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
# 					self.inputs.append({'host': host_mem, 
# 									'device': device_mem, 
# 									'name': binding,
# 									'shape': shape,
# 									'dtype': dtype})
# 				else:
# 					self.outputs.append({'host': host_mem,
# 									'device': device_mem,
# 									'name': binding,
# 									'shape': shape,
# 									'dtype': dtype})
# 		for i in range(self.engine.num_io_tensors):
# 			self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

# 	def run_trt_inference(self):
# 		# transfer input data to the gpu
# 		for inp in self.inputs:
# 			cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
# 			self.context.execute_async_v3(self.stream.handle)
# 			# fetch outputs from gpu
# 			for out in self.outputs:
# 				cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
# 			# synchronize stream
# 			self.stream.synchronize()
# 			trt_inference_outputs = [out['host'] for out in self.outputs]
# 			# output = np.reshape(data, (1, 2, 640, 640))[0]
# 			return trt_inference_outputs

# 	def log_engine_io_details(self, engine_name):
# 		logging.warning(f"[INSPECTING {engine_name}.inputs/outputs] ==> ")
# 		for input in self.inputs:
# 			name = input['name']
# 			logging.warning(f"{name}.shape: {input['shape']} {name}.dtype: {input['dtype']}") 
# 		for output in self.outputs: 
# 			name = output['name']
# 			logging.warning(f"{name}.shape: {output['shape']} {name}.dtype: {output['dtype']}")
# 		logging.warning(f"\n")

		
		
		
if __name__ == '__main__':
	coloredlogs.install(level="INFO", force=True)  # install a handler on the root logger
	