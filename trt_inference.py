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
# ssimport pycuda.driver as cuda


# TO-DO
# - using polygraphy for quick onnx model checking
# - refer official nvidia docs for best practices
# - explore onnx-graphsurgeon and polygraphy 
# - checck pip install trt-cu11
# - tinker with cudnn version 
# - change cuda version to 12.1 
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


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
runtime = trt.Runtime(TRT_LOGGER)
ZED_IMAGE_DIR = "zed_input/images"


class TRTEngine:
	def __init__(self, engine_path):
		# self.logger = trt.logger(trt.Logger.INFO)
		self.logger = trt.Logger(trt.Logger.INFO)
		trt.init_libnvinfer_plugins(self.logger, namespace="")
		with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
			assert runtime
			self.engine = runtime.deserialize_cuda_engine(f.read())
		assert self.engine
		self.context = self.engine.create_execution_context()
		assert self.context
		self.allocate_buffers()
		assert len(self.inputs) > 0
		assert len(self.outputs) > 0
		assert len(self.bindings) > 0
		
	def allocate_buffers(self):
		self.inputs = []
		self.outputs = []
		self.bindings = []
		self.stream = cuda.Stream()

		with self.engine.create_execution_context() as context:
			for binding in self.engine:
				# shape = self.engine.get_binding_shape(binding)	
				shape = self.engine.get_tensor_shape(binding)	
				size = trt.volume(shape)
				# dtype = trt.nptype(self.engine.get_binding_dtype(binding))
				dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
				# logging.debug(f"{binding}.shape: {shape} dtype: {dtype}")
				host_mem = cuda.pagelocked_empty(size, dtype)
				device_mem = cuda.mem_alloc(host_mem.nbytes)
				self.bindings.append(int(device_mem))
				if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
					self.inputs.append({'host': host_mem, 
						 				'device': device_mem, 
										'name': binding,
										'shape': shape,
										'dtype': dtype})
				else:
					self.outputs.append({'host': host_mem,
						   				'device': device_mem,
										'name': binding,
										'shape': shape,
										'dtype': dtype})
		
		for i in range(self.engine.num_io_tensors):
			self.context.set_tensor_address(self.engine.get_tensor_name(i), self.bindings[i])

	def run_trt_inference(self):
		# transfer input data to the gpu
		for inp in self.inputs:
			cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

			# run inference
			# self.context.execute_async_v2(
			# 	bindings=self.bindings,
			# 	stream_handle=self.stream.handle)
			self.context.execute_async_v3(self.stream.handle)

			# fetch outputs from gpu
			for out in self.outputs:
				cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

			# synchronize stream
			self.stream.synchronize()

			trt_inference_outputs = [out['host'] for out in self.outputs]
			# output = np.reshape(data, (1, 2, 640, 640))[0]
			return trt_inference_outputs

	def log_engine_io_details(self, engine_name):
		logging.warning(f"[INSPECTING {engine_name}.inputs/outputs] ==> ")
		for input in self.inputs:
			name = input['name']
			logging.warning(f"{name}.shape: {input['shape']} {name}.dtype: {input['dtype']}") 
			
		for output in self.outputs: 
			name = output['name']
			logging.warning(f"{name}.shape: {output['shape']} {name}.dtype: {output['dtype']}")
		logging.warning(f"\n")



def main():
	# managing cv2 window
	cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("TEST", (640, 480))
	
	# managing plt datasets
	plt_datasets = []

	# onnx model paths
	path_onnx_model = "models/crestereo.onnx"
	path_onnx_model_without_flow = "models/crestereo_without_flow.onnx"
	
	# trt engine paths
	path_trt_engine = path_onnx_model.replace(".onnx", ".trt")		
	path_trt_engine_without_flow = path_onnx_model_without_flow.replace(".onnx", ".trt")

	
	# IMG DIMS
	(H, W) = (480, 640)

	# READING IMAGES FROM DISK
	left_img = cv2.imread(f"{ZED_IMAGE_DIR}/left_18.png")
	right_img = cv2.imread(f"{ZED_IMAGE_DIR}/right_18.png")

	# < --------------- TRT_ENGINE_WITHOUT_FLOW ------------------- >
	# engine construction
	trt_engine_without_flow = TRTEngine(path_trt_engine_without_flow)
	trt_engine_without_flow.log_engine_io_details(engine_name="TRT_ENGINE_WITHOUT_FLOW")
	
	# pre-processing input
	imgL_dw2 = cv2.resize(left_img, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)
	imgR_dw2 = cv2.resize(right_img, (W //2, H //2),  interpolation=cv2.INTER_LINEAR)
	
	imgL_dw2 = np.ascontiguousarray(imgL_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32) 
	imgR_dw2 = np.ascontiguousarray(imgR_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	
	# loading input
	trt_engine_without_flow.inputs[0]['host'] = imgL_dw2
	trt_engine_without_flow.inputs[1]['host'] = imgR_dw2
	
	# inference
	trt_inference_outputs =  trt_engine_without_flow.run_trt_inference()
	logging.info(f"len(trt_inference_outputs): {len(trt_inference_outputs)}")
	logging.info(f"trt_inference_outputs[0].shape: {trt_inference_outputs[0].shape} trt_inference_outputs[0].dtype: {trt_inference_outputs[0].dtype}")

	# resizing outputs
	trt_output = trt_inference_outputs[0].reshape(1, 2, H // 2, W // 2)
	trt_output_squeezed = np.squeeze(trt_output[:, 0, :, :])
	trt_output_uint8 = utils.uint8_normalization(trt_output_squeezed)	
	cv2.imshow("TEST", trt_output_uint8)
	cv2.waitKey(0)	

	logging.warning(f"output_uint8.shape: {trt_output_uint8.shape} output_uint8.dtype: {trt_output_uint8.dtype}")

	# < --------------- TRT_ENGINE ------------------- >
	# engine construction
	trt_engine = TRTEngine(path_trt_engine)
	trt_engine.log_engine_io_details(engine_name="TRT_ENGINE")
	
	# pre-processing input
	imgL = cv2.resize(left_img, (H, W), interpolation=cv2.INTER_LINEAR)
	imgR = cv2.resize(right_img, (H, W), interpolation=cv2.INTER_LINEAR)
	flow_init = trt_output
	
	imgL = np.ascontiguousarray(imgL.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	imgR = np.ascontiguousarray(imgR.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	flow_init = np.ascontiguousarray(flow_init)

	# loading input	
	trt_engine.inputs[0]['host'] = imgL
	trt_engine.inputs[1]['host'] = imgR
	trt_engine.inputs[2]['host'] = flow_init

	# inference
	trt_inference_outputs =  trt_engine.run_trt_inference()
	logging.info(f"len(trt_inference_outputs): {len(trt_inference_outputs)}")
	logging.info(f"trt_inference_outputs[0].shape: {trt_inference_outputs[0].shape} trt_inference_outputs[0].dtype: {trt_inference_outputs[0].dtype}")

	# resizing outputs
	trt_output = trt_inference_outputs[0].reshape(1, 2, H, W)
	
	# adding disparity_float to plt_datasets
	plt_datasets.append([trt_output, 'disparity_map_float', 100, [0,10]])

	trt_output_squeezed = np.squeeze(trt_output[:, 0, :, :])
	trt_output_uint8 = utils.uint8_normalization(trt_output_squeezed)
	
	# adding disparity_uint8 to plt_datasets
	plt_datasets.append([trt_output_uint8, 'disparity_map_uint8', 255, [0,255]])	
	logging.warning(f"output_uint8.shape: {trt_output_uint8.shape} output_uint8.dtype: {trt_output_uint8.dtype}")


	cv2.imshow("TEST", trt_output_uint8)
	cv2.waitKey(0)

	utils.plot_histograms(plt_datasets)

if __name__ == '__main__':
	coloredlogs.install(level="INFO", force=True)  # install a handler on the root logger
	logging.debug(f"TensortRT version: {trt.__version__}")
	main()