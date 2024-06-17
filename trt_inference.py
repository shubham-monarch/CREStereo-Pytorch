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
# - check tensort version compatibility
# - using polygraphy for quick onnx model checking
# - refer official nvidia docs for best practices
# - explore onnx-graphsurgeon and polygraphy 
# - checck pip install trt-cu11
# - tinker with cudnn version 
# - change cuda version to 12.1 
# - read onnxsim => removes unsupported operations -> both python / cli apis
# - how to specify precision
# - intergrate onnxsimp api to code
# - explore FPS() class -> YOLO -> https://github.com/sithu31296/PyTorch-ONNX-TRT/blob/master/examples/yolov4/yolov4/scripts/infer.py
# - package structure from YOLO 
# - add trt_info functions -> https://github.com/NVIDIA/TensorRT/tree/main/tools/experimental/trt-engine-explorer#workflow
# - add more attributes to the engine configuration -> optimization profile, f16, explore config.set_flag()
# - add pre / post processing
# - add trt engine class
# - make onxx_without_flow work without onnx-simplifier
# - addded batch_inference
# - refactor code based on official nvidia examples
# - to use ravel
# - upgrade HostDeviceMem class -> https://github.com/NVIDIA/TensorRT/blob/main/samples/python/common_runtime.py
# - add cuda loading in .ppth to .onnx conversion script
# - reload output to torch for further processing
# - fix negative values in flow_init
# - improve output normalization

# (H, W)
DIMS = (480, 640)
H,W = DIMS
ZED_IMAGE_DIR = "zed_input/images"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
runtime = trt.Runtime(TRT_LOGGER)


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

	def allocate_buffers(self):
		self.inputs = []
		self.outputs = []
		self.bindings = []
		self.stream = cuda.Stream()

		with self.engine.create_execution_context() as context:
			for binding in self.engine:
				shape = self.engine.get_binding_shape(binding)	
				size = trt.volume(shape)
				dtype = trt.nptype(self.engine.get_binding_dtype(binding))
				
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
			


	def run_trt_inference(self):
		for inp in self.inputs:
			cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

			# run inference
			self.context.execute_async_v2(
				bindings=self.bindings,
				stream_handle=self.stream.handle)

			# fetch outputs from gpu
			for out in self.outputs:
				cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

			# synchronize stream
			self.stream.synchronize()

			trt_inference_outputs = [out['host'] for out in self.outputs]
			# output = np.reshape(data, (1, 2, 640, 640))[0]
			return trt_inference_outputs

	def do_inference_v2(self, context, bindings, inputs, outputs, stream):
		# Transfer input data to the GPU.
		[cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
		# Run inference.
		context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
		# Transfer predictions back from the GPU.
		[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
		# Synchronize the stream
		stream.synchronize()
		# Return only the host outputs.
		return [out.host for out in outputs]




def main():
	
	cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("TEST", (W, H))

	# plt.figure(figsize=(10,10))
	plt_datasets = []

	logging.debug(f"TensortRT version: {trt.__version__}")
	
	# onnx_model = "models/crestereo_without_flow.onnx"
	# onnx_model = "models/crestereo_without_flow_simp.onnx"
	onnx_model = "models/crestereo.onnx"
	# trt_engine = "models/crestereo.trt"
	trt_engine = onnx_model.replace(".onnx", ".trt")	
	
	# engine = generate_engine_from_onnx(onnx_model)
	# trt.init_libnvinfer_plugins(None, "")
	# engine = load_engine("models/crestereo.trt")
	
	trt_engine = TRTEngine(trt_engine)
	trt_engine.allocate_buffers()

	logging.debug(f"[BEFORE LOADING INPUT DATA]")
	for input in trt_engine.inputs:
		name = input['name']
		logging.warning(f"{name}.shape: {input['shape']} {name}.dtype: {input['dtype']}") 
		
	for output in trt_engine.outputs: 
		name = output['name']
		logging.warning(f"{name}.shape: {output['shape']} {name}.dtype: {output['dtype']}")

	# PREPARING INPUT DATA
	left_img = cv2.imread(f"{ZED_IMAGE_DIR}/left_18.png")
	right_img = cv2.imread(f"{ZED_IMAGE_DIR}/right_18.png")

	(w ,h) = (W, H)
	# imgL_dw2 = cv2.resize(left_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
	# imgR_dw2 = cv2.resize(right_img, (w//2, h//2),  interpolation=cv2.INTER_LINEAR)
	# flow_init = np.random.random_sample((1, 2, h//2, w//2)).astype(np.float32)
	# flow_init = np.zeros((1, 2, h//2, w//2)).astype(np.float32)
	imgL = cv2.resize(left_img, (w, h), interpolation=cv2.INTER_LINEAR)
	imgR = cv2.resize(right_img, (w, h), interpolation=cv2.INTER_LINEAR)
	# imgL = cv2.resize(left_img, (h, w), interpolation=cv2.INTER_LINEAR)
	# imgR = cv2.resize(right_img, (h, w), interpolation=cv2.INTER_LINEAR)
	flow_init = np.random.randn(1, 2, h//2, w//2).astype(np.float32) * 255.0
	# flow_init = np.zeros((1, 2, h//2, w//2)).astype(np.float32)
	
	# imgL_dw2 = np.ascontiguousarray(imgL_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32) 
	# imgR_dw2 = np.ascontiguousarray(imgR_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	imgL = np.ascontiguousarray(imgL.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	imgR = np.ascontiguousarray(imgR.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	flow_init = np.ascontiguousarray(flow_init)

	logging.debug("[INPUT DATA FINAL SHAPES] => ")
	logging.warning(f"imgL.shape: {imgL.shape}")
	logging.warning(f"imgR.shape: {imgR.shape}")
	logging.warning(f"flow_init.shape: {flow_init.shape}")

	# VISUALIZING INPUT DATA
	left_image_cv = np.squeeze(imgL[:, :, :, :])
	left_image_cv = utils.uint8_normalization(left_image_cv).transpose(1, 2, 0)
	
	right_image_cv = np.squeeze(imgR[:, :, :, :])
	right_image_cv = utils.uint8_normalization(right_image_cv).transpose(1, 2, 0)
	
	logging.warning(f"left_image_cv.shape: {left_image_cv.shape} left_image_cv.dtype: {left_image_cv.dtype}")
	logging.warning(f"right_image_cv.shape: {right_image_cv.shape} right_image_cv.dtype: {right_image_cv.dtype}")

	
	cv2.imshow("TEST", cv2.hconcat([left_image_cv, right_image_cv]))
	cv2.waitKey(0)
	
	

	# LOADING PREPARED INPUT DATA TO TRT ENGINE 
	# trt_engine.inputs[0]['host'] = imgL_dw2
	# trt_engine.inputs[1]['host'] = imgR_dw2
	trt_engine.inputs[0]['host'] = imgL
	trt_engine.inputs[1]['host'] = imgR
	trt_engine.inputs[2]['host'] = flow_init

	# trt_engine.inputs[0]['host'] = np.ravel(imgL_dw2)
	# trt_engine.inputs[1]['host'] = np.ravel(imgR_dw2)
	# trt_engine.inputs[2]['host'] = np.ravel(flow_init)
	
	
	# RUNNING INFERENCE
	trt_inference_outputs =  trt_engine.run_trt_inference()
	logging.debug(f"len(trt_inference_outputs): {len(trt_inference_outputs)}")
	logging.debug(f"trt_inference_outputs[0].shape: {trt_inference_outputs[0].shape} trt_inference_outputs[0].dtype: {trt_inference_outputs[0].dtype}")

	# RESIZE TRT_INFERENCE_OUTPUT
	output = trt_inference_outputs[0].reshape(1, 2, H, W)
	output = np.squeeze(output[:, 0, :, :])
	
	# flattened_data_float32 = output.flatten()
	# logging.warning(f"flattened_data_float32.min: {output.flatten().min()} flattened_data_float32.max: {output.flatten().max()}")
	# plt.subplot(1,2,1)
	logging.warning(f"[BEFORE OUTLIER FILTERING]output.shape: {output.shape} output.dtype: {output.dtype}")
	output_outlier_filtered = utils.reject_outliers_2(output)
	logging.warning(f"[AFTER OUTLIER FILTERING] output_outlier_filtered.shape: {output_outlier_filtered.shape} output_outlier_filtered.dtype: {output_outlier_filtered.dtype}")	

	# utils.plot_histogram(output_outlier_filtered, 'output_outlier_filtered', bins=20, range=[0,1])
	plt_datasets.append([output_outlier_filtered, 'output_outlier_filtered', 20, [0.,1.]])
	output_uint8 = utils.uint8_normalization(output_outlier_filtered)
	output_uint8 = np.where(output_uint8 != 0, output_uint8, np.nan)
	logging.warning(f"output_uint8.shape: {output_uint8.shape} output_uint8.dtype: {output_uint8.dtype}")

	plt_datasets.append([output_uint8, 'output_uint8', 255, [0,255]])
	
	utils.plot_histograms(plt_datasets)

	cv2.imshow("TEST", output_uint8)
	cv2.waitKey(0)

	plt.close()
	
if __name__ == '__main__':
	coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger
	main()