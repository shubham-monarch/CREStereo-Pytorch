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



TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
# TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
runtime = trt.Runtime(TRT_LOGGER)
ZED_IMAGE_DIR = "zed_input/images"
BASELINE = 0.13
FOCAL_LENGTH = 1093.5

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
		logging.warning(f"type(self.engine): {type(self.engine)}")
		logging.warning(f"(engine.num_layers): {(self.engine.num_layers)}")	
		
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

def main(num_frames):
	# IMG DIMS
	(H, W) = (480, 640)

	# managing cv2 window
	cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
	cv2.resizeWindow("TEST", (2 * W, H))
		
	image_files_left = [os.path.join(ZED_IMAGE_DIR, f) for f in os.listdir(ZED_IMAGE_DIR) if f.startswith('left_') and f.endswith('.png')]
	image_files_right = [os.path.join(ZED_IMAGE_DIR, f) for f in os.listdir(ZED_IMAGE_DIR) if f.startswith('right_') and f.endswith('.png')]
	
	image_files_left.sort()
	image_files_right.sort()

	assert(len(image_files_left) == len(image_files_right)), "Number of left and right images should be equal"
	assert(len(image_files_left) > num_frames), "Number of frames should be less than total number of images"
	
	# generating random frame indices
	frame_indices = random.sample(range(0, len(image_files_left) - 1), num_frames)
	logging.info(f"frame_indices: {frame_indices}")
	fps = trt_utils.FPS()

	# onnx model paths
	path_onnx_model = "models/crestereo.onnx"
	path_onnx_model_without_flow = "models/crestereo_without_flow.onnx"
	
	# trt engine paths
	path_trt_engine = path_onnx_model.replace(".onnx", ".trt")		
	path_trt_engine_without_flow = path_onnx_model_without_flow.replace(".onnx", ".trt")

	# trt-engine-without-flow construction
	trt_engine_without_flow = TRTEngine(path_trt_engine_without_flow)
	trt_engine_without_flow.log_engine_io_details(engine_name="TRT_ENGINE_WITHOUT_FLOW")
	
	# trt-engine construction
	trt_engine = TRTEngine(path_trt_engine)
	trt_engine.log_engine_io_details(engine_name="TRT_ENGINE")
	
	for i in (frame_indices):
		
		plts = []

		# READING IMAGES FROM DISK
		# fps.start()
		start_time = time.time()
		rand_idx = random.randint(0, num_frames - 1)
		logging.info(f"rand_idx: {rand_idx}")
		left_img = cv2.imread(image_files_left[rand_idx])
		right_img = cv2.imread(image_files_right[rand_idx])

		# < --------------- TRT_ENGINE_WITHOUT_FLOW ------------------- >	
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
		# logging.info(f"len(trt_inference_outputs): {len(trt_inference_outputs)}")
		# logging.info(f"trt_inference_outputs[0].shape: {trt_inference_outputs[0].shape} trt_inference_outputs[0].dtype: {trt_inference_outputs[0].dtype}")

		# resizing outputs
		trt_output = trt_inference_outputs[0].reshape(1, 2, H // 2, W // 2)
		logging.warning(f"trt_output.shape: {trt_output.shape}")
		# disparity_data = np.squeeze(trt_output[:, 0, :, :])
		
		# < --------------- TRT_ENGINE ------------------- >	
		# pre-processing input
		imgL = cv2.resize(left_img, (W, H), interpolation=cv2.INTER_LINEAR)
		imgR = cv2.resize(right_img, (W, H), interpolation=cv2.INTER_LINEAR)
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
		
		# resizing outputs
		trt_output = trt_inference_outputs[0].reshape(1, 2, H, W)
		trt_output = np.squeeze(trt_output[:, 0, :, :]) # (H * W)
		# logging.warn(f"trt_output.shape: {trt_output.shape} trt_output.dtype: {trt_output.dtype}")
		
		# fps.stop()
		end_time = time.time()
		frame_rate = 1 / (end_time - start_time)
		logging.error(f"Frame Rate: {frame_rate}")

		# DISPARITY NORMALIZATION
		# <---------------------------------------------------------------------------------->
		# approach 1
		disp_data = trt_output
		disp_data_uint8 = utils.uint8_normalization(disp_data)
		# logging.info(f"disp_data.shape: {disp_data.shape} disp_data_uint8.shape: {disp_data_uint8.shape}")
		# plts.append({"data": disp_data, "disparity": 10, "bins": 100})
		# plts.append((disp_data_uint8, 'uint8', 10))s
		logging.warning("DISP_DATA=>")
		logging.info(f"mn: {disp_data.min()} mx: {disp_data.max()}")
		logging.info(f"\n50-percentile: {np.percentile(disp_data, 50)} \
						\n90-percentile: {np.percentile(disp_data, 90)} \
						\n99-precentile: {np.percentile(disp_data, 99)}")
		# logging.warning(f"[disp_data] mx: {disp_data.max()} mn: {disp_data.min()} [disp_data_uint8] mx: {disp_data_uint8.max()} mn: {disp_data_uint8.min()}")
		plts.append(utils.PLT(data=disp_data, 
						title='disparity',
						bins=100, 
						range=(disp_data.min(), disp_data.max())))
		


		plts.append(utils.PLT(data=disp_data_uint8, title='uint8', bins=100, range=(0,255)))
		
		percentile_99_disp = np.percentile(disp_data, 99) 
		clipped_disp_data = np.clip(disp_data, 0.0,  percentile_99_disp)
		clipped_disp_data_uint8 = utils.uint8_normalization(clipped_disp_data)

		plts.append(utils.PLT(data=clipped_disp_data,
						title='clipped disparity',
						bins=100,
						range=(clipped_disp_data.min(), clipped_disp_data.max())))
		
		plts.append(utils.PLT(data=clipped_disp_data_uint8, 
						title='clipped uint8',
						bins=100,
						range=(0, 255)))
		
		imgL_ = np.squeeze(imgL[:, 0, :, :]).astype(np.uint8)
	
		# .transpose(1, 2, 0).astype(np.uint8)
		logging.info(f"imgL_.shape: {imgL_.shape} disp_data_uint8.shape: {disp_data_uint8.shape}")
		# logging.info(f"left_img.shape: {left_img.shape} clipped_disp_data_uint8.shape: {clipped_disp_data_uint8.shape}")
		cv2.imshow("TEST", cv2.hconcat([imgL_, clipped_disp_data_uint8]))
		cv2.waitKey(0)
		# cv2.imshow()
		# # <---------------------------------------------------------------------------------->
		# # approach 1.1
		# disp_data = trt_output
		# disp_data_uint8 = utils.uint8_normalization(disp_data)
		# # logging.info(f"disp_data.shape: {disp_data.shape} disp_data_uint8.shape: {disp_data_uint8.shape}")
		# plts.append((disp_data, 'disparity'))
		# plts.append((disp_data_uint8, 'uint8'))
		# # <---------------------------------------------------------------------------------->
		# # approach two
		# disp_data_norm_log = utils.normalization_log(disp_data)
		# disp_norm_log_uint8 = utils.uint8_normalization(disp_data_norm_log)
		# # logging.warning(f"disp_data_norm_log.shape: {disp_data_norm_log.shape} disp_norm_log_uint8.shape: {disp_norm_log_uint8.shape}")
		# plts.append((disp_data_norm_log, 'disparity [normalized log]'))
		# plts.append((disp_norm_log_uint8, 'uint8'))
		# # <---------------------------------------------------------------------------------->
		# # approach three
		# disp_norm_percentile = utils.normalization_percentile(disp_data, 2, 98)
		# disp_norm_percentile_uint8 = utils.uint8_normalization(disp_norm_percentile)	
		# # logging.warning(f"disp_norm_percentile.shape: {disp_norm_percentile.shape} disp_norm_percentile_uint8.shape: {disp_norm_percentile_uint8.shape}")
		# plts.append((disp_norm_percentile, 'disparity  [normalized percentile]'))
		# plts.append((disp_norm_percentile_uint8, 'uint8'))
		# # <---------------------------------------------------------------------------------->
		# imgL = np.squeeze(imgL[:, :, :, :]).transpose(1, 2, 0).astype(np.uint8)	
		# imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

		# # logging.info(f"imgL.shape: {imgL.shape} disp_data_uint8.shape: {disp_data_uint8.shape}")
		
		# hconcat_uint8 = cv2.hconcat([ disp_data_uint8, disp_norm_log_uint8, disp_norm_percentile_uint8])
		# hconcat_imgL = cv2.hconcat([imgL, imgL, imgL])
		
		# cv2.imshow("TEST", cv2.vconcat([hconcat_imgL, hconcat_uint8]))
		# # cv2.imwrite(f"frame_{i}.png", cv2.vconcat([hconcat_imgL, hconcat_uint8]))
		
		# cv2.waitKey(0)
		# utils.plot_histograms(plts)
		
		
		
if __name__ == '__main__':
	coloredlogs.install(level="INFO", force=True)  # install a handler on the root logger
	logging.debug(f"TensortRT version: {trt.__version__}")
	num_images =20
	main(num_images)