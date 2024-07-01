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

# camera constants
BASELINE = 0.13
FOCAL_LENGTH = 1093.5

# io constants
ZED_IMAGE_DIR = zed_inference.ZED_IMG_DIR
TRT_INFERENCE_DIR = "trt_inference"
TRT_INIT_FLOW_DIR = f"{TRT_INFERENCE_DIR}/trt_init_flow"

FOLDERS_TO_CREATE = [TRT_INFERENCE_DIR]


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

	def load_input(self, inputs):
		for idx, input in enumerate(inputs): 
			self.inputs[idx]['host'] = input
			# self.inputs[idx]['host'] = np.ravel(input)

	def pre_process_input(self,inputs, dims):
		pre_processed_inputs = [] 
		for (input, dim) in zip (inputs, dims):
			if dim is not None: 
				input = cv2.resize(input, (dim[1], dim[0]), interpolation=cv2.INTER_LINEAR)
				input = np.ascontiguousarray(input.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)		
			pre_processed_inputs.append(input)
		return pre_processed_inputs

	def allocate_buffers(self):
		self.inputs = []
		self.outputs = []
		self.bindings = []
		self.stream = cuda.Stream()
		with self.engine.create_execution_context() as context:
			for binding in self.engine:
				shape = self.engine.get_tensor_shape(binding)	
				size = trt.volume(shape)
				dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
				logging.warning(f"shape: {shape} size: {size} dtype: {dtype}")
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

	utils.delete_folders(FOLDERS_TO_CREATE)
	utils.create_folders(FOLDERS_TO_CREATE)
	
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
	# path_onnx_model = "models/crestereo.onnx"
	# path_onnx_model_without_flow = "models/crestereo_without_flow.onnx"
	
	path_onnx_model = "models/simp_crestereo.onnx"
	path_onnx_model_without_flow = "models/simp_crestereo_without_flow.onnx"
	

	# trt engine paths
	path_trt_engine = path_onnx_model.replace(".onnx", ".trt")		
	path_trt_engine_without_flow = path_onnx_model_without_flow.replace(".onnx", ".trt")

	# trt-engine-without-flow construction
	trt_engine_without_flow = TRTEngine(path_trt_engine_without_flow)
	trt_engine_without_flow.log_engine_io_details(engine_name="TRT_ENGINE_WITHOUT_FLOW")
	
	# trt-engine construction
	trt_engine = TRTEngine(path_trt_engine)
	trt_engine.log_engine_io_details(engine_name="TRT_ENGINE")
	
	for i in tqdm(frame_indices):
		
		plts = []

		# START TIMER
		start_time = time.time()
		
		# read left and right images
		rand_idx = random.randint(0, num_frames - 1)
		left_img = cv2.imread(image_files_left[rand_idx])
		right_img = cv2.imread(image_files_right[rand_idx])

		# ENGINE ONE [without flow]	
		ppi_model1 = trt_engine_without_flow.pre_process_input([left_img, right_img], 
														 [(H // 2, W // 2), (H // 2, W // 2)])
		# loading inputs to engine
		trt_engine_without_flow.load_input(ppi_model1)
		# inference
		trt_inference_outputs =  trt_engine_without_flow.run_trt_inference()
		# resizing outputs
		trt_output = trt_inference_outputs[0].reshape(1, 2, H // 2, W // 2)
		logging.error(f"trt_output.shape: {trt_output.shape}")	
		init_flow = np.squeeze(trt_output[:, 0, :, :]) 
		logging.error(f"init_flow.shape: {init_flow.shape}")	
		utils_matplotlib.plot_arrays(init_flow)


		
		
		
if __name__ == '__main__':
	coloredlogs.install(level="INFO", force=True)  # install a handler on the root logger
	logging.debug(f"TensortRT version: {trt.__version__}")
	num_images =2
	main(num_images)