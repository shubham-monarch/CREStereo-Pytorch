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
import utils
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
# - add engine serialization and deserialization 
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
				
				logging.debug(f"{binding}.shape: {shape} dtype: {dtype}")
				
				host_mem = cuda.pagelocked_empty(size, dtype)
				device_mem = cuda.mem_alloc(host_mem.nbytes)
				
				# Append the device buffer to device bindings.
				self.bindings.append(int(device_mem))
				
				# Append to the appropriate list.
				if self.engine.binding_is_input(binding):
					self.inputs.append(trt_utils.HostDeviceMem(host_mem, device_mem))
				else:
					self.outputs.append(trt_utils.HostDeviceMem(host_mem, device_mem))	

	
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
	# engine = build_engine("models/crestereo_without_flow.onnx")
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

	# # do_inference_v2(context, inputs, outputs, bindings, stream, batch_size=1)
	# do_inference_v2(context, inputs, outputs, bindings, stream)




	# load the inputs
	# Assuming left_img and right_img are your input images
	left_img = cv2.imread(f"{ZED_IMAGE_DIR}/left_18.png")
	right_img = cv2.imread(f"{ZED_IMAGE_DIR}/right_18.png")

	(w ,h) = (W, H)
	imgL_dw2 = cv2.resize(left_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
	imgR_dw2 = cv2.resize(right_img, (w//2, h//2),  interpolation=cv2.INTER_LINEAR)
	# flow_init = np.random.random_sample((1, 2, h//2, w//2)).astype(np.float32)
	flow_init = np.zeros((1, 2, h//2, w//2)).astype(np.float32)
	
	imgL_dw2 = np.ascontiguousarray(imgL_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32) 
	imgR_dw2 = np.ascontiguousarray(imgR_dw2.transpose(2, 0, 1)[None, :, :, :]).astype(np.float32)
	flow_init = np.ascontiguousarray(flow_init)


	
	# imgL_dw2 /= 255.0 
	# imgR_dw2 /= 255.0
	
	# trt_engine.inputs[0].host = imgL_dw2
	# trt_engine.inputs[1].host = imgR_dw2
	# trt_engine.inputs[2].host = flow_init

	trt_engine.inputs[0].host = np.ravel(imgL_dw2)
	trt_engine.inputs[1].host = np.ravel(imgR_dw2)
	trt_engine.inputs[2].host = np.ravel(flow_init)

	outputs = trt_engine.do_inference_v2(trt_engine.context,
										trt_engine.bindings,
										trt_engine.inputs, 
										trt_engine.outputs,
										trt_engine.stream)

	logging.debug(f"type(outputs): {type(outputs)} len(outputs): {len(outputs)}")

	# trt_outputs = trt_engine.do_inference_v2(context, bindings, inputs, outputs, stream)[0]
	
	# logging.debug(f"len(trt_outputs): {len(trt_outputs)}")
	# logging.debug(f"trt_outputs[0].shape: {trt_outputs.shape}")
	# logging.debug(f"type(trt_outputs): {type(trt_outputs)}")
	# # logging.debug(f"trt_outputs: {trt_outputs.shape}")

	# # output = trt_outputs.reshape((1, 2, h, w))
	# output = trt_outputs.reshape((1, w, h, 2))
	# logging.debug(f"output.shape: {output.shape}")
	# # output_ = np.squeeze(output[:, 0, :, :])
	# output_ = np.squeeze(output[:, :, :, 0])

	# output_uint8 = utils.uint8_normalization(output_)
	# cv2.imshow("output", output_uint8)
	# cv2.waitKey(0)

	# output_ = np.squeeze(trt_outputs[:, 0, :, :])
	# logging.debug(f"output_.shape: {output_.shape}")
	# logging.debug(f"output.dtype: {output_.dtype}")
	# # # inputs[0].host = np.random.random_sample(inputs[0].host.shape).astype(np.float32)
	# # # inputs[1].host = np.random.random_sample(inputs[1].host.shape).astype(np.float32)
	
	# # output = do_inference_v2(context, bindings, inputs, outputs, stream)



if __name__ == '__main__':
	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	main()