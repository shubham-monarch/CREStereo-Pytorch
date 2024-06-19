#! /usr/bin/env python3

import tensorrt as trt	
import coloredlogs, logging

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
# MAX_BATCH_SIZE = 1


runtime = trt.Runtime(TRT_LOGGER)


def GiB(val):
    return val * 1 << 30

if __name__ == "__main__":

	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	# sonnx_file_path = 'models/m1.onnx'	
	onnx_file_path = 'models/crestereo_without_flow.onnx'	
	# onnx_file_path = 'models/crestereo.onnx'	
	logging.debug(f"TensortRT version: {trt.__version__}")

	builder = trt.Builder(TRT_LOGGER)
	network = builder.create_network(EXPLICIT_BATCH)
	# builder.max_batch_size = MAX_BATCH_SIZE
	
	logging.debug(f"network.num_layers: {network.num_layers}")

	config = builder.create_builder_config()
	# config.max_workspace_size = GiB(1) => deprecated
	
	# if builder.platform_has_fast_fp16:
	# 	logging.warning("build.platform_has_fp16 is True")
	# 	config.set_flag(trt.BuilderFlag.FP16)
	# else:
	# 	logging.warning("build.platform_has_fp16 is False")
	
	if builder.platform_has_tf32:
		logging.warning("build.platform_has_tf32 is True")
		config.set_flag(trt.BuilderFlag.TF32)	
	else:
		logging.warning("build.platform_has_tf32 is False")
		
	
	# config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
	config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(1))
	
	parser = trt.OnnxParser(network, TRT_LOGGER)
	
	with open(onnx_file_path, "rb") as model:
		if not parser.parse(model.read()):
			logging.error("ERROR: Failed to parse the ONNX file.")
			for error in range(parser.num_errors):
				logging.error(parser.get_error(error))
				exit(1)
		else:
			logging.warning("ONNX file was successfully parsed!")	
	
	logging.warning('Building the TensorRT engine.  This would take a while...')
	serialized_engine = builder.build_serialized_network(network, config)
	if serialized_engine is not None:
			# engine = runtime.deserialize_cuda_engine(serialized_engine)
			engine_file_path = onnx_file_path.replace(".onnx", ".trt")
			logging.debug(f"engine_file_path: {engine_file_path}")
			with open(engine_file_path, "wb") as f:
				f.write(engine.serialize())
	logging.warning(f'TensorRT engine was successfully built and saved to disk at {engine_file_path}')
