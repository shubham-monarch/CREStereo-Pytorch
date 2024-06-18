#! /usr/bin/env python3

import tensorrt as trt	
import coloredlogs, logging

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
# MAX_BATCH_SIZE = 1


runtime = trt.Runtime(TRT_LOGGER)


def GiB(val):
    return val * 1 << 30

if __name__ == "__main__":

	coloredlogs.install(level="DEBUG", force=True)  # install a handler on the root logger
	onnx_file_path = 'models/m1.onnx'	
	logging.debug(f"TensortRT version: {trt.__version__}")

	builder = trt.Builder(TRT_LOGGER)
	network = builder.create_network(EXPLICIT_BATCH)
	# builder.max_batch_size = MAX_BATCH_SIZE
	
	config = builder.create_builder_config()
	config.max_workspace_size = GiB(1)
	config.set_flag(trt.BuilderFlag.FP16)
	# config.set_flag(trt.BuilderFlag.FP8)
	config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, GiB(1))
	
	parser = trt.OnnxParser(network, TRT_LOGGER)
	
	with open(onnx_file_path, "rb") as model:
		if not parser.parse(model.read()):
			logging.error("ERROR: Failed to parse the ONNX file.")
			for error in range(parser.num_errors):
				logging.error(parser.get_error(error))
				exit(1)
		else:
			logging.warning("ONNX file was successfully parsed")	
	
	logging.warning('Building the TensorRT engine.  This would take a while...')
	serialized_engine = builder.build_serialized_network(network, config)
	if serialized_engine is not None:
			engine = runtime.deserialize_cuda_engine(serialized_engine)
			engine_file_path = onnx_file_path.replace(".onnx", ".trt")
			logging.debug(f"engine_file_path: {engine_file_path}")
			with open(engine_file_path, "wb") as f:
				f.write(engine.serialize())
	logging.warning(f'TensorRT engine was successfully built and saved to disk at {engine_file_path}')
