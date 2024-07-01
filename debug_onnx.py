#! /usr/bin/env python3
import coloredlogs, logging
import onnx 

if __name__ == "__main__": 
    coloredlogs.install(level="WARN", force=True)  # install a handler on the root logger

    model = onnx.load("models/crestereo.onnx")
    simp_model = onnx.load("models/simp_crestereo.onnx") 

    logging.warning(f"model.num_layers: {len(model.graph.node)}")
    logging.warning(f"simp_model.num_layers: {len(simp_model.graph.node)}")

    model_no_flow = onnx.load("models/crestereo_without_flow.onnx")
    simp_model_no_flow = onnx.load("models/simp_crestereo_without_flow.onnx")

    logging.warning(f"model_no_flow.num_layers: {len(model_no_flow.graph.node)}")
    logging.warning(f"simp_model_no_flow.num_layers: {len(simp_model_no_flow.graph.node)}")