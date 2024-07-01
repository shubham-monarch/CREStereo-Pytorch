#! /bin/bash


TRTEXEC=/usr/src/tensorrt/bin/trtexec


MODEL_NO_FLOW="models/crestereo_without_flow"
ONNX_MODEL_NO_FLOW="${MODEL_NO_FLOW}.onnx"
TRT_MODEL_NO_FLOW="${MODEL_NO_FLOW}.trt"

MODEL="models/crestereo"
ONNX_MODEL="${MODEL}.onnx"
TRT_MODEL="${MODEL}.trt"

SIMP_MODEL_NO_FLOW="models/simp_crestereo_without_flow"
SIMP_ONNX_MODEL_NO_FLOW="${SIMP_MODEL_NO_FLOW}.onnx"
SIMP_TRT_MODEL_NO_FLOW="${SIMP_MODEL_NO_FLOW}.trt"

SIMP_MODEL="models/simp_crestereo"
SIMP_ONNX_MODEL="${SIMP_MODEL}.onnx"
SIMP_TRT_MODEL="${SIMP_MODEL}.trt"


$TRTEXEC \
--onnx=$SIMP_ONNX_MODEL_NO_FLOW \
--memPoolSize=workspace:1G \
--saveEngine=$SIMP_TRT_MODEL_NO_FLOW \
--timingCacheFile=timing.cache


$TRTEXEC \
--onnx=$SIMP_ONNX_MODEL \
--memPoolSize=workspace:1G \
--saveEngine=$SIMP_TRT_MODEL \
--timingCacheFile=timing.cache


# $TRTEXEC \
# --onnx=$ONNX_MODEL_NO_FLOW \
# --memPoolSize=workspace:1G \
# --saveEngine=$TRT_MODEL_NO_FLOW \
# --timingCacheFile=timing.cache


# $TRTEXEC \
# --onnx=$ONNX_MODEL \
# --memPoolSize=workspace:1G \
# --saveEngine=$TRT_MODEL \
# --timingCacheFile=timing.cache
    #--noTF32 \
    #--stronglyTyped \
    #--precisionConstraints=obey

