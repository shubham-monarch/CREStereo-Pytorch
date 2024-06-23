#! /bin/bash


TRTEXEC=/usr/src/tensorrt/bin/trtexec


MODEL="models/crestereo_without_flow"
ONNX_MODEL="${MODEL}.onnx"
TRT_MODEL="${MODEL}.trt"

echo "ONNX_MODEL: $ONNX_MODEL"
echo "TRT_MODEL: $TRT_MODEL"

$TRTEXEC \
--onnx=$ONNX_MODEL \
--memPoolSize=workspace:1G \
--saveEngine=$TRT_MODEL \
--timingCacheFile=timing.cache
    #--noTF32 \
    #--stronglyTyped \
    #--precisionConstraints=obey

if [ $? -eq 0 ]; then
    # Print success message in green
    tput setaf 2; echo "ONXX ==> TFT Conversion successful"; tput sgr0
else
    # Print error message in red
    tput setaf 1; echo "ONNX ==> TFT Conversion failed"; tput sgr0
fi