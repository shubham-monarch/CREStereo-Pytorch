#! /bin/bash


TRTEXEC=/usr/src/tensorrt/bin/trtexec

$TRTEXEC --onnx=models/crestereo.onnx \
       	 --memPoolSize=workspace:1G \
	 --saveEngine=models/crestereo.trt \
	 --timingCacheFile=timing.cache \
	 --noTF32 \
	 #--stronglyTyped \
	 #--precisionConstraints=obey


