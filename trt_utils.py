#! /usr/bin/env python3

import time
import pycuda.driver as cuda 
import tensorrt as trt
import os
import coloredlogs, logging

from tensorrt import TensorIOMode


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class FPS:
    def __init__(self):
        self.accum_time = 0
        self.curr_fps = 0
        self.fps = "FPS: ??"

    def start(self):
        self.prev_time = time.time()

    def stop(self):
        self.curr_time = time.time()
        exec_time = self.curr_time - self.prev_time
        self.prev_time = self.curr_time
        self.accum_time += exec_time

    def get_fps(self):
        self.curr_fps += 1
        if self.accum_time > 1:
            self.accum_time -= 1
            self.fps = "FPS: " + str(self.curr_fps)
            self.curr_fps = 0
        return self.fps

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    with engine.create_execution_context() as context:
        for binding in engine:
            # size = trt.volume(engine.get_tensor_shape(binding))
            # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # dtype = trt.nptype(engine.get_tensor_dtype(binding))
            # logging.debug(f"dtype: {dtype}")
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
            # if engine.get_tensor_mode(binding) == TensorIOMode.INPUT:
            #     inputs.append(HostDeviceMem(host_mem, device_mem)) 
            # else: 
            #     outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

    # self.inputs, self.outputs, self.bindings = [], [], []
    #     self.stream = cuda.Stream()
    #     for binding in engine:
    #         size = trt.volume(engine.get_binding_shape(binding))
    #         dtype = trt.nptype(engine.get_binding_dtype(binding))
    #         host_mem = cuda.pagelocked_empty(size, dtype)
    #         device_mem = cuda.mem_alloc(host_mem.nbytes)
    #         self.bindings.append(int(device_mem))
    #         if engine.binding_is_input(binding):
    #             self.inputs.append({'host': host_mem, 'device': device_mem})
    #         else:
    #             self.outputs.append({'host': host_mem, 'device': device_mem})