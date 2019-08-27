
import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
import logging
logging.basicConfig(level=logging.INFO)
from onnx import checker
import onnx

syms   = './outputs/pfld_7.5w/lmks_detector-symbol.json'
params = './outputs/pfld_7.5w/lmks_detector-0400.params'

input_shape = (1,3,96,96)

onnx_file = './outputs/pfld_7.5w/pfld-lite.onnx'

# Invoke export model API. It returns path of the converted onnx model
converted_model_path = onnx_mxnet.export_model(syms, params, [input_shape], np.float32, onnx_file)

# Load onnx model
model_proto = onnx.load_model(converted_model_path)

# Check if converted ONNX protobuf is valid
checker.check_graph(model_proto.graph)
