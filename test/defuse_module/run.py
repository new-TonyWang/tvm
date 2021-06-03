import os

import tvm
import tvm.relay as relay
from tvm.contrib import graph_runtime
from tvm.driver.tvmc.runner import get_input_info

ctx = tvm.cpu()
loaded_lib = tvm.runtime.load_module('/media/s3lab-1/570c75c9-cb6f-4ce6-bcc5-9f636c7310f3/s3lab-1/workspace/Deeplearning_Framework/TVM/tvm/test/defuse_module/cpu.so')
module = graph_runtime.GraphModule(loaded_lib["default"](ctx))
with open('/media/s3lab-1/570c75c9-cb6f-4ce6-bcc5-9f636c7310f3/s3lab-1/workspace/python/tvm/defuse_module/deploy_param.params','b')as param:
    params = bytearray(param.read())
module.load_params(params)
graph = open('/media/s3lab-1/570c75c9-cb6f-4ce6-bcc5-9f636c7310f3/s3lab-1/workspace/python/tvm/defuse_module/deploy_graph.json').read()
shape_dict, dtype_dict = get_input_info(graph, params)

