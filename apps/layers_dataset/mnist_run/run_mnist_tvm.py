import tvm
from tvm.ir import module
import tvm.relay as relay
from tvm.contrib import utils, graph_executor
import numpy as np
import timeit as time 
input_tensor = np.ones(shape=(1,28,28,1))
output_tensor = np.zeros(shape=(1,1))
shape_dict = {"input_1": (1,28,28,1)}
dtype_dict = {"input_1": "float32"}
def test_runtime(path):
    lib = tvm.runtime.load_module(path)
    grt_mod = graph_executor.GraphModule(lib["default"](tvm.cpu()))
    #print(grt_mod._get_input())
    grt_mod.get_input(0)
    grt_mod.set_input(key="input_1",value=input_tensor)
    start=time.default_timer()
    grt_mod.run()
    end=time.default_timer()
    print("时间:{}".format(end-start))
    output_count = grt_mod.get_num_outputs()
    out = [grt_mod.get_output(i).asnumpy() for i in range(output_count)]
    print(out)

model="/media/s3lab-1/570c75c9-cb6f-4ce6-bcc5-9f636c7310f3/s3lab-1/workspace/Deeplearning_Framework/TVM/tvm2/test/mnist_run/mnist_llvm_x86_.so"
opt_model=lib_path="/media/s3lab-1/570c75c9-cb6f-4ce6-bcc5-9f636c7310f3/s3lab-1/workspace/Deeplearning_Framework/TVM/tvm2/test/mnist_run/mnist_llvm_x86_tune.so"
test_runtime(model)
test_runtime(opt_model)