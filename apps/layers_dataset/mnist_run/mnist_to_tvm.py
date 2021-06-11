import tvm
from tvm import target
import tvm.relay as relay
import tvm.relay.analysis.call_graph as cg
import tvm.relay.analysis.analysis as an
import tvm._ffi.registry as registry
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os
from tvm import relay, auto_scheduler
"""
编译mnist模型
"""
def  test_mnist():
    path = "./mnist_02.3.h5"
    print(tf.__version__)
    ctx = tvm.cpu(0)
    #target=tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu")
    target_host = 'llvm'
    target = tvm.target.Target("llvm -mcpu=skylake")
    layout = 'NHWC'
    mod = keras.models.load_model(path,compile=True)
    input_tensor = np.ones(shape=(1,28,28,1))
    output_tensor = np.zeros(shape=(1,1))
    shape_dict = {"conv2d_input": (1,28,28,1)}
    dtype_dict = {"input": "float32"}
    mod, params = relay.frontend.from_keras(mod,layout = layout,shape=shape_dict)
    print("Tensorflow keras imported to relay frontend.")
    log_file = "%s-%s-%s.json" % ("./mnist_02.h5", layout, target.kind.name)
    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)

    def run_tuning():
        print("Begin tuning...")
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )

        tuner.tune(tune_option)
    run_tuning()
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3,config={"relay.backend.use_auto_scheduler": True}):
        #with tvm.transform.PassContext(opt_level=3,config={"tir.disable_vectorize": True}):
            #graph, lib, params= relay.build(mod, target=arget, target_host=target_host, params=params)#return IRModule
            #opt_model, opt_params = relay.optimize(mod, target_host, params)#return IRModule
            #callgraph = cg.CallGraph(opt_model)
            #dfscall = an.post_order_visit(opt_model)
            #print(dfscall)
            #print(opt_model.astext(show_meta_data=True))
            print(os.getpid())
            #lib = relay.build(mod,target="cpu --mcpu=core-avx2 ",target_host="llvm  --runtime=aot --link-params ", params=params)
            lib = relay.build(mod,target="llvm ",target_host="llvm ", params=params)
        
            #lib = relay.build(mod,target="c --executor=aot   --link-params ", params=params)
            #lib.lib.save('lib_aot.c')
        # print(lib.lib.get_source("asm"))
            #lib = relay.build_module.create_executor(kind="graph",mod = mod, target=target_host)
            #func = relay.Function(relay.analysis.free_vars(opt_model), opt_model)
    # m = graph_runtime.GraphModule(lib["default"](ctx))
    # a = tvm.nd.array(input_tensor, ctx)
    # b = tvm.nd.array(output_tensor,ctx)
    #lib(a,b)
    #m = graph_runtime.GraphModule(lib["default"](tvm.cpu(0)))
    path_lib = "mnist_02.3.so"

    # from tvm.contrib import ndk
    # lib.export_library(path_lib,ndk.create_shared)
    lib.export_library(path_lib)

    # with open("./cbrc.c",'w') as file:
    #     file.write(lib.get_source())

test_mnist()