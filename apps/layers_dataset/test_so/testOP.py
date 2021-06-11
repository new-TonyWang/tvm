import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import tvm
from tvm import relay,auto_scheduler

ctx = tvm.cpu(0)
target = tvm.target.Target("llvm -mcpu=skylake")

def auto_schedule_run_tuning(mod,params):
    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)
    print("auto_schedule Begin tuning...")

    log_file = "./log_auto_schedule-%s.json" % ( target.kind.name)
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=200,  # change this to 20000 to achieve the best performance
    runner=auto_scheduler.LocalRunner(repeat=20, enable_cpu_cache_flush=True),
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)
    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3,config={"relay.backend.use_auto_scheduler": True}):
            mod=relay.build(ir_mod=mod,target=target,params=params)

            mod.export_library("./auto_schedule-conv1d")



path="Conv1D-__filters___4,__kernel_size____1_,__strides___1,__padding____valid_,__groups___4,__activation____linear_,__use_bias___True,__input_shape____10,_500_,__batch_size___1_"
mod = keras.models.load_model(path,compile=True)
input = tf.convert_to_tensor(np.ones([2,10,1000]))
output=mod.predict(input)
print(output)
print(output.shape)
{'filters': 4, 'kernel_size': [1], 'strides': 1, 'padding': 'valid', 'groups': 4, 'activation': 'linear', 'use_bias': True, 'input_shape': [1, 10, 500], 'batch_size': 1}
shape_dict={"input":param['input_shape']}
mod=keras.models.load_model(path,compile=True)
mod,params=relay.frontend.from_keras(mod,layout="NWC",shape=shape_dict)
auto_schedule_run_tuning(mod,params)

