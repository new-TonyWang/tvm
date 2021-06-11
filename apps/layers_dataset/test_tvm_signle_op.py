from tensorflow.keras import layers
from tensorflow.python.ops.control_flow_ops import with_dependencies
import tvm
import tvm.relay as relay
from tvm import relay, auto_scheduler,autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tensorflow.keras as keras
import numpy as np
import os
import sys
"""
使用该文件测试编译单个算子排查错误
"""
layer_path="/test_so"
log_path="/test_so/logs"
threads=2
ctx = tvm.cpu(0)
target = tvm.target.Target("llvm -mcpu=skylake-avx512")
parentdir=os.path.join(os.getcwd(),"test_so")
logdir=os.path.join(os.getcwd(),"test_so/logs")
# filelist=os.listdir(os.path.join(os.getcwd(),"Activation_layers"))
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 


def compile_directly(mod,params,parentdir,layer_name):
    with tvm.transform.PassContext(opt_level=3):
        mod=relay.build(ir_mod=mod,target=target,params=params)
    mod.export_library(os.path.join(parentdir,layer_name))
    return layer_name

def auto_schedule_run_tuning(mod,params,parentdir,layer_name):
    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    if(len(tasks)==0):#没有task
        compile_directly(mod,params,parentdir,layer_name)
        return None,False
    for idx, task in enumerate(tasks):
        print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
        print(task.compute_dag)
    print("auto_schedule Begin tuning...")
    log_file = "%s/logs/auto_schedule-%s-%s.json" % (parentdir,layer_name, target.kind.name)
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

            mod.export_library(os.path.join(parentdir,"auto_schedule-{}".format(layer_name)))
    return "auto_schedule-{}".format(layer_name),True


def can_load(param:dict):
    no_need_dict={"axis":1}
    if(param.get("axis")!=None):
        if((isinstance(param.get("axis"),(int,float))==False)):
            if(len(param.get("axis"))>1):
                return False
            
    return True

if(__name__=="__main__"):
    layer_name="Conv1D-__filters___4,__kernel_size____1_,__strides___1,__padding____valid_,__groups___4,__activation____linear_,__use_bias___True,__input_shape____10,_500_,__batch_size___1_.h5"
    file_dict={}
    with open("{}/label.txt".format(parentdir),"r") as label:
        
        for line in label:
            l=line.split("\\")
            file_dict[l[0]]=eval(l[1])
    elfdir=parentdir
    path="{}/{}".format(parentdir,layer_name)
    #os.path.join(parentdir,layer_name)
    l_name=layer_name[0:layer_name.find(".h5")]
    param=file_dict[l_name]
    if(can_load(param)):
        input_shape=param["input_shape"]
        input_shape.insert(0,param["batch_size"])
        shape_dict={"input":input_shape}
       
        layer_so_name=l_name+".so"
        try:
            mod=keras.models.load_model(path,compile=True)
            mod,params=relay.frontend.from_keras(mod,layout="NWC",shape=shape_dict)
            # mod["main"]
            print(mod)
            # mod["main"]
            
            two,doshchedule=auto_schedule_run_tuning(mod,params,elfdir,layer_so_name)
            if(doshchedule):
                one=compile_directly(mod,params,elfdir,layer_so_name)
            
            
        except Exception as e:
            print("{}\n{}-{} not support\n".format(e,layer_so_name,file_dict[l_name]))
               
        #with auto_scheduler.ApplyHistoryBest(log_file):
            #with tvm.transform.PassContext(opt_level=3,config={"relay.backend.use_auto_scheduler": True}):
            
sys.exit()