from genericpath import exists
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
import copy
"""
使用TVM批量编译算子
"""
layer_path="Conv2d_class"#keras文件目录
log_path=layer_path+"/logs"#TVM日志目录(无需修改)
threads=2
ctx = tvm.cpu(0)
target = tvm.target.Target("llvm -mcpu=skylake-avx512")
parentdir=os.path.join(os.getcwd(),layer_path)
logdir=os.path.join(os.getcwd(),log_path)
filelist=os.listdir(os.path.join(os.getcwd(),layer_path))
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
os.environ["TVM_BIND_THREADS"]="0"#是否将将线程绑定到核心(0为否)
os.environ["TVM_NUM_THREAD"]="48"#使用几个TVM线程
print(filelist[0])

def compile_directly(mod,params,parentdir,layer_name):
    with tvm.transform.PassContext(opt_level=3):
        mod=relay.build(ir_mod=mod,target=target,params=params)
    mod.export_library(os.path.join(parentdir,layer_name))
    return layer_name

def auto_schedule_run_tuning(mod,params,parentdir,layer_name):
    """
        使用autoschedule优化模型(建议使用)
    """
    print("Extract tasks...")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    if(len(tasks)==0):#没有task
        
        return compile_directly(mod,params,parentdir,layer_name),False
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

def auto_TVM_run_tuning(mod,params):
        """
        使用autoTVM优化模型(未使用)
        """
        print("auto_TVM Begin tuning...")
        log_file = "auto_TVM-%s-%s-%s.json" % (name, target.kind.name)
        tuning_option = {
        "log_filename": log_file,
        "tuner": "random",
        "early_stopping": None,
        "measure_option": autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.LocalRunner(
                number=1, repeat=20, min_repeat_ms=0, enable_cpu_cache_flush=True
            ),
        ),
    
        }
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

        def tune_kernels(
        tasks, measure_option, tuner="gridsearch", early_stopping=None, log_filename="tuning.log"
        ):

            for i, task in enumerate(tasks):
                prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

                # create tuner
                if tuner == "xgb" or tuner == "xgb-rank":
                    tuner_obj = XGBTuner(task, loss_type="rank")
                elif tuner == "ga":
                    tuner_obj = GATuner(task, pop_size=50)
                elif tuner == "random":
                    tuner_obj = RandomTuner(task)
                elif tuner == "gridsearch":
                    tuner_obj = GridSearchTuner(task)
                else:
                    raise ValueError("Invalid tuner: " + tuner)

                # do tuning
                n_trial = len(task.config_space)
                tuner_obj.tune(
                    n_trial=n_trial,
                    early_stopping=early_stopping,
                    measure_option=measure_option,
                    callbacks=[
                        autotvm.callback.progress_bar(n_trial, prefix=prefix),
                        autotvm.callback.log_to_file(log_filename),
                    ],
                )
            print("auto_TVM Begin tuning...")
            tune_kernels(tasks, **tuning_option)
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3,config={"relay.backend.use_auto_scheduler": True}):
                mod=relay.build(ir_mod=mod,target=target,params=params)

                mod.export_library("cropping2.1d.so")

def can_load(fname:str,param:dict):
    """
    通过控制台的报错输出来修改该函数，尽可能得避开TVM不支持的算子参数
    """
    if(param.get("axis")!=None):
        if((isinstance(param.get("axis"),(int,float))==False)):
            if(len(param.get("axis"))>1):
                return False
    if(fname.find("Conv1D")!= -1 and param.get("padding")=="causal"):
        return False
    if(fname.find("SeparableConv1D")!=-1):
        return False       
    if(fname.find("Conv1DTranspose")!=-1):
        return False     
 
    return True

def get_layout(name:str):
    if(name.find("1D")!=-1):
        return "NWC"
    elif(name.find("2D")!=-1):
        return "NHWC"
    elif(name.find("3D")!=-1):
        return "NDHWC"
    else:
        return "NCHW"

if(__name__=="__main__"):
    if(len(filelist)<=0):
        sys.exit()
    file_dict={}
    with open("{}/label.txt".format(parentdir),"r") as label:
        
        for line in label:
            l=line.split("\\")
            file_dict[l[0]]=eval(l[1])
    elfdir=parentdir+"/elf"
    if(os.path.exists(elfdir)==False):
        os.mkdir(elfdir)
    existing=[]
    already_generate=os.listdir(os.path.join(os.getcwd(),elfdir))

    for i in range(len(already_generate)):#读取输出文件的目录，来保证多次运行该文件值不会重复得编译算子，减少运行时间
        if(already_generate[i].endswith(".so")):
            already_generate[i] = already_generate[i][0:already_generate[i].find(".so")]+".h5"
        
    with open("{}/label.txt".format(elfdir), "w") as label:
        for layer_name in filelist:
            if(layer_name.endswith(".h5")==False or layer_name in already_generate):
                continue
            path="{}/{}".format(parentdir,layer_name)
            #os.path.join(parentdir,layer_name)
            l_name=layer_name[0:layer_name.find(".h5")]
            param=file_dict.get(l_name)
            if(param==None):
                continue
            if(can_load(l_name,param)):
                input_shape=param["input_shape"]
                input_shape.insert(0,param["batch_size"])
                shape_dict={"input":input_shape}
                layer_so_name=l_name+".so"
                try:
                    mod=keras.models.load_model(path,compile=True)
                    mod,params=relay.frontend.from_keras(mod,layout=get_layout(l_name),shape=shape_dict)
                    # mod["main"]
                    print(mod)
                    # mod["main"]
                    
                    two,doshchedule=auto_schedule_run_tuning(mod,params,elfdir,layer_so_name)
                    if(doshchedule):
                        one=compile_directly(mod,params,elfdir,layer_so_name)
                        label.write("{}\\{}\n{}\\{}\n".format(copy.deepcopy(one),str(param),two,str(param)))
                    else:
                        label.write("{}\\{}\n".format(two,str(param)))
                    
                except Exception as e:
                    print("{}\n{}-{} not support\n".format(e,layer_so_name,file_dict[l_name]))
                    continue
               
        #with auto_scheduler.ApplyHistoryBest(log_file):
            #with tvm.transform.PassContext(opt_level=3,config={"relay.backend.use_auto_scheduler": True}):
            
sys.exit()