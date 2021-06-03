import tvm
import tvm.relay as relay
from tvm import relay, auto_scheduler
import tensorflow.keras as keras
import numpy as np
import os
layer_path="/media/s3lab-1/570c75c9-cb6f-4ce6-bcc5-9f636c7310f3/s3lab-1/workspace/Deeplearning_Framework/TVM/tvm2/apps/layers_dataset/layers"
threads=2
ctx = tvm.cpu(0)
target = tvm.target.Target("llvm -mcpu=skylake-avx512")