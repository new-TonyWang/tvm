import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pprint
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
params={'filters': 256, 'kernel_size': 3, 'padding': 'valid', 'dilation_rate': 5, 'depth_multiplier': 10, 'activation': 'elu', 'use_bias': True, 'input_shape': [10, 1000], 'batch_size': 1}
d1 = layers.SeparableConv1D(**params)
model=keras.Sequential([
    d1
])


d2={"d21":21,
    "d22":22
}
d3={"d31":31,
    "d32":32
}
d = {"d2":d2,"d3":d3}
pprint.PrettyPrinter(indent=4)

with open("./testdect.txt","w")as file:
    pprint.pp(d,indent=1,width=1,depth=3,stream=file)
