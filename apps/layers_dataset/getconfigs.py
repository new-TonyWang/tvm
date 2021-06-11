import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pprint
import os
"""
若某个算子的生成一直出问题则可以使用该文件排查参数错误的原因（复制粘贴命令行的params和算子）
"""
os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
params={'filters': 256, 'kernel_size': 5, 'padding': 'same', 'output_padding': 10, 'dilation_rate': (5, 1), 'activation': 'swish', 'use_bias': False, 'input_shape': [200, 200, 10], 'batch_size': 2}
d1 = layers.Conv2DTranspose(**params)
model=keras.Sequential([
    d1
])


# d2={"d21":21,
#     "d22":22
# }
# d3={"d31":31,
#     "d32":32
# }
# d = {"d2":d2,"d3":d3}
# pprint.PrettyPrinter(indent=4)

# with open("./testdect.txt","w")as file:
#     pprint.pp(d,indent=1,width=1,depth=3,stream=file)
