import tensorflow.keras as keras
import keras.layers as layers
import pprint
param=param=param={'units': 10, 'activation': 'softmax', 'recurrent_activation': 'softmax', 'use_bias': False, 'unit_forget_bias': True, 'implementation': 2, 'return_sequences': True, 'return_state': False, 'go_backwards': True, 'stateful': True, 'time_major': False, 'unroll': True, 'input_shape': (20, 30), 'batch_size': 3}
d1 = layers.LSTM(**param)
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
