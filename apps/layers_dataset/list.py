1.每种算子都需要设置'input_shape'和'batch_size',如果该算子没有其他的参数则只需要设置这两个值(start, end和step_size)
2.如果参数是数字的话给出数字的取值范围和步长(start, end和step_size)
3.多维数组则需要给出一个单独的能够包含所有列的范围的区间和步长。例如kernel_size，想要取(3, 3), (5, 5), (5, 7), (7, 7), 则需要给出start = 3, end = 7, step_size = 2
4.如果参数是字符串，例如padding有valid和same, 则需要给出全部的字符串
5.标注一下每种算子最少需要多少数据，我可以稍微修改一些参数
6.如果有对推理不重要的参数，直接把参数删去
7.参数之间的制约关系也帮忙标注一下，我把两个参数分开设置
8.bool类型的参数不需要设置，因为只有两个选项

'LeakyReLU': {
    'alpha'
},

'PReLU': {
    'shared_axes'
},

'ELU': {
    'alpha'  # 1.0
},

'ThresholdedReLU': {
    'theta'
},
'Softmax': {
    'axis'  # -1
},

'Conv1D': {
    'filters'
    'kernel_size'
    'strides'
    'padding'

    'dilation_rate'
    'groups'
    'activation'

},
'Conv2D': {
    'filters'
    'kernel_size'
    'strides'
    'padding'
    'dilation_rate'
    'groups'
    'activation'

},

'Conv3D': {
    'filters'
    'kernel_size'
    'strides'
    'padding'

    'dilation_rate'
    'groups'
    'activation'

},

'Conv1DTranspose': {
    'filters'
    'kernel_size'
    'strides'
    'padding'
    'output_padding'

    'dilation_rate'
    'activation'
},

'Conv2DTranspose': {
    'filters'
    'kernel_size'
    'strides'
    'padding'
    'output_padding'

    'dilation_rate'
    'activation'

},

'Conv3DTranspose': {
    'filters'
    'kernel_size'
    'strides'
    'padding'
    'output_padding'
    'dilation_rate'
    'activation'

},

'SeparableConv1D': {
    'filters'
    'kernel_size'
    'strides'
    'padding'
    'dilation_rate'
    'depth_multiplier'
    'activation'
    'depthwise_initializer'
    'pointwise_initializer'
    'bias_initializer'
},

'SeparableConv2D': {
    'filters'
    'kernel_size'
    'strides'
    'padding'
    'dilation_rate'
    'depth_multiplier'
    'activation'

},

'DepthwiseConv2D': {
    'kernel_size'
    'strides'
    'padding'
    'depth_multiplier'
    'dilation_rate'
    'activation'

},

'UpSampling1D': {
    'size'
},

'UpSampling2D': {
    'size'
    'interpolation'
},

'UpSampling3D': {
    'size'

},
'ZeroPadding1D': {
    'padding'
},

'ZeroPadding2D': {
    'padding'

},

'ZeroPadding3D': {
    'padding'

},

'Cropping1D': {
    'cropping'
},

'Cropping2D': {
    'cropping'

},

'Cropping3D': {
    'cropping'

},

'Masking': {
    'mask_value'
},


'Activation': {
    'activation'
},
'Reshape': {
    'target_shape'
},

'Permute': {
    'dims'
},


'Dense': {
    'units'
    'activation'


},

'ActivityRegularization': {
    'l1'
    'l2'
},

'Embedding': {
    'input_dim'
    'output_dim'
    'mask_zero'
    'input_length'
},

'LocallyConnected1D': {
    'filters'
    'kernel_size'
    'strides'
    'padding'

    'activation'

},

'LocallyConnected2D': {
    'filters'
    'kernel_size'
    'strides'
    'padding'

    'activation'
},

'Concatenate': {
    'axis'  # -1
},

'Add': { },

 'Subtract': { 
              },

 'Multiply': { 
              },

 'Average': { 
             },

 'Maximum': { 
             },

 'Minimum': { 
             },

'Dot': {
    'axes'
    'normalize'
},

'GaussianNoise': {
    'stddev'
},

'GaussianDropout': {
    'rate'
},

'LayerNormalization': {
    'axis'  
    'epsilon' 
    'center'
    'scale'
},

'BatchNormalization': {
    'axis'  
    'momentum' 
    'epsilon' 
    'center'
    'scale'
    'renorm'
    'renorm_clipping'
    'renorm_momentum'  
},

'MaxPooling1D': {
    'pool_size'
    'strides'
    'padding'

},

'MaxPooling2D': {
    'pool_size'
    'strides'
    'padding'

},

'MaxPooling3D': {
    'pool_size'
    'strides'
    'padding'

},

'AveragePooling1D': {
    'pool_size'
    'strides'
    'padding'

},

'AveragePooling2D': {
    'pool_size'
    'strides'
    'padding'

},

'AveragePooling3D': {
    'pool_size'
    'strides'
    'padding'

},

'SimpleRNN': {
    'units'
    'activation'

    'recurrent_dropout'
    'return_sequences'
    'return_state'
    'go_backwards'
    'stateful'
    'unroll'
},

'GRU': {
    'units'
    'activation'
    'recurrent_activation'

    'dropout'
    'recurrent_dropout'
    'implementation'
    'return_sequences'
    'return_state'
    'go_backwards'
    'stateful'
    'unroll'
    'time_major'
    'reset_after'
},

'LSTM': {
    'units'
    'activation'
    'recurrent_activation'
    'unit_forget_bias'
    'dropout'
    'recurrent_dropout'
    'implementation'
    'return_sequences'
    'return_state'
    'go_backwards'
    'stateful'
    'time_major'
    'unroll'
},
