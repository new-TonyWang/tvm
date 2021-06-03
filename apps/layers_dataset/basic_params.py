
def funchook(func):
    def wrapper():
        print("call Nothing")
        return 0
    return wrapper

def get_activation():
    """
    循环获取所有激活函数
    """
    ptr = 0
    activtions = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'swish', 'relu', 'tanh', 'sigmoid', 'exponential', 'hard_sigmoid', 'linear']
    def get_next():
        nonlocal ptr
        if(ptr<len(activtions)):
            result = activtions[ptr]
            ptr = ptr+1
            return result
        else:
            ptr = ptr%len(activtions)
            return activtions[ptr]

    return get_next()

def get_use_bias():
    switch =0
    options=[True,False]
    def get_next():
        nonlocal switch
        switch=(switch+1)%2
        return  options[switch]
    return get_next()


def get_kernel_initializer():
    """
    对推理无影响
    """
    return 'glorot_uniform'

def bias_initializer():
    """
    对推理无影响
    """
    return 'zero'


_Global_var_dict={
    "activation":None,
    "use_bias":True,
    "kernel_initializer":'glorot_uniform',
    "bias_initializer":'zeros',
    "kernel_regularizer":None,
    "bias_regularizer":None,
    "activity_regularizer":None,
    "kernel_constraint":None,
    "bias_constraint":None}
