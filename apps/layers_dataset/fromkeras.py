from re import T
from click.decorators import option
import tensorflow.keras as keras
import keras.layers as layers
import tensorflow as tf
import sys
import inspect
import math
import pprint

from tensorflow.python import module
from read_layers import read_layers_xml 
from param_type import PType
from layer_param_pair import LayerParampair
import logging
import os
logging.basicConfig(filename=os.path.join(os.getcwd(),'log.txt'),level=logging.FATAL)
layers_dir=os.path.join(os.getcwd(),"layers")
class paramPair(object):
    param=None
    default=None
    def __init__(self,param,default) -> None:
        super().__init__()
        self.param = param
        self.default = default
    
    def __eq__(self, o: object) -> bool:
        if(isinstance(o,paramPair)):
            if(o.param==self.param and o.default==self.default):
                return True
        return False

def apppend_func(var:list,name):
    if(var!=None):
        var.append(name)
    else:
        var=[name]
    return var

def do_insprct():
    layers_content=read_layers_xml("./layers_list.xml")
    param_dict=dict()
    layers_content.keys()
    for layer in layers_content.keys():
        if hasattr(layers,layer):
            constructor = getattr(getattr(layers,layer),'__init__')
            r = inspect.signature(constructor)
            layer_param={}
            for param,default in r.parameters.items():
                #fun =lambda i:[].append(layer) if i==None else i.append(layer)#三目运算
                tmp = (param,default.default)
               
                layer_param[param]=default.default
            param_dict[layer]=layer_param
        else:
            print("No layer named {}\n".format(layer))
        
   
    # with open("./layers_param_data.py","w")as file:
    #     pprint.pp(param_dict,indent=1,width=1,depth=99,stream=file)
    return param_dict
def exectue_param(layer,param):
    if hasattr(layers,layer):    
        try:
            d1 = eval("layers.{}(**{})".format(layer,param))
            model=keras.Sequential([d1])
            model.save(os.path.join(layers_dir,"{}{}".format(str(param),".h5")),save_format="h5")
        except:
            print("layer:{} does not have params={}".format(layer,param))
            logging.fatal("layer:{} does not have params={}".format(layer,param))


def test_permute():
    layers_content=read_layers_xml("./layers_list_test.xml")
    params={}
    for node in layers_content:
        i=0
        
        while(True):
            
            params,isfinish=node.next_permutation(params)
            print("name={},param={}".format(node.Lname,params))
            exectue_param(node.Lname,params)
            if(params==None or isfinish):
                print(i)
                sys.exit()
            i=i+1
        
    pass
#do_insprct()
test_permute()
