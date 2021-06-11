import sys
from layer_info import keras_layer
import re
from xml.etree.ElementTree import XMLPullParser,ElementTree
from classification_tree import *
"""
读取xml文件并得到所有算子的参数设置
"""

def read_layers_xml(path):
    parser = XMLPullParser(["start", "end", "comment", "pi", "start-ns", "end-ns"])
    with open(path,'r') as file:
        for line in file:
            parser.feed(line)
    root = Ptree(get_next_event(parser))
    
    
    return root.get_layer_nodes()


def get_next_event(parser=None):
    plist=[]
    ptr=-1
    size=0
    if (parser!=None):
        for event,elem in parser.read_events():
            plist.append((event,elem))
    plist.append(("xmlend",None))#结束符
    def next_event():
        nonlocal ptr
        nonlocal plist
        ptr=ptr+1
        event,elem= plist[ptr]
      
        return event,elem
    return next_event
#read_layers_xml("./layers_list.xml")
# for i in a:
#     print("{}\n".format(i))
