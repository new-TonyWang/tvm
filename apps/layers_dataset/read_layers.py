import sys
from layer_info import keras_layer
import re
from xml.etree.ElementTree import XMLPullParser,ElementTree
from classification_tree import *

def read_layers(path):
    layers_list= []
    kind = None
    with open(path,'r') as file:
       for line in file:
            line=line.strip()
            if(line.startswith('#')):
               kind=line[1:].strip()#读取类型
            elif(re.match('^[^s][0-9]*[a-z]*[A-Z]*',line)):#字符串行
                layers_list.append(keras_layer(line,kind))
            else:#空行
                continue
    return layers_list

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
