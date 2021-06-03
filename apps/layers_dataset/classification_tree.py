import copy
from layer_param_pair import LayerParampair
import re
class Ptree():
    
    class Node(object):
        
        def __init__(self, name, param_config=None):
            super().__init__()
            self.__name = name
            self.__param_config = param_config

        def get__param_config(self):
            return self.__param_config

        def get__name(self):
            return self.__name

        def __str__(self) -> str:
            return "name = {}".format(self.__name)

    class treeNode(Node):
        def __init__(self, name=None, param_config=None):
            self.__param_config = param_config
            self.__name = name
            self.param_node=[]
            super().__init__(name, param_config)
        
    class NodeGRoup(treeNode):
        
        def __init__(self, name=None, param_config=None):
            self.__childs = []
            self.__size = 0
            super().__init__(name, param_config=param_config)
        
        def get__childs(self):
            return self.__childs

        def add_child(self, child):

            self.__childs.append(child)
            self.__size = self.__size+1

        def __str__(self) -> str:
            out=[]
            for item in self.__childs:
                out.append(item.get__name())
            return out
        
    class layerNodeGroup(NodeGRoup):
        def __init__(self, name=None, param_config=None):
            super().__init__(name, param_config)

    class rootNodeGroup(NodeGRoup):
        def __init__(self):
            super().__init__("root", None)

    def create_NodeGRoup_param(self, parent, name, param_conf=None):
        """
        先从祖先节点获取参数，再使用新读取的参数更新祖先节点的参数
        """

        child_conf = copy.deepcopy(parent.get__param_config())
        if(param_conf):
            child_conf.update(param_conf)
        child = Ptree.NodeGRoup(name,param_config=child_conf)
        return child

    def create_layerNodeGRoup_param(self, parent, name, param_conf=None):
        """
        先从祖先节点获取参数，再使用新读取的参数更新祖先节点的参数
        """

        child_conf = copy.deepcopy(parent.get__param_config())
        if(param_conf):
            child_conf.update(param_conf)
        child = Ptree.NodeGRoup(name,param_config=child_conf)
        self.layernodes.append((name,child))#在叶子节点中建立hash
        return child

    

    def create_Node_param(self, parent,name, param_conf=None):
        child_conf = copy.deepcopy(parent.get__param_config())
        if(param_conf):
            child_conf.update(param_conf)
        child = Ptree.Node(name,param_config=child_conf)
        
        
        return child



    def rinflate(self, parent: NodeGRoup, next_event):
        while(True):
            event, elem = next_event()
            if(event == "xmlend"):#读到文档结尾
                    return
            param_conf = elem.attrib#当前从xml读取的参数
           
            name = param_conf.get('name')
            if(name!=None):
                del param_conf['name']
            for key in param_conf.keys():
                if(isinstance(param_conf[key],str)):
                    if(re.match(".*[A-Z]+.*",param_conf[key]) or re.match(".*[a-z]+.*",param_conf[key])):
                        continue
                    if (isinstance(eval(param_conf[key]),float)):
                        param_conf[key] = float(param_conf.get(key))
                    elif(isinstance(eval(param_conf[key]),int)):
                        param_conf[key] = int(param_conf.get(key))
            if(event == "start"):
                print(event)
                if(elem.tag == "class"):
                    child = self.create_NodeGRoup_param(parent, name, param_conf)
                    #self.rinflate(child, next_event)
                    print("this is a group")

                elif(elem.tag == "data"):

                    print("this is root")
                    default_param_config = param_conf
                    child = Ptree.NodeGRoup(param_config=default_param_config)
                   

                elif(elem.tag == "layer"):
                    child = self.create_layerNodeGRoup_param(parent, name, param_conf)
                    print("this is layer")

                elif(elem.tag == "param"):
                    child = self.create_Node_param(parent, name, param_conf)
                print(param_conf)
                #print("depth is {}".format(tree_depth))

            elif(event == "end"):
                print(event)
                return
            elif(event == "comment"):
                print(event)
                continue
            elif(event == "pi"):
                print(event)
                continue
            elif(event == "start-ns"):
                print(event)
                continue
            elif(event == "end-ns"):
                print(event)
                break
            else:
                continue
            self.rinflate(child, next_event)
            
            parent.add_child(child)

    def inflate(self, rootNode,parser):
        return self.rinflate(rootNode, parser)
    
    def __init__(self,parser):
        self.rootde = self.rootNodeGroup()
        self.layernodes=[]
        self.inflate(self.rootde,parser)

    def get_layer_nodes(self):
        layers_and_param=[]
        for name,node in self.layernodes:#name为算子名称
            param_conf={}
            params_nodes=node.get__childs()
            for pnode in params_nodes:  #node为算子的参数
                pname=pnode.get__name()
                pparam_conf=pnode.get__param_config()
                param_conf[pname]=pparam_conf
            layers_and_param.append(LayerParampair(name,param_conf))   
        #    layers_and_param[key]=param_conf
        return layers_and_param   
        #return copy.deepcopy(self.__layernodes)
