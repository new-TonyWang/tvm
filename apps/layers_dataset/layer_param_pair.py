import copy
from layers_param_data import global_table
from generate_param_value import *
class LayerParampair(object):
   
    class Permu:
        except_param={#和输入有关的参数
            'input_shape':get_input_shape,
            'batch_size':get_value,
        }
        """
        permu_conf存的是(参数名称,参数配置)
        """
        
        def init_serch_space(self,param_func):
            """
            通过配置初始化枚举函数并返回(参数名称,枚举函数)
            """
            funcs={}
            for funcname in self.permu_conf.keys():
                func_config=self.permu_conf.get(funcname)
                if(param_func.get(funcname)!=None):
                    print(funcname)
                    funcs[funcname]=param_func[funcname](**func_config)
                if(self.except_param.get(funcname)!=None):
                    funcs[funcname]=self.except_param[funcname](**func_config)
            return funcs
        
        def get_useful_conf(self,permu_conf:dict,params:dict):
            """
            通过参数列表和except_param来丢掉配置中多余的配置
            """
            final_keys={}
            for param_name in params.keys():
                if(permu_conf.get(param_name)!=None):
                    final_keys[param_name]=permu_conf[param_name]
            for param_name in self.except_param.keys():
                if(permu_conf.get(param_name)!=None):
                    final_keys[param_name]=permu_conf[param_name]

            return final_keys

        def get_first_permu(self,funcs={}):
            final_param_dict={}
            for p in funcs.keys():
                retuen_value=funcs[p]()
                final_param_dict[p]=retuen_value.rv
            return final_param_dict
        
        def next_permutation(self,previous_config):
           return self.state.next_permutation(previous_config)
    

        def __init__(self,permu_conf,params_func):
            """
            从全局表中获取参数函数，并从配置中读取参数
            """
            
                     
            self.permu_conf=self.get_useful_conf(permu_conf,params_func)
            self.params_func=self.init_serch_space(params_func)
            self.state=LayerParampair.running_state(self,[x for x in self.params_func.keys()])#状态机

    class absstate():
        
        def __init__(self,env) -> None:
            self.env=env

        def next_permutation(self):
            pass

    class start_state(absstate):
        def __init__(self, env) -> None:
            super().__init__(env)

        def next_permutation(self,previous_config=None):
            final_param_dict={}
            func=self.env.params_func
            for p in func.keys():
                retuen_value=func[p]()
                final_param_dict[p]=retuen_value.rv
            self.env.state=LayerParampair.running_state(self.env,[x for x in self.env.params_func.keys()])
            
            return final_param_dict
            
    class running_state(absstate):
        def __init__(self, env,funclist=[]) -> None:
            self.ptr=0
            self.funclist=funclist
            self.len=(len(self.funclist))
            self.boollist=[False]*self.len
            super().__init__(env)
        
        
        def next_permutation(self,previous_config):
            func=self.env.params_func
            while(self.ptr<self.len-1):
                fname=self.funclist[self.ptr]
                p=func[fname]()
                #print("p={}".format(fname))
                self.boollist[self.ptr]=p.isEnd
                previous_config[fname]=p.rv
                self.ptr=self.ptr+1

            if(self.ptr==self.len-1):
                fname=self.funclist[self.ptr]
                p=func[fname]()
                self.boollist[self.ptr]=p.isEnd
                previous_config[fname]=p.rv
                
                while(self.boollist[self.ptr] and self.ptr>=0):
                    #print("-------------------------------------------------------{}".format(self.ptr))
                    self.ptr=self.ptr-1                            
            
            if(self.ptr==-1 and self.boollist[0]):
                return previous_config,True
            else:
                
                return previous_config,False

    def next_permutation(self,previous_config):
        return self.permutor.next_permutation(previous_config)

    def __init__(self,Lname,permu_conf=None):
        super().__init__()
        self.Lname = Lname
        params=global_table[self.Lname]
        if(params==None):
            print("no layer {}, skip permu".format(self.Lname))
        self.permutor = LayerParampair.Permu(permu_conf,params)
       

    
    

    
            


    