import inspect
class PType(object):
    """
    设定每种参数的类型，
    主要针对tuple类型，使用一个list来存储多家tuple的维度
    """

    def get_tuple_shape(self,t,shapelist):
        if(isinstance(t,int)):
            return shapelist
        elif(isinstance(t,tuple)):
            shapelist.append(len(t))
            return self.get_tuple_shape(t[0],shapelist) 

        else:
            return TypeError

    def __init__(self,name=None,t=None) -> None:
        super().__init__()
        self.name=name
        if(isinstance(t,tuple)):
            self.type = self.get_tuple_shape(t,list())
        else:
            self.type = type(t)

    def __str__(self) -> str:
        return self.type.__str__()
    
    def __eq__(self, o: object) -> bool:
        
        if(~isinstance(o,PType)):
            return False
        else:
            if(o.type==self.type):
                return True
            
        return super().__eq__(o)
    def __str__(self) -> str:
        return "{},type={}".formate(self.name,self.type)

    def __hash__(self) -> int:
        return self.name.__hash__()+"{}".format(self.type).__hash__()
       
       
    
            
        