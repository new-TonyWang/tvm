import inspect

import copy
import sys
from itertools import product
class retValue():
    def __init__(self,rv,isEnd=False) :
        self.rv=rv
        self.isEnd=isEnd
def get_item(start,end,step_size=1,var_list=[],enable_perum=True):
    """
    用于生成下一个list中的数据
    """
    lvar_list=var_list
    lstart=start
    lstep_size=step_size
    lend=end
    current=start-step_size
    def next():
        nonlocal current
        current = current+lstep_size
        if(current>lend):
            current=lstart
            return retValue(lvar_list[current])
        else:
            if(current+lstep_size>lend):#假设没有溢出的情况下可以使用
                if(enable_perum):
                    return retValue(lvar_list[current],True)
                else:
                    return retValue(lvar_list[current])
            return retValue(lvar_list[current])
    return next   

def get_activation(startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    """
    循环获取所有激活函数,current!=none的时候是直接获取值
    """
    activtions = ['softmax', 'elu', 'selu', 'softplus', 'softsign', 'swish', 'relu', 'tanh', 'sigmoid', 'exponential', 'hard_sigmoid', 'linear']
    
    endrun=min(len(activtions)-1,endrun)
    return get_item(startrun,endrun,steprun,activtions,enable_perum)

activation =get_activation(0,99,1,True)
# i =0
# while(i<15):
#     rv=activation()
#     print("rv={},isEnd={}".format(rv.rv,rv.isEnd))
#     i=i+1

def get_bool(enable_perum=True):
    ptr =-1
    options=[True,False]
    def get_next():
        nonlocal ptr
        ptr=ptr+1
        if(ptr+1==len(options)):
            if(enable_perum):
                    return retValue(options[ptr],True)
        elif(ptr==len(options)):
            ptr=0
        return  retValue(options[ptr])
    return get_next


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

def get_value(start,end,step_size=1,enable_perum=True):
    """
    用于生成下一个int,bool或者float
    """
    lstart=start
    lstep_size=step_size
    lend=end
    current=start-step_size
    def next():
        nonlocal current
        current = current+lstep_size
        if(current>lend):
            current=lstart
            return retValue(current)
        else:
            if(current+step_size>lend):
                if(enable_perum):
                        return retValue(current,True)
                
            return retValue(current)
           
    return next
def get_input_shape(dim,start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    assert start<=endg and startrun <=endrun
    signleDim=[i for i in range(start,endg+1,step_size)]
    #print(signleDim)
    items=list(product(signleDim,repeat=dim))
    endrun=min(len(items)-1,endrun)
    return  get_item(startrun,endrun,steprun,items,enable_perum)
    

def get_shared_axes(dim=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    assert startrun <=endrun
    """
    PReLU shared axes
    """
    ldim=dim
    def combine(num,total):
        """
        寻找所有组合
        """
        result=[]
        tmp = [i for i in range(1,num+1)]
        tmp.append(total+1)
        #print(tmp)
        j = 0
        while(j<num):
            result.append(copy.deepcopy(tmp[0:num]))
            j=0
            while(j<num and (tmp[j]+1)== tmp[j+1]):
                tmp[j]=j+1
                j=j+1
            tmp[j]=tmp[j]+1
        return result 

    def generate_all_cases(dim):
        result=[]
        len=dim
        for depth in range(1,dim+1):
            result.extend(combine(depth,dim))
        return result
     
    all_combine=generate_all_cases(ldim)
    endrun=min(len(all_combine)-1,endrun)
    return get_item(startrun,endrun,steprun,all_combine,enable_perum)
    
def kernel_size_dispatch(kernel_kind:str,start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    signleDim = [i for i in range(start,endg+1,step_size)]
    items=[]
    
    if(kernel_kind.find("1D")!=-1):
        items.append(signleDim)

    elif(kernel_kind.find("2D")!=-1):
        for i in signleDim:
            for j in signleDim:
                items.append((i,j))
                
    elif(kernel_kind.find("3D")!=-1):
        for i in signleDim:
            for j in signleDim:
                for k in signleDim:
                    items.append((i,j,k))

    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_stride_or_dilation_rate_pool_size(start,endg,step_size=1,startrun=0,steprun=1,endrun=sys.maxsize,enable_perum=True):
    #检查stride合法性
    items = [i for i in range(start,endg+1,step_size)]
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_padding(startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    paddings=['valid','same']
    endrun=min(len(paddings)-1,endrun)
    return get_item(startrun,1,steprun,paddings,enable_perum)

def get_group(start,channelnum,endrun=sys.maxsize,startrun=0,step_size=1,steprun=1,enable_perum=True):
    """
    group的数量一定要比channel少
    """
    items = [i for i in range(start,channelnum+1,step_size)]
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_data_format(step_size=1,enable_perum=True):
    format=['channels_first','channels_last']
    return get_item(0,1,step_size,format,enable_perum)

def get_strides2D_and_dilation_rate_pool_size(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    signleDim = [i for i in range(start,endg+1,step_size)]
    items=[]
    for i in signleDim:
        for j in signleDim:
            items.append((i,j))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)


def get_strides3D_and_dilation_rate_pool_size(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    signleDim = [i for i in range(start,endg+1,step_size)]
    items=[]
    for i in signleDim:
        for j in signleDim:
            for k in signleDim:
                items.append((i,j,k))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def output_padding_dispatch_for_Transpose(kernel_kind:str,start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    signleDim = [i for i in range(start,endg+1,step_size)]
    items=[]
    
    if(kernel_kind.find("1D")!=-1):
        items.append(signleDim)

    if(kernel_kind.find("2D")!=-1):
        for i in signleDim:
            for j in signleDim:
                items.append((i,j))
                
    if(kernel_kind.find("3D")!=-1):
        for i in signleDim:
            for j in signleDim:
                for k in signleDim:
                    items.append((i,j,k))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_depth_multiplier(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    items = [i for i in range(start,endg+1,step_size)]
    
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_size1D_and_padding(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    items = [i for i in range(start,endg+1,step_size)]
    
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_size2D_and_padding(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    signleDim = [i for i in range(start,endg+1,step_size)]
    items=[]
    for i in signleDim:
            for j in signleDim:
                items.append((i,j))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_size3D_and_padding(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    signleDim = [i for i in range(start,endg+1,step_size)]
    items=[]
    for i in signleDim:
            for j in signleDim:
                for k in signleDim:
                    items.append((i,j,k))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_interpolation(step_size=1,enable_perum=True):
    interpolation=['nearest','bilinear']

    return  get_item(0,1,step_size,interpolation,enable_perum)    

def get_croping1D(start,inputshape,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    signleDim = [i for i in range(start,inputshape//2+1,step_size)]
    items=[]
    for i in signleDim:
            for j in signleDim:
                items.append((i,j))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_croping2D(start,end,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    end=end//2+1
    assert start<=end
    items1d = [i for i in range(start,end//2+1,step_size)]
    
    item=[]
    for i in items1d:
            for j in items1d:
                item.append((i,j))
    items=list(product(item,repeat=2))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)
    
def get_croping3D(start,end,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    items1d = [i for i in range(start,end//2+1,step_size)]
    item=[]
    for i in items1d:
            for j in items1d:
                item.append((i,j))
    items=list(product(item,repeat=3))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_target_shape(inputshape,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    total=0
    items=[]
    lshape = inputshape
    for ele in range(0, len(inputshape)):
        total = total * inputshape[ele]
    #case1 仅增加维度但是不改变元素个数
    tmp=copy.deepcopy(lshape)
    tmp.append(1)
    items.append(tmp)
    #case 降维融合
    last=inputshape[-1]
    if(len(lshape)>1):
        dp=copy.deepcopy(lshape[0:len(lshape)-1])
        dp[-1]=dp[-1]*lshape[len(lshape)-1]
        items.append(copy.deepcopy(dp))
        for i in range(len(dp)-2,-1,-1):
            dp[i] = dp[i+1]*dp[i]
            items.append(copy.deepcopy(dp[0:i+1]))
    else:
        items.append(lshape)

    endrun=min(len(items)-1,endrun)
    return  get_item(startrun,endrun,steprun,items,enable_perum)

target_shape = get_target_shape([1,2,3,4,5])
target_shape()

def get_next_permute(inputshape,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    dim = len(inputshape)
    def combine(num,total):
        """
        寻找所有组合
        """
        result=[]
        tmp = [i for i in range(1,num+1)]
        tmp.append(total+1)
        #print(tmp)
        j = 0
        while(j<num):
            result.append(copy.deepcopy(tmp[0:num]))
            j=0
            while(j<num and (tmp[j]+1)== tmp[j+1]):
                tmp[j]=j+1
                j=j+1
            tmp[j]=tmp[j]+1
        return result 

    def generate_all_cases(dim):
        result=[]
        len=dim
        for depth in range(1,dim+1):
            result.extend(combine(depth,dim))
        return result
     
    all_combine=generate_all_cases(dim)
    endrun=min(len(all_combine)-1,endrun)
    return get_item(startrun,endrun,steprun,all_combine,enable_perum)

# permute = get_next_permute(1,2,[10,20,3])
# permute()

def get_implementation(startrun=0,endrun=1,steprun=1,enable_perum=True):
    items=[1,2,3]
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)


def get_axes(inputshape,startrun=0,endrun=1,steprun=1,enable_perum=True):
    dim = len(inputshape)
    def combine(num,total):
        """
        寻找所有组合
        """
        result=[]
        tmp = [i for i in range(1,num+1)]
        tmp.append(total+1)
        #print(tmp)
        j = 0
        while(j<num):
            result.append(copy.deepcopy(tmp[0:num]))
            j=0
            while(j<num and (tmp[j]+1)== tmp[j+1]):
                tmp[j]=j+1
                j=j+1
            tmp[j]=tmp[j]+1
        return result 

    def generate_all_cases(dim):
        result=[]
        len=dim
        for depth in range(1,dim+1):
            result.extend(combine(depth,dim))
        return result
     
    all_combine=generate_all_cases(dim)
    endrun=min(len(all_combine)-1,endrun)
    return get_item(startrun,endrun,steprun,all_combine,enable_perum)
