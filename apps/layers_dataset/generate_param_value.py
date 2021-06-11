import inspect

import copy
from pickle import NONE
import sys
from itertools import product
from itertools import combinations
import math
import traceback
"""
该文件实现了所有参数的枚举，不要轻易修改，除非出现异常或者等待了很久却没有任何输出结果
"""
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
    previous=current-step_size
    def next():
        nonlocal current
        nonlocal previous
       
        current = current+lstep_size
       
        if(current>lend):
            try:
                current=lstart+(current-lend)-1
                if(previous==current):
                    if(enable_perum):
                        return retValue(lvar_list[current],True)
                    else:
                        return retValue(lvar_list[current])
                previous= current
                return retValue(lvar_list[current])
            except Exception as e:
                traceback.print_exc()
                print("{}\n lvar_list={}\n lstart={}\n lstep_size={}\n lend={}\n current={}"
                .format(e,lvar_list,lstart,lstep_size,lend,current))

                sys.exit()
            
        else:
            if(current+lstep_size>lend):#假设没有溢出的情况下可以使用
                previous= current
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
    activtions = ['softmax', 'elu', 'selu', 'softplus',  'relu','softsign', 'swish', 'tanh', 'sigmoid', 'exponential', 'hard_sigmoid', 'linear',None]
    
    endrun=min(len(activtions)-1,endrun)
    return get_item(startrun,endrun,steprun,activtions,enable_perum)

# activation =get_activation(0,99,1,True)
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
    # if(lstart==lend):
    #     return next1
    def next():
        nonlocal current
        current = current+lstep_size
        if(current>lend):
            current=lstart+(current-lend)-1
            return retValue(round(current, 3))
        else:
            if(current+step_size>lend):
                if(enable_perum):
                        return retValue(round(current, 3),True)
                
            return retValue(round(current, 3))
           
    return next
def get_input_shape(dim,start,endg,data_format="NWHC",channelstart=1,channelend=sys.maxsize,channel_step=1,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    assert start<=endg and startrun <=endrun
    channelend=min(endg,channelend)
    channelDim=[i for i in range(channelstart,channelend+1,channel_step)]
    signleDim=[i for i in range(start,endg+1,step_size)]
    #print(signleDim)
    datadim=list(product(signleDim,repeat=dim))
    if(data_format=="NWHC"):
        items=[[j for j in i[1]]+[i[0]] for i in product(channelDim,datadim)]
    elif(data_format=="NCHW"):
        items=[[i[0]]+[j for j in i[1]] for i in product(channelDim,datadim)]
    endrun=min(len(items)-1,endrun)
    return  get_item(startrun,endrun,steprun,items,enable_perum)
    

def get_shared_axes(dim=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    assert startrun <=endrun
    """
    PReLU shared axes
    """
    ldim=dim

    def generate_all_cases(dim):
        result=[]
        len=dim
        l = [i for i in range(0,dim+1)]
        for depth in range(1,dim+1):
            result.extend([i for i in  combinations(l,depth)])
        return result
     
    all_combine=generate_all_cases(ldim)
    all_combine.append(-1)#默认加入-1
    endrun=min(len(all_combine)-1,endrun)
    return get_item(startrun,endrun,steprun,all_combine,enable_perum)
    
def kernel_size_dispatch(kernel_kind:str,start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    signleDim = [i for i in range(start,endg+1,step_size)]
    items=[]
    
    if(kernel_kind.find("1D")!=-1):
        items.extend(signleDim)

    elif(kernel_kind.find("2D")!=-1):
        for i in signleDim:
            for j in signleDim:
                items.extend((i,j))
                
    elif(kernel_kind.find("3D")!=-1):
        for i in signleDim:
            for j in signleDim:
                for k in signleDim:
                    items.extend((i,j,k))

    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_stride_or_dilation_rate_pool_size(start,endg,step_size=1,startrun=0,steprun=1,endrun=sys.maxsize,enable_perum=True):
    #检查stride合法性
    items = [i for i in range(start,endg+1,step_size)]
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_padding(startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    paddings=['valid','same','causal']
    endrun=min(len(paddings)-1,endrun)
    return get_item(startrun,endrun,steprun,paddings,enable_perum)

def get_group(start,channelnum,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    """
    group的数量一定要比channel少
    """
    items = [i for i in range(start,channelnum+1,step_size)]
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_data_format(steprun=1,enable_perum=True):
    format=['channels_first','channels_last']
    return get_item(0,1,steprun,format,enable_perum)

def get_strides2D_and_dilation_rate_pool_size(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,fallbacktoone=0,enable_perum=True):
    if(fallbacktoone==1):
        return get_stride_or_dilation_rate_pool_size(start,endg,step_size,startrun,steprun,endrun,enable_perum)
    signleDim = [i for i in range(start,endg+1,step_size)]
    items=[]
    for i in signleDim:
        for j in signleDim:
            items.append((i,j))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)


def get_strides3D_and_dilation_rate_pool_size(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,fallbacktoone=0,enable_perum=True):
    if(fallbacktoone==1):
        return get_stride_or_dilation_rate_pool_size(start,endg,step_size,startrun,steprun,endrun,enable_perum)
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
        items.extend(signleDim)

    if(kernel_kind.find("2D")!=-1):
        for i in signleDim:
            for j in signleDim:
                items.extend((i,j))
                
    if(kernel_kind.find("3D")!=-1):
        for i in signleDim:
            for j in signleDim:
                for k in signleDim:
                    items.extend((i,j,k))
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
    items=signleDim
    # items=[]
    # for i in signleDim:
    #         for j in signleDim:
    #             items.append((i,j))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_size3D_and_padding(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    signleDim = [i for i in range(start,endg+1,step_size)]
    items=signleDim
    # items=[]
    # for i in signleDim:
    #         for j in signleDim:
    #             for k in signleDim:
    #                 items.append((i,j,k))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_interpolation(step_size=1,enable_perum=True):
    interpolation=['nearest','bilinear']

    return  get_item(0,1,step_size,interpolation,enable_perum)    

def get_croping1D(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    signleDim = [i for i in range(start,endg+1,step_size)]
    items=[]
    for i in signleDim:
            for j in signleDim:
                items.append((i,j))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_croping2D(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    end=endg//2+1
    assert start<=end
    items1d = [i for i in range(start,end+1,step_size)]
    
    item=[]
    for i in items1d:
            for j in items1d:
                item.append((i,j))
    items=list(product(item,repeat=2))
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)
    
def get_croping3D(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    items1d = [i for i in range(start,endg+1,step_size)]
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
    for ele in range(0, inputshape):
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

# target_shape = get_target_shape([1,2,3,4,5])
# target_shape()

def get_next_permute(dim,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    # dim = len(inputshape)
 
    def generate_all_cases(dim):
        result=[]
        len=dim
        l = [i for i in range(0,dim+1)]
        for depth in range(1,dim+1):
            result.extend([i for i in  combinations(l,depth)])
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

    def generate_all_cases(dim):
        result=[]
        len=dim
        l = [i for i in range(0,dim+1)]
        for depth in range(1,dim+1):
            result.extend([i for i in  combinations(l,depth)])
        return result
     
    all_combine=generate_all_cases(dim)
    endrun=min(len(all_combine)-1,endrun)
    return get_item(startrun,endrun,steprun,all_combine,enable_perum)

def get_filters(start=0,endg=1024,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,fixsequence=0,enable_perum=True):
    if(fixsequence==0):
        items = [i for i in range(start,endg+1,step_size)]
    elif(fixsequence==1):
        endg=min(endg,50) 
        items  = [(1<<i) for i in range(start,endg+1,step_size)]
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_epsilon(start=0,endg=1024,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,fixsequence=0,enable_perum=True):
    if(fixsequence==0):
        items = [i for i in range(start,endg+1,step_size)]
    elif(fixsequence==1):
        endg=min(endg,50) 
        items  = [math.pow(1,-1*i) for i in range(start,endg+1,step_size)]
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_units(start,endg,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,fixsequence=0,enable_perum=True):
    items = [i for i in range(start,endg+1,step_size)]
    items[0]=1
    endrun=min(len(items)-1,endrun)
    return get_item(startrun,endrun,steprun,items,enable_perum)

def get_axis(dim,step_size=1,startrun=0,endrun=sys.maxsize,steprun=1,enable_perum=True):
    assert startrun <=endrun
    signleDim=[i for i in range(0,dim+1,step_size)]
    #print(signleDim)
    items=[-1]
    for r in range(1,dim+1):
        items.extend([i for i in combinations(signleDim,r)])
    endrun=min(len(items)-1,endrun)
    return  get_item(startrun,endrun,steprun,items,enable_perum)
