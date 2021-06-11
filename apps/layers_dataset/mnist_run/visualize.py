import tvm
import tvm.relay as relay
import tvm.relay.testing as testing
from graphviz import Digraph
import tensorflow.keras as keras
path = "./mnist_02.h5"
ctx = tvm.cpu(0)
#target=tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu")
target_host = 'llvm'
layout = 'NHWC'
mod = keras.models.load_model(path,compile=True)

shape_dict = {"input_1": (1,28,28,1)}
dtype_dict = {"input": "float32"}
mod, params = relay.frontend.from_keras(mod,layout = layout,shape=shape_dict)

with tvm.transform.PassContext(opt_level=3):
    opt_model, opt_params = relay.optimize(mod, target_host, params)#return IRModule

def _traverse_expr(node, node_dict):
    if node in node_dict:
        return
    if isinstance(node, tvm.ir.op.Op):
        return 
    node_dict[node] = len(node_dict)

dot = Digraph(format='svg')
dot.attr(rankdir='BT')
dot.attr('node', shape='box')

node_dict = {}
relay.analysis.post_order_visit(opt_model['main'], lambda node: _traverse_expr(node, node_dict))
for node, node_idx in node_dict.items():
    if isinstance(node, relay.expr.Var):
        print(f'node_idx: {node_idx}, Var(name={node.name_hint}, type=Tensor[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}])')
        dot.node(str(node_idx), f'{node.name_hint}:\nTensor[{tuple(node.type_annotation.shape)}, {node.type_annotation.dtype}]')
    elif isinstance(node, relay.expr.Call):
        args = [node_dict[arg] for arg in node.args]
        print(f'node_idx: {node_idx}, Call(op_name={node.op.name}, args={args})')
        dot.node(str(node_idx), f'Call(op={node.op.name})')
        for arg in args:
            dot.edge(str(arg), str(node_idx))
    elif isinstance(node, relay.Function):
        print(f'node_idx: {node_idx}, Function(body={node_dict[node.body]})')
        dot.node(str(node_idx), f'Function')
        dot.edge(str(node_dict[node.body]), str(node_idx))
    elif isinstance(node, relay.expr.TupleGetItem):
        print(f'node_idx: {node_idx}, TupleGetItem(tuple={node_dict[node.tuple_value]}, idx={node.index})')
        dot.node(str(node_idx), f'TupleGetItem(idx={node.index})')
        dot.edge(str(node_dict[node.tuple_value]), str(node_idx))
    elif isinstance(node,relay.expr.Tuple):
        print(f'node_idx: {node_idx}, span={node.span})')
        dot.node(str(node_idx), f'Tuple')
        for child in node.fields:
            dot.edge(str(child), str(node_idx))
    elif isinstance(node,relay.expr.Constant):
        print(f'node_idx: {node_idx}, data={node.data})')
        dot.node(str(node_idx), f'Constant')
    #     dot.edge(,str(node_idx))

    # elif isinstance(node,relay.expr.ExprWithOp):
    # elif isinstance(node,relay.expr.Let):
    # elif isinstance(node,relay.expr.GlobalVar):
    # elif isinstance(node,relay.expr.If):
    # elif isinstance(node,relay.expr.TempExpr):
    # else:
        raise RuntimeError(f'Unknown node type. node_idx: {node_idx}, node: {type(node)}')

print(dot.render())
