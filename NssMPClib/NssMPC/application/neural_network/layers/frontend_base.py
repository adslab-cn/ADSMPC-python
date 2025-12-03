import torch
import numpy as np
import os
import struct
from abc import ABC, abstractmethod
from typing import List, Dict, Union, Set
from NssMPC import RingTensor

# 假设这些基础类在其他地方定义，这里为了代码跑通需要简单的占位
# 实际项目中请引用 NssMPClib 里的真实类
from NssMPC.config import DEVICE


import torch
import torch.nn as nn
import math
import numpy as np
from typing import List, Tuple, Union, Any

# 为了避免循环引用，这里假设 SytorchTensor 和 LayerGraphNode 在外部定义
# 运行时请确保从正确路径导入
# from NssMPClib.common.sytorch_module import SytorchTensor, LayerGraphNode 

class Layer:
    """
    对应 C++: template <typename T> class Layer
    所有算子的基类，处理图构建(GraphGen)与执行(Execution)的分发。
    """
    def __init__(self, name: str):
        self.name = name
        self.backend = None  # Backend<T> *backend
        self.activation = None # Tensor<T> activation
        self.node = None       # LayerGraphNode<T> *node

        # Config options
        self.do_truncation_forward = False
        self.do_pre_sign_extension = False
        self.do_post_sign_extension = False
        self.scale = 0
        self.mode = 0
        self.forward_truncation_mode = 0
        self.use_bias = True
        self.is_training_mode = False
        self.param_string = "" # 用于生成 Unique ID

        # 模拟权重存储
        self.weight = None
        self.bias = None
        self.A = None # for BN
        self.B = None # for BN

    def init_scale(self, scale: int):
        """对应 C++ initScale"""
        self.scale = scale
        self._init_scale(scale)

    def _init_scale(self, scale: int):
        """虚函数，子类覆盖"""
        pass

    def resize(self, shapes: List[Tuple]):
        """对应 C++ resize"""
        out_dims = self.get_output_dims(shapes)
        # 模拟 activation.resize(outdims)
        # 在 Python 中我们通常创建一个新的 Tensor 占位符或更新 shape
        if self.activation is None:
            # 这里需要引用 SytorchTensor，为了解耦暂用类似结构
            from NssMPClib.common.sytorch_module import SytorchTensor
            self.activation = SytorchTensor(shape=out_dims)
        else:
            self.activation.shape = out_dims
        
        self._resize(shapes)

    def _resize(self, shapes):
        pass

    def set_backend(self, backend):
        self.backend = backend

    def train(self):
        self.is_training_mode = True

    def eval(self):
        self.is_training_mode = False

    def get_weights(self):
        return self.weight

    def get_bias(self):
        return self.bias

    def get_output_dims(self, in_shapes):
        raise NotImplementedError

    def _forward(self, inputs):
        """纯虚函数，子类实现具体的 Backend 调用"""
        raise NotImplementedError

    def forward(self, inputs: List[Any]) -> Any:
        """
        对应 C++: Tensor<T>& forward(std::vector<Tensor<T> *> &a)
        核心逻辑：区分建图模式与执行模式
        """
        # 统一输入格式处理
        if not isinstance(inputs, list):
            inputs = [inputs]
        
        # ---------------------------------------------------------
        # 分支 1: 图构建模式 (Graph Generation Mode)
        # ---------------------------------------------------------
        if inputs[0].graph_gen_mode:
            # 引入依赖
            from NssMPClib.common.sytorch_module import LayerGraphNode, SytorchTensor
            
            # C++: node = new LayerGraphNode<T>();
            self.node = LayerGraphNode(layer=self)
            
            # C++: node->allNodesInExecutionOrderRef = a[0]->graphNode->allNodesInExecutionOrderRef;
            # C++: node->allNodesInExecutionOrderRef->push_back(node);
            ref_list = inputs[0].graph_node.all_nodes_in_execution_order_ref
            ref_list.append(self.node)
            self.node.all_nodes_in_execution_order_ref = ref_list

            # 建立连接
            for inp in inputs:
                parent_node = inp.graph_node
                self.node.parents.append(parent_node)
                parent_node.children.append(self.node)

            # 设置 Activation 为新的 Dummy Tensor
            self.activation = SytorchTensor()
            self.activation.graph_node = self.node
            self.activation.graph_gen_mode = True
            return self.activation

        # ---------------------------------------------------------
        # 分支 2: 执行模式 (Execution Mode)
        # ---------------------------------------------------------
        # C++: always_assert(node != nullptr);
        assert self.node is not None, f"Layer {self.name} node not initialized via graph gen"
        
        self.activation.graph_gen_mode = False
        
        # 1. Resize
        input_shapes = [inp.shape for inp in inputs]
        self.resize(input_shapes)
        
        # 2. Link Node
        self.node.curr_tensor = self.activation
        self.activation.graph_node = self.node

        # 3. Pre-Sign Extension (if needed)
        if self.do_pre_sign_extension:
            # Python 模拟：backend.signext
            pass 

        # 4. Actual Forward (调用子类实现)
        # C++ 传入的是 vector<Tensor*>
        self._forward(inputs)

        # 5. Truncation
        if self.do_truncation_forward and self.backend:
            # backend->truncateForward(...)
            self.backend.truncate_forward(self.activation, self.scale, self.forward_truncation_mode)

        # 6. Post-Sign Extension
        if self.do_post_sign_extension:
            pass

        # 7. Garbage Collection (Increment usage and free parents)
        for inp in inputs:
            if inp.graph_node:
                inp.graph_node.increment_and_gc()

        return self.activation


# ==============================================================================
# 具体 Layer 实现
# ==============================================================================

class Conv2D(Layer):
    def __init__(self, ci, co, f, padding=0, stride=1, use_bias=False):
        super().__init__("Conv2D")
        self.ci = ci
        self.co = co
        # 处理 kernel size，可能是 int 或 list/tuple
        if isinstance(f, (list, tuple)):
            self.fh, self.fw = f[0], f[1]
        else:
            self.fh = self.fw = f
        
        self.padding = padding
        self.stride = stride
        self.use_bias = use_bias
        self.do_truncation_forward = True
        
        # 初始化权重张量占位符
        self.filter = torch.zeros(co, ci, self.fh, self.fw) # PyTorch 格式
        self.bias = torch.zeros(co)
        self.inp = None # 缓存 Training input

    def _init_scale(self, scale):
        # Xavier 初始化模拟 (仅作参考，实际权重由 load 覆盖)
        pass 

    def _resize(self, shapes):
        assert shapes[0][1] == self.ci, f"Input channels mismatch: {shapes[0][1]} vs {self.ci}"
        # PyTorch format: N, C, H, W. C++ Sytorch often uses N, H, W, C.
        # 假设这里跟随 Sytorch 惯例 (Last dim is channel?) -> C++ Check: assert(shape[3] == ci)
        # 对，Sytorch 是 NHWC。
        if self.is_training_mode:
            # self.inp.resize(...)
            pass

    def _forward(self, inputs):
        inp = inputs[0]
        # 调用 Backend 执行 Conv2D
        # backend->conv2D(...)
        if self.backend:
            self.backend.conv2d(inp, self.filter, self.bias, self.activation, 
                                self.stride, self.padding)

    def get_output_dims(self, in_shapes):
        # NHWC 格式计算
        n, h, w, c = in_shapes[0]
        new_h = (h + 2*self.padding - self.fh) // self.stride + 1
        new_w = (w + 2*self.padding - self.fw) // self.stride + 1
        return (n, new_h, new_w, self.co)


class FC(Layer):
    """Fully Connected Layer"""
    def __init__(self, in_dim, out_dim, use_bias=False):
        super().__init__("FC")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.do_truncation_forward = True
        
        # 权重占位 (in, out) 对应 C++ Tensor2D<T> weight(in, out)
        self.weight = torch.zeros(in_dim, out_dim)
        self.bias = torch.zeros(out_dim)

    def _resize(self, shapes):
        assert shapes[0][1] == self.in_dim

    def _forward(self, inputs):
        inp = inputs[0]
        # backend->matmul(inp, weight, activation)
        if self.backend:
            # 1. MatMul
            self.backend.matmul(inp, self.weight, self.activation)
            # 2. Add Bias
            if self.use_bias:
                self.backend.add_bias(self.activation, self.bias)

    def get_output_dims(self, in_shapes):
        return (in_shapes[0][0], self.out_dim)


class BatchNormInference(Layer):
    """
    特殊层：推理时使用融合后的参数 A 和 B
    """
    def __init__(self, channels):
        super().__init__("BatchNormInference")
        self.A = torch.zeros(channels) # Scale
        self.B = torch.zeros(channels) # Shift
        self.do_truncation_forward = True

    def _resize(self, shapes):
        # NHWC: 最后一维必须等于 channels
        assert shapes[0][-1] == self.A.shape[0]

    def _forward(self, inputs):
        if self.is_training_mode:
            raise RuntimeError("BatchNormInference not for training")
        
        if self.backend:
            # backend->batchNormInference(A, B, x, y, scale)
            self.backend.batchnorm_inference(self.A, self.B, inputs[0], self.activation, self.scale)

    def get_output_dims(self, in_shapes):
        return in_shapes[0] # Shape 不变


class Add(Layer):
    def __init__(self):
        super().__init__("Add")

    def _resize(self, shapes):
        # 检查所有输入 shape 是否一致
        base_shape = shapes[0]
        for s in shapes[1:]:
            assert s == base_shape

    def _forward(self, inputs):
        # backend->add(inputs, activation)
        if self.backend:
            self.backend.add(inputs, self.activation)

    def get_output_dims(self, in_shapes):
        return in_shapes[0]


class Concat(Layer):
    def __init__(self):
        super().__init__("Concat")

    def _forward(self, inputs):
        if self.backend:
            self.backend.concat(inputs, self.activation)

    def get_output_dims(self, in_shapes):
        # NHWC: Concat 沿着最后一维
        base_shape = list(in_shapes[0])
        total_channels = base_shape[-1]
        
        for s in in_shapes[1:]:
            total_channels += s[-1]
            
        base_shape[-1] = total_channels
        return tuple(base_shape)


class GeLU(Layer):
    def __init__(self):
        super().__init__("GeLU")

    def _forward(self, inputs):
        if self.backend:
            self.backend.gelu(inputs[0], self.activation, self.scale)

    def get_output_dims(self, in_shapes):
        return in_shapes[0]


class SoftMax(Layer):
    def __init__(self):
        super().__init__("SoftMax")

    def _forward(self, inputs):
        if self.backend:
            self.backend.softmax(inputs[0], self.activation, self.scale)

    def get_output_dims(self, in_shapes):
        return in_shapes[0]


class Flatten(Layer):
    def __init__(self):
        super().__init__("Flatten")

    def _forward(self, inputs):
        # 逻辑：将多维数据展平到 2D (Batch, Features)
        # C++ 代码手动搬运了数据，Python 模拟可以使用 reshape/flatten
        inp = inputs[0]
        # 这里需要注意：SytorchTensor 只是 wrapper，改变它的 shape 即可，或者数据拷贝
        # 假设 activation 已经 resize 好了，直接 copy 数据
        if self.backend:
            # 简单的内存拷贝或 reshape
            # self.activation.tensor = inp.tensor.reshape(self.activation.shape)
            # 或者调用 backend 的 flatten
            pass 

    def get_output_dims(self, in_shapes):
        shape = in_shapes[0]
        # 假设保留 Batch 维度 (dim 0)，展平后面所有
        prod = 1
        for d in shape[1:]:
            prod *= d
        return (shape[0], prod)


class Split(Layer):
    """
    对应 C++ Split: 将 tensor 的最后一维 reshape 为 (n_splits, rest)
    注意：C++代码显示它其实是把最后一维拆分并插入一个新维度，而不是拆成多个 Tensor
    """
    def __init__(self, n_splits):
        super().__init__("Split")
        self.n_splits = n_splits

    def _resize(self, shapes):
        assert shapes[0][-1] % self.n_splits == 0

    def _forward(self, inputs):
        # 逻辑：Reshape
        # C++: [..., dim] -> [..., n_splits, dim/n_splits]
        # 实际数据重排
        pass 

    def get_output_dims(self, in_shapes):
        shape = list(in_shapes[0])
        old_dim = shape[-1]
        shape[-1] = old_dim // self.n_splits
        # 插入 n_splits 到倒数第二维
        shape.insert(len(shape)-1, self.n_splits)
        return tuple(shape)


class View(Layer):
    """
    对应 C++ View(idx): 这是一个切片/选择操作，而不是 PyTorch 的 view
    取出 Input[idx] (沿 dim 0)
    """
    def __init__(self, idx):
        super().__init__("View")
        self.idx = idx

    def _forward(self, inputs):
        # 取出 inputs[0] 的第 idx 个元素
        # C++: a.view(i)
        pass

    def get_output_dims(self, in_shapes):
        shape = list(in_shapes[0])
        # 去掉 dim 0
        return tuple(shape[1:])


class Transpose(Layer):
    def __init__(self, perm):
        super().__init__("Transpose")
        self.perm = perm # e.g., [0, 2, 1, 3]

    def _forward(self, inputs):
        if self.backend:
            # backend->transpose(...)
            pass

    def get_output_dims(self, in_shapes):
        shape = in_shapes[0]
        out_shape = [shape[p] for p in self.perm]
        return tuple(out_shape)


class _MatMul(Layer):
    """内部使用的 MatMul Layer"""
    def __init__(self):
        super().__init__("_MatMul")
        self.do_truncation_forward = True

    def _forward(self, inputs):
        # inputs[0] @ inputs[1]
        if self.backend:
            self.backend.matmul(inputs[0], inputs[1], self.activation)

    def get_output_dims(self, in_shapes):
        # (M, K) @ (K, N) -> (M, N)
        return (in_shapes[0][0], in_shapes[1][1])


class _ScalarMul(Layer):
    def __init__(self, scalar):
        super().__init__("_ScalarMul")
        self.scalar = scalar
        self.do_truncation_forward = True

    def _forward(self, inputs):
        if self.backend:
            # Fix scalar to int
            scalar_fix = int(self.scalar * (1 << self.scale))
            self.backend.scalarmul(inputs[0], scalar_fix, self.activation)

    def get_output_dims(self, in_shapes):
        return in_shapes[0]
    
    
class Backend:
    """对应 C++ Backend<T>"""
    def optimize(self, root):
        pass

class ClearText(Backend):
    """对应 C++ ClearText<T>"""
    pass


# 假设 Layer 和 Tensor 的基类定义在这里或被导入
# 为了避免循环引用，这里使用 Type Hint 的字符串形式

class LayerGraphNode:
    """
    对应 C++: struct LayerGraphNode<T>
    """
    def __init__(self, layer=None):
        self.layer = layer
        self.parents = []   # List[LayerGraphNode]
        self.children = []  # List[LayerGraphNode]
        self.num_usages = 0
        self.curr_tensor = None  # SytorchTensor
        self.mark = False
        
        # 对应 C++: std::vector<LayerGraphNode<T> *> *allNodesInExecutionOrderRef
        # 在 Python 中直接引用列表对象即可
        self.all_nodes_in_execution_order_ref = None 

    def increment_and_gc(self) -> bool:
        """
        对应 C++: bool incrementAndGc()
        用于引用计数和垃圾回收模拟
        """
        if self.layer.name == "Input":
            return False
        
        self.num_usages += 1
        
        # 当使用次数等于子节点数量时，说明该节点的数据已被所有下游消费完毕
        if self.num_usages == len(self.children):
            if self.curr_tensor is not None:
                # 对应 C++: currTensor->free();
                # 在 Python 中通常依靠 GC，但为了模拟 MPC 内存管理逻辑，我们可以显式清理数据
                if hasattr(self.curr_tensor, 'free'):
                    self.curr_tensor.free()
                else:
                    self.curr_tensor.data = None # 模拟释放显存/内存
            return True
            
        return False


def topological_visit(visited: set, node: LayerGraphNode, root: LayerGraphNode, visit_fn):
    """
    对应 C++: void topologicalVisit(...)
    注意 C++ 的逻辑：先递归 Parents，再访问 Self，再递归 Children。
    这是一种混合遍历方式，确保连通图的覆盖。
    """
    if node in visited:
        return
    
    visited.add(node)
    
    # 1. 递归访问父节点
    for parent in node.parents:
        topological_visit(visited, parent, root, visit_fn)
    
    # 2. 访问当前节点 (Apply Function)
    visit_fn(node, root)
    
    # 3. 递归访问子节点
    for child in node.children:
        topological_visit(visited, child, root, visit_fn)


def topological_apply(root: LayerGraphNode, visit_fn):
    """
    对应 C++: void topologicalApply(...)
    """
    visited = set()
    topological_visit(visited, root, root, visit_fn)


def print_dot_graph(root: LayerGraphNode, filename="graph.dot"):
    """
    对应 C++: void print_dot_graph(...)
    生成 Graphviz 格式的计算图文件，用于可视化调试。
    """
    try:
        with open(filename, "w") as dotfile:
            dotfile.write("digraph G {\n")

            def _write_node(node: LayerGraphNode, _root):
                if node.layer is not None:
                    # 构造 Label
                    # C++: label = layer->name
                    label = node.layer.name
                    
                    # 处理 paramstring (例如把 | 换成 ,)
                    # C++: if (node->layer->paramstring != "") ...
                    if hasattr(node.layer, 'param_string') and node.layer.param_string:
                        args = node.layer.param_string.replace('|', ',')
                        if args.endswith(','):
                            args = args[:-1]
                        label += f"({args})"
                    
                    # 构造节点属性
                    # C++ 使用 (uint64_t)(node->layer) 作为 ID，Python 使用 id(node.layer)
                    node_id = str(id(node.layer))
                    color_attr = ' color="red"' if node.mark else ""
                    
                    # 写入节点定义
                    dotfile.write(f'  "{node.layer.name}{node_id}" [label="{label}"{color_attr}];\n')
                    
                    # 写入边 (Children)
                    for child in node.children:
                        child_id = str(id(child.layer))
                        dotfile.write(f'  "{node.layer.name}{node_id}" -> "{child.layer.name}{child_id}";\n')

            # 执行遍历并写入
            topological_apply(root, _write_node)

            dotfile.write("}\n")
            
    except IOError as e:
        print(f"Error writing dot graph: {e}")


class SytorchTensor:
    """对应 C++ Tensor<T>"""
    def __init__(self, tensor=None, shape=None):
        self.tensor = tensor # 实际数据 (RingTensor 或 torch.Tensor)
        self.shape = shape if shape is not None else (tensor.shape if tensor is not None else None)
        self.graph_gen_mode = False
        self.graph_node: LayerGraphNode = None

    def resize(self, shape):
        self.shape = shape

    def copy(self, other: 'SytorchTensor'):
        self.tensor = other.tensor
        self.shape = other.shape
    def free(self):
        """模拟释放内存"""
        self.tensor = None

# =============================================================================
# SytorchModule 主类
# =============================================================================

class SytorchModule(ABC):
    # 对应 C++: static std::map<std::string, LayerGraphNode<T> *> functionalLayerMap;
    # 这是一个静态成员变量
    functional_layer_map: Dict[str, LayerGraphNode] = {}

    # 对应 C++: const std::vector<std::string> functionalLayers
    functional_layers_names = {
        "Add", "Concat", "GeLU", "SoftMax", "Split", 
        "View", "Transpose", "_MatMul", "_ScalarMul"
    }

    def __init__(self):
        self.activation = SytorchTensor()
        self.backend = ClearText()
        self.root: LayerGraphNode = None
        self.debug = True
        self.scale = 0
        
        # 对应 C++: std::vector<LayerGraphNode<T> *> allNodesInExecutionOrder;
        self.all_nodes_in_execution_order: List[LayerGraphNode] = []

    @abstractmethod
    def _forward(self, input_tensor: SytorchTensor) -> SytorchTensor:
        """纯虚函数，子类必须实现"""
        pass

    # =========================================================================
    # 核心图构建与管理逻辑
    # =========================================================================

    def topological_apply(self, root: LayerGraphNode, func):
        """
        对应 C++ topologicalApply 辅助函数
        执行简单的图遍历（DFS/BFS）
        """
        if root is None:
            return
        
        visited = set()
        stack = [root] # 或者根据具体的拓扑排序逻辑实现
        
        # 这里模拟 C++ 代码的行为，遍历所有可达节点
        # 注意：C++ 的 topologicalApply 通常保证父节点在子节点前被访问
        # 如果 allNodesInExecutionOrder 已经生成，直接遍历它可能更准，
        # 但这里是对任意 root 的遍历。
        
        def dfs(node:LayerGraphNode):
            if node in visited:
                return
            visited.add(node)
            func(node, root)
            for child in node.children:
                dfs(child)
        
        dfs(root)

    def generate_functional_layer_map(self):
        """对应 C++ generateFunctionalLayerMap"""
        # C++ 中注释掉了 clear()，我们也不 clear，意味着跨模块可能共享 map
        
        def _visit(node: LayerGraphNode, _root):
            if node.layer.name in self.functional_layers_names:
                # 生成唯一 ID
                # id = layerName + "|" + parent1_ptr + "|" + parent2_ptr... + "|" + paramstring
                node_id = node.layer.name
                for parent in node.parents:
                    # C++ 使用指针地址 (uint64_t)(parent)
                    # Python 使用 id(parent)
                    node_id += "|" + str(id(parent))
                node_id += "|" + node.layer.paramstring
                
                # assert not exists
                if node_id in self.functional_layer_map:
                    raise RuntimeError(f"Layer already exists in map: {node_id}")
                
                self.functional_layer_map[node_id] = node

        self.topological_apply(self.root, _visit)

    def get_functional_node(self, layer_name: str, ips: List[SytorchTensor], *args) -> LayerGraphNode:
        """对应 C++ getFunctionalNode"""
        node_id = layer_name
        for ip in ips:
            # 同样使用 graph_node 的内存地址作为 ID
            node_id += "|" + str(id(ip.graph_node))
        
        # 参数转字符串 (paramstring)
        # 假设 args 的 string 转换与 C++ 一致
        param_str = str(args) if len(args) == 1 else str(args).replace(" ", "") # 需根据 Layer 实现调整
        # 这里为了简单，假设 Layer 层有个 helper 能够生成和 C++ 一样的 paramstring
        # 暂时用简单的 str(args) 模拟
        # node_id += "|" + param_str 
        # C++ 代码中是 paramstring(args...)，这里简化处理，实际要看 Layer 怎么存
        # 假设 args 已经包含了 params，或者由调用者保证一致性
        # 我们这里假设 args 的拼接逻辑由各 functional 函数内部处理好传进来可能更合适，
        # 但为了还原 C++ 签名：
        
        # Python 变通：我们在 functionalGraphGen 里保存了 paramstring，这里需要重新生成一遍
        # 假设 args 里的元素能直接转 string
        p_str = ""
        for arg in args:
             p_str += str(arg) # 极简模拟
        
        # 注意：C++ 的 paramstring 是模板函数的特化，这里是一个难点。
        # 我们假设调用 getFunctionalNode 前，上层已经确信 ID 生成逻辑是匹配的。
        # 为了严谨，应该去 functional_layer_map 的 key 里找匹配。
        
        # 修正逻辑：由于 Python 动态参数不好完全对齐 C++ 模板展开，
        # 我们在下面的 add/matmul 等具体函数里手动拼接 paramstring 会更稳。
        # 这里暂时用一个占位符，下文具体函数会覆盖。
        
        # 重新看 C++: id = id + "|" + paramstring(args...);
        pass # 具体实现在各个 op 函数里
        
        return self.functional_layer_map.get(node_id) # 暂未实现完整 ID 拼接

    def functional_graph_gen(self, LayerType, inputs: List[SytorchTensor], *args) -> SytorchTensor:
        """对应 C++ functionalGraphGen"""
        for a in inputs:
            assert a.graph_gen_mode, "Input must be in graphGenMode"
        
        layer = LayerType(*args)
        # C++: layer->paramstring = paramstring(args...);
        layer.paramstring = str(args) # 简易模拟
        
        return layer.forward(inputs)

    def gen_graph_and_execution_order(self):
        """对应 C++ genGraphAndExecutionOrder"""
        ip = SytorchTensor()
        ip.graph_gen_mode = True
        ip.graph_node = LayerGraphNode()
        
        # 这里需要引入 PlaceHolderLayer
        # ip.graph_node.layer = PlaceHolderLayer("Input") 
        # 假设有个占位层对象
        class PlaceHolderLayer:
            def __init__(self, name): self.name = name; self.paramstring = ""
            def init_scale(self, s): pass
            
        ip.graph_node.layer = PlaceHolderLayer("Input")
        
        # 这里的引用传递是个坑，Python 列表是引用的，所以直接赋值即可
        # C++: ip.graphNode->allNodesInExecutionOrderRef = &allNodesInExecutionOrder;
        # 我们给 graph_node 绑一个 list 引用
        ip.graph_node.all_nodes_in_execution_order_ref = self.all_nodes_in_execution_order
        
        # 清空列表，准备录制
        self.all_nodes_in_execution_order.clear()
        self.all_nodes_in_execution_order.append(ip.graph_node) # Input 是第一个
        
        # 执行追踪
        res = self._forward(ip)
        
        ip.graph_gen_mode = False
        self.root = ip.graph_node

    def init(self, scale: int):
        """对应 C++ init"""
        self.gen_graph_and_execution_order()
        
        def _init_scale(node, _root):
            node.layer.init_scale(scale)
            
        self.topological_apply(self.root, _init_scale)
        
        self.scale = scale
        self.generate_functional_layer_map()

    def zero(self):
        """对应 C++ zero"""
        def _zero(node, _root):
            if hasattr(node.layer, 'get_weights'):
                node.layer.get_weights().zero_() # PyTorch in-place zero
            if hasattr(node.layer, 'get_bias'):
                node.layer.get_bias().zero_()
        
        self.topological_apply(self.root, _zero)

    def set_backend(self, b: Backend):
        """对应 C++ setBackend"""
        def _set(node, _root):
            node.layer.set_backend(b)
        
        self.topological_apply(self.root, _set)
        self.backend = b

    def forward(self, input_tensor: SytorchTensor) -> SytorchTensor:
        """对应 C++ forward"""
        if input_tensor.graph_gen_mode:
            return self._forward(input_tensor)
        
        if input_tensor.graph_node is None:
            # 顶层模块调用
            def _reset_usage(node, _root):
                node.num_usages = 0
            self.topological_apply(self.root, _reset_usage)
            
            input_tensor.graph_node = self.root
            input_tensor.graph_node.curr_tensor = input_tensor
        
        if self.debug:
            res = self._forward(input_tensor)
            self.activation.resize(res.shape)
            self.activation.copy(res)
            return self.activation
        else:
            # TODO: calculate using generated graph (C++ comments)
            res = self._forward(input_tensor)
            self.activation.resize(res.shape)
            self.activation.copy(res)
            return self.activation

    def optimize(self):
        """对应 C++ optimize"""
        self.backend.optimize(self.root)

    def train(self):
        """对应 C++ train"""
        def _train(node, _root):
            node.layer.train() # 假设 Layer 有 train 方法
        self.topological_apply(self.root, _train)

    def eval(self):
        """对应 C++ eval"""
        def _eval(node, _root):
            node.layer.eval() # 假设 Layer 有 eval 方法
        self.topological_apply(self.root, _eval)

    # =========================================================================
    # 权重加载与 Dump (核心难点)
    # =========================================================================

    def load(self, weights_file: str):
        """对应 C++ load"""
        if not os.path.exists(weights_file):
            raise FileNotFoundError(f"Weights file not found: {weights_file}")

        file_size = os.path.getsize(weights_file)
        assert file_size % 4 == 0, "File size must be multiple of 4 (float32)"
        num_parameters = file_size // 4
        
        print(f"Model Weights Size: {file_size} bytes")
        
        # 使用 numpy 读取 float32，效率接近 mmap
        float_weights = np.fromfile(weights_file, dtype=np.float32)
        
        scale = self.scale
        w_idx = 0
        
        # 遍历执行顺序 (Execution Order)
        for node in self.all_nodes_in_execution_order:
            layer = node.layer
            layer_name = layer.name # 假设 Layer 有 name 属性
            
            # --- BatchNormInference 融合逻辑 ---
            if layer_name == "BatchNormInference":
                # 假设 bn 层有属性 A 和 B (对应 C++ bn->A, bn->B)
                # 且 A, B 是 Tensor
                bn = layer 
                channel = bn.A.shape[0] # d1
                
                # 指针偏移模拟
                gamma = float_weights[w_idx : w_idx + channel]
                beta  = float_weights[w_idx + channel : w_idx + 2*channel]
                mean  = float_weights[w_idx + 2*channel : w_idx + 3*channel]
                var   = float_weights[w_idx + 3*channel : w_idx + 4*channel]
                
                # C++ Logic Reconstruction:
                # A(j) = (gamma / sqrt(var)) * (1 << scale)
                # B(j) = (beta - gamma * mean / sqrt(var)) * (1 << (2*scale))
                
                sqrt_var = np.sqrt(var)
                
                # 计算 A (Gamma part)
                val_a = (gamma / sqrt_var) * (1 << scale)
                # 类型转换 (type_cast<T>) -> int64
                bn.A.data.copy_(torch.from_numpy(val_a.astype(np.int64)))
                
                # 计算 B (Beta part)
                val_b = (beta - (gamma * mean) / sqrt_var) * (1 << (2 * scale))
                bn.B.data.copy_(torch.from_numpy(val_b.astype(np.int64)))
                
                w_idx += 4 * channel
            
            # --- 普通层逻辑 ---
            else:
                # 处理 Weights
                if hasattr(layer, 'get_weights'):
                    weights = layer.get_weights()
                    if weights is not None:
                        size = weights.numel()
                        # data = float * (1 << scale)
                        chunk = float_weights[w_idx : w_idx + size]
                        scaled = chunk * (1 << scale)
                        weights.data.copy_(torch.from_numpy(scaled.astype(np.int64)).view_as(weights))
                        w_idx += size
                
                # 处理 Bias
                if hasattr(layer, 'get_bias'):
                    bias = layer.get_bias()
                    if layer.use_bias and bias is not None:
                        size = bias.numel()
                        # Bias scale 通常是 2*scale (因为 input*weight 已经是 2*scale 了)
                        chunk = float_weights[w_idx : w_idx + size]
                        scaled = chunk * (1 << (2 * scale)) # 注意这里是 2*scale
                        bias.data.copy_(torch.from_numpy(scaled.astype(np.int64)).view_as(bias))
                        w_idx += size
                    elif bias is not None:
                        bias.zero_()

        assert w_idx == num_parameters, f"Weights mismatch: Read {w_idx}, Expected {num_parameters}"

    def dumpi64(self, weights_file: str):
        """对应 C++ dumpi64"""
        scale = self.scale
        
        with open(weights_file, 'wb') as f:
            for node in self.all_nodes_in_execution_order:
                layer = node.layer
                layer_name = layer.name
                
                if layer_name == "BatchNormInference":
                    bn = layer
                    channel = bn.A.shape[0]
                    
                    # 写入 A
                    f.write(bn.A.cpu().numpy().astype(np.int64).tobytes())
                    # 写入 B
                    f.write(bn.B.cpu().numpy().astype(np.int64).tobytes())
                    # 写入 0 (mean占位)
                    zeros = np.zeros(channel, dtype=np.int64)
                    f.write(zeros.tobytes())
                    # 写入 1<<scale (var占位)
                    ones_scaled = np.full(channel, 1 << scale, dtype=np.int64)
                    f.write(ones_scaled.tobytes())
                    
                else:
                    # 普通权重
                    if hasattr(layer, 'get_weights') and layer.get_weights() is not None:
                        w = layer.get_weights()
                        f.write(w.cpu().numpy().astype(np.int64).tobytes())
                    
                    # Bias
                    if hasattr(layer, 'get_bias') and layer.get_bias() is not None:
                        if layer.use_bias:
                            b = layer.get_bias()
                            f.write(b.cpu().numpy().astype(np.int64).tobytes())

    # =========================================================================
    # 功能层 Wrappers (Functional Helpers)
    # =========================================================================
    
    # 辅助：获取 ID 的通用逻辑
    def _get_node_by_id(self, name, tensors, *args):
        node_id = name
        for t in tensors:
            node_id += "|" + str(id(t.graph_node))
        # 拼接 args 参数字符串
        # 注意：这里需要和 functionalGraphGen 里保存的 paramstring 一致
        # 简易实现：直接拼 args tuple
        node_id += "|" + str(args)
        
        if node_id not in self.functional_layer_map:
            raise RuntimeError(f"Layer not found: {node_id}")
        return self.functional_layer_map[node_id]

    def add(self, *args: Union[SytorchTensor, List[SytorchTensor]]) -> SytorchTensor:
        # 处理输入：可能是 vector<Tensor*> (list) 或者可变参数
        if isinstance(args[0], list):
            inputs = args[0]
        else:
            inputs = list(args)

        if inputs[0].graph_gen_mode:
            # 需要引用 Add 层类
            from NssMPC.application.neural_network.layers.sytorch_layers import Add 
            return self.functional_graph_gen(Add, inputs) # Add 无额外参数
        
        node = self._get_node_by_id("Add", inputs, ()) # 空 tuple 对应无参数
        return node.layer.forward(inputs)

    def concat(self, *args) -> SytorchTensor:
        if isinstance(args[0], list):
            inputs = args[0]
        else:
            inputs = list(args)

        if inputs[0].graph_gen_mode:
            from NssMPC.application.neural_network.layers.sytorch_layers import Concat
            return self.functional_graph_gen(Concat, inputs)
        
        node = self._get_node_by_id("Concat", inputs, ())
        return node.layer.forward(inputs)

    def gelu(self, a: SytorchTensor) -> SytorchTensor:
        inputs = [a]
        if a.graph_gen_mode:
            from NssMPC.application.neural_network.layers.sytorch_layers import GeLU
            return self.functional_graph_gen(GeLU, inputs)
        
        node = self._get_node_by_id("GeLU", inputs, ())
        return node.layer.forward(inputs)

    def softmax(self, a: SytorchTensor) -> SytorchTensor:
        inputs = [a]
        if a.graph_gen_mode:
            from NssMPC.application.neural_network.layers.sytorch_layers import SoftMax
            return self.functional_graph_gen(SoftMax, inputs)
        
        node = self._get_node_by_id("SoftMax", inputs, ())
        return node.layer.forward(inputs)

    def split(self, a: SytorchTensor, n_splits: int) -> SytorchTensor:
        inputs = [a]
        if a.graph_gen_mode:
            from NssMPC.application.neural_network.layers.sytorch_layers import Split
            return self.functional_graph_gen(Split, inputs, n_splits)
        
        node = self._get_node_by_id("Split", inputs, (n_splits,))
        return node.layer.forward(inputs) # Split forward 可能返回 list，需注意 Tensor 定义

    def view(self, a: SytorchTensor, idx: int) -> SytorchTensor:
        inputs = [a]
        if a.graph_gen_mode:
            from NssMPC.application.neural_network.layers.sytorch_layers import View
            return self.functional_graph_gen(View, inputs, idx)
        
        node = self._get_node_by_id("View", inputs, (idx,))
        return node.layer.forward(inputs)

    def transpose(self, a: SytorchTensor) -> SytorchTensor:
        inputs = [a]
        if a.graph_gen_mode:
            from NssMPC.application.neural_network.layers.sytorch_layers import Transpose
            return self.functional_graph_gen(Transpose, inputs)
        
        node = self._get_node_by_id("Transpose", inputs, ())
        return node.layer.forward(inputs)

    def matmul(self, a: SytorchTensor, b: SytorchTensor) -> SytorchTensor:
        inputs = [a, b]
        if a.graph_gen_mode:
            from NssMPC.application.neural_network.layers.sytorch_layers import MatMul
            return self.functional_graph_gen(MatMul, inputs)
        
        node = self._get_node_by_id("_MatMul", inputs, ())
        return node.layer.forward(inputs)

    def scalarmul(self, a: SytorchTensor, scalar: float) -> SytorchTensor:
        inputs = [a]
        if a.graph_gen_mode:
            from NssMPC.application.neural_network.layers.sytorch_layers import ScalarMul
            return self.functional_graph_gen(ScalarMul, inputs, scalar)
        
        node = self._get_node_by_id("_ScalarMul", inputs, (scalar,))
        return node.layer.forward(inputs)

    def invsqrt(self, x: float) -> int:
        """对应 C++ T invsqrt(double x)"""
        t = 1.0 / np.sqrt(x)
        # 对应: T(t * (1LL << scale))
        return int(t * (1 << self.scale))