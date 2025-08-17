# utils/plan_and_generate_keys.py
import os
import torch
import pickle
import argparse # 用于解析命令行参数
from NssMPC.config import param_path

# 导入 PyTorch 模型 (以 transformers 为例)
from transformers import AutoModelForSequenceClassification, AutoConfig

# 导入您的 CrypTen/SHAFT 代码
import crypten
# 确保 crypten.__file__ 指向的是您修改过的版本
print("正在使用的 crypten 模块位于:", crypten.__file__)

from crypten.config import cfg
from crypten.nn.module import Embedding, LayerNormalization, MatMul, ReLUFastSecNet, GELU, SiLU, Softmax # 导入您的自定义模块
from crypten.protocols import FastSecNetReLUKey # 导入您的密钥定义
from crypten.nn.onnx_converter import _from_pytorch_to_bytes, _load_onnx_model
from crypten.protocols import FastSecNetReLUKey 

# 导入主密钥对象内部包含的所有自定义属性的类
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.dcf_key import DCFKey
from NssMPC.common.ring.ring_tensor import RingTensor
from NssMPC.crypto.primitives.arithmetic_secret_sharing import ArithmeticSecretSharing
# ... 以及 DCFKey 内部可能包含的任何其他自定义类，比如 CW, CWList ...
from NssMPC.crypto.aux_parameter.function_secret_sharing_keys.cw import CW, CWList


def plan_and_generate_keys_with_hooks(pytorch_model, dummy_input, num_inferences=1):
    print("--- Starting Dummy Run using Hooks for Resource Planning ---")

    # --- 1. 将 PyTorch 模型转换为 CrypTen Graph ---
    #    这是必需的，因为我们想在 CrypTen 的模块 (如 ReLU_FSS) 上挂钩
    print("[1/3] Converting PyTorch model to CrypTen graph...")
    # 假设 from_pytorch 已经修改，能将 ReLU/GeLU 转换为 ReLUFastSecNet/GELU 等
    crypten_model = crypten.nn.from_pytorch(pytorch_model, dummy_input)
    print("    ...Conversion complete.")

    # --- 2. 定义钩子并执行“空运行”来捕获信息 ---
    print("[2/3] Registering hooks and performing dry run to capture manifest...")
    
    manifest = []
    handles = []
    
    def capture_hook(module, input_tensors, output_tensor):
        if isinstance(module, (ReLUFastSecNet, GELU, SiLU, Softmax)):
            input_shape = tuple(input_tensors[0].shape)
            num_keys = input_tensors[0].numel()
            node_name = module.node_name

            manifest.append({
                "layer_name": node_name, 
                "op_type": FastSecNetReLUKey, 
                "shape": input_shape,
                "count": num_keys
            })
    def set_shape_inference_mode_hook(module, input_tensors):
        return input_tensors, {'mode': 'shape_inference'}
    for name, module_instance in crypten_model.named_modules():
        if isinstance(module_instance, (ReLUFastSecNet, GELU, SiLU, Softmax)):
            hook_with_name = lambda m, i, o, n=name: capture_hook_with_name(m, i, o, n)
            handle = module_instance.register_forward_hook(hook_with_name)
            handles.append(handle)
            handle = module_instance.register_forward_pre_hook(set_shape_inference_mode_hook)
            handles.append(handle)
        if isinstance(module_instance, (MatMul,Embedding,LayerNormalization)):
            handle = module_instance.register_forward_pre_hook(set_shape_inference_mode_hook)
            handles.append(handle)

    # 修改钩子函数以接收名字
    def capture_hook_with_name(module, input_tensors, output_tensor, node_name):
        input_shape = tuple(input_tensors[0].shape)
        num_keys = input_tensors[0].numel()
        manifest.append({
            "layer_name": node_name, 
            "op_type": FastSecNetReLUKey,
            "shape": input_shape,
            "count": num_keys
        })

    crypten_model.encrypt()
    with torch.no_grad():
        crypten_model.eval().cpu()
        
        if isinstance(dummy_input, (list, tuple)):
            crypten_dummy_input = [crypten.cryptensor(t.cpu()) for t in dummy_input]
            crypten_model(*crypten_dummy_input)
        else:
            crypten_dummy_input = crypten.cryptensor(dummy_input.cpu())
            crypten_model(crypten_dummy_input)

    for handle in handles:
        handle.remove()
    del pytorch_model
    del crypten_model 
    torch.cuda.empty_cache() 
    print(f"    ...Manifest created with {len(manifest)} entries.")
    print("[3/3] Generating and saving keys based on the manifest...")
    keys = {}
    for item in manifest:
        layer_name = item['layer_name']
        count = item['count'] * num_inferences
        if "Softmax" in layer_name:
            count *= 2
        keys[layer_name] = FastSecNetReLUKey.gen(count,device="cpu")
    if len(manifest) > 0:
        data_path = f"{param_path}{FastSecNetReLUKey.__name__}/"
        os.makedirs(data_path, exist_ok=True)
        saved_name = f"relu_{FastSecNetReLUKey.__name__}"
        key0_path = os.path.join(data_path, f"{saved_name}_0.pkl")
        key1_path = os.path.join(data_path, f"{saved_name}_1.pkl")
        keys_0 = {item: keys[item][0] for item in keys}
        keys_1 = {item: keys[item][1] for item in keys}
        with open(key0_path, 'wb') as f:
            pickle.dump(keys_0, f)
        with open(key1_path, 'wb') as f:
            pickle.dump(keys_1, f)
        manifest_path = os.path.join(data_path, f"{saved_name}_manifest.pkl")
        with open(manifest_path, 'wb') as f:
            pickle.dump(manifest, f)
        print(f"    ...{len(keys)} {FastSecNetReLUKey.__name__} and manifest saved to '{data_path}'.")
    else:
        print("    ...No layers requiring FSS keys were found.")
    print("\n--- Dummy Run and Key Generation Finished Successfully! ---")

def main():
    # ... (加载 pytorch_model, 创建 dummy_input) ...

    # 1. 调用新的规划函数
    manifest = plan_and_generate_keys_with_hooks(pytorch_model, dummy_input)
    
    # 2. 根据 manifest 生成密钥
    # ... (这部分逻辑和之前一样，遍历 manifest，计算总数，调用 gen_and_save) ...
def _topological_sort(graph_dict, start_nodes):
    """
    一个健壮的拓扑排序实现，专门用于处理 CrypTen 的计算图。

    Args:
        graph_dict (dict): CrypTen 模型内部的 `_graph` 字典。
                           格式为 {'output_name': ['input_name_1', ...]}
        start_nodes (list): 模型的初始输入节点名称列表 (例如 ['input', 'attention_mask'])

    Returns:
        list: 一个包含所有计算节点名称的、按拓扑顺序排列的列表。
    """
    
    # --- 步骤 1: 构建适合 Kahn 算法的数据结构 ---
    # 我们需要一个正向邻接表 (adj) 和一个入度表 (in_degree)

    # 1a. 收集图中所有出现过的节点名称
    all_nodes = set(start_nodes)
    for node_name, input_names in graph_dict.items():
        all_nodes.add(node_name)
        for input_name in input_names:
            all_nodes.add(input_name)
            
    # 1b. 初始化邻接表和入度表
    adj = {node: [] for node in all_nodes}
    in_degree = {node: 0 for node in all_nodes}

    # 1c. 遍历 CrypTen 的反向图 (graph_dict)，填充我们的正向图结构
    for node_name, input_names in graph_dict.items():
        for input_name in input_names:
            # `input_name` 指向 `node_name`
            # 所以在正向图中，有一条从 input_name 到 node_name 的边
            adj[input_name].append(node_name)
            
            # 因为有一条边指向 node_name，所以它的入度加 1
            in_degree[node_name] += 1

    # --- 步骤 2: 初始化队列 ---
    # 队列的初始成员是所有入度为 0 的节点。
    # 这些是图的起点，包括模型的初始输入和所有不依赖任何计算的参数节点。
    queue = [node for node in all_nodes if in_degree[node] == 0]
    
    # 这是一个健壮性检查，确保模型的初始输入确实是图的起点
    if not all(node in queue for node in start_nodes):
        missing = [node for node in start_nodes if node not in queue]
        print(f"Warning: Model start_nodes {missing} have dependencies, which is unusual.")

    # --- 步骤 3: 执行 Kahn 算法 ---
    sorted_nodes = [] # 用于存储最终的排序结果
    head = 0 # 使用列表模拟队列，避免频繁 pop(0) 的低效

    while head < len(queue):
        # 从队列中取出一个节点 u
        u = queue[head]
        head += 1
        
        # 我们只关心计算节点，即那些在 graph_dict 中作为 key 存在的节点。
        # 参数节点和初始输入节点虽然参与排序，但不是我们要执行的 "操作"。
        if u in graph_dict:
            sorted_nodes.append(u)
        
        # 遍历 u 的所有邻居 v (即所有以 u 为输入的节点)
        # 想象一下把 u 节点和它发出的所有边从图中“移除”
        for v in adj[u]:
            # v 的入度减 1
            in_degree[v] -= 1
            
            # 如果 v 的入度变为 0，说明它的所有前驱节点都已经被处理完毕
            # 现在可以将 v 加入队列了
            if in_degree[v] == 0:
                queue.append(v)

    # --- 步骤 4: 检查结果 ---
    # 如果排序后队列的总长度不等于图中所有节点的数量，说明图中有环。
    if len(queue) != len(all_nodes):
        # 找出图中仍然有入度的节点，它们是环的一部分
        cycle_nodes = [node for node, degree in in_degree.items() if degree > 0]
        raise RuntimeError(f"Graph has a cycle, topological sort failed. Cycle involves nodes: {cycle_nodes}")
        
    return sorted_nodes


def plan_and_generate_relu_keys(pytorch_model, dummy_input, key_type, num_inferences=1):
    """
    【修正版】能够正确处理多输入模型（如 BERT）的规划和密钥生成。
    """
    print("--- Starting Dummy Run for Resource Planning ---")

    print("[1/4] Tracing PyTorch model to build computation graph...")
    # 获取 ONNX 字节流
    onnx_bytes_io = _from_pytorch_to_bytes(pytorch_model, dummy_input)
    # 加载为 ONNX 模型对象，以便访问 Initializer
    onnx_model = _load_onnx_model(onnx_bytes_io)
    # 转换为 CrypTen Graph
    crypten_graph = crypten.nn.from_onnx(onnx_bytes_io)
    onnx_bytes_io.close()
    print("    ...Tracing complete.")

    # --- 2. 遍历图，创建物料清单 ---
    print("[2/3] Traversing graph to create resource manifest...")
    resource_manifest = []
    tensor_shapes = {}

    # --- 关键修正：正确初始化所有输入的形状 ---
    # 检查 dummy_input 是否是元组/列表，以处理多输入情况
    is_multi_input = isinstance(dummy_input, (list, tuple))
    
    if is_multi_input:
        # 确保输入名称数量和虚拟输入张量数量匹配
        if len(crypten_graph.input_names) != len(dummy_input):
             raise ValueError(
                f"Model has {len(crypten_graph.input_names)} inputs, "
                f"but {len(dummy_input)} dummy tensors were provided."
            )
        # 遍历所有输入，逐一初始化形状
        for name, tensor in zip(crypten_graph.input_names, dummy_input):
            tensor_shapes[name] = tuple(tensor.shape)
            print(f"    - Initializing input '{name}' with shape {tuple(tensor.shape)}")
    else: # 单输入情况
        input_name = crypten_graph.input_names[0]
        tensor_shapes[input_name] = tuple(dummy_input.shape)
        print(f"    - Initializing single input '{input_name}' with shape {tuple(dummy_input.shape)}")
    for initializer in onnx_model.graph.initializer:
        # initializer.name 是参数的名称
        # initializer.dims 是参数的形状
        shape = tuple(initializer.dims)
        tensor_shapes[initializer.name] = shape
        print(f"    - Found Model Parameter: '{initializer.name}' with shape {shape}")
    
    print("    ...Shape initialization complete.")
    # 获取正确的计算顺序
    nodes_in_order = _topological_sort(crypten_graph._graph, crypten_graph.input_names)
    print(f"    ...Topological sort complete. Found {len(nodes_in_order)} computation nodes.")
    keys = {}
    for node_name in nodes_in_order:
        module_instance = crypten_graph._modules[node_name]
        input_names = crypten_graph._graph.get(node_name, [])
        if not input_names: continue

        # --- 关键修正：正确处理多输入模块 ---
        # 形状推断逻辑需要更健壮
        # 这里我们做一个简化：假设大多数模块是单输入的
        # 对于 Add 等多输入模块，需要特殊处理
        try:
            current_input_shape = tensor_shapes[input_names[0]]
            output_shape = current_input_shape # 简化推断
            tensor_shapes[node_name] = output_shape
        except KeyError:
            print(f"    - WARNING: Could not find shape for input '{input_names[0]}' of node '{node_name}'. Shape inference might be incomplete.")
            continue # 跳过无法推断的节点
        if isinstance(module_instance, (ReLUFastSecNet, GELU, SiLU, Softmax)):
            num_keys = torch.prod(torch.tensor(current_input_shape)).item()
            manifest_item = {
                "layer_name": node_name,
                "op_type": key_type,
                "shape": current_input_shape,
                "count": num_keys
            }
            resource_manifest.append(manifest_item)
            keys[node_name] = key_type.gen(num_keys)
            print(f"    - Found compatible layer '{node_name}', generated a keys for shape {current_input_shape}.")
    print("    ...Manifest created.")
    print("[3/3] Saving keys and the manifest...")
    if len(resource_manifest) > 0:
        data_path = f"{param_path}{key_type.__name__}/"
        os.makedirs(data_path, exist_ok=True)
        saved_name = f"relu_{key_type.__name__}"
        key0_path = os.path.join(data_path, f"{saved_name}_0.pkl")
        key1_path = os.path.join(data_path, f"{saved_name}_1.pkl")
        keys_0 = {item: keys[item][0] for item in keys}
        keys_1 = {item: keys[item][1] for item in keys}
        with open(key0_path, 'wb') as f:
            pickle.dump(keys_0, f)
        with open(key1_path, 'wb') as f:
            pickle.dump(keys_1, f)
        with open(key1_path, 'rb') as f:
            temp = list(pickle.load(f).values())
        for i in temp:
            i = DCFKey.from_dic(i.dcf_key)
        print(temp)
        manifest_path = os.path.join(data_path, f"{saved_name}_manifest.pkl")
        with open(manifest_path, 'wb') as f:
            pickle.dump(resource_manifest, f)
        print(f"    ...{len(keys)} {key_type.__name__} and manifest saved to '{data_path}'.")
    else:
        print("    ...No layers requiring FSS keys were found.")
    print("\n--- Dummy Run and Key Generation Finished Successfully! ---")


# --- 这是您需要添加的 main 函数 ---
def old_main():
    # 1. 设置命令行参数解析
    parser = argparse.ArgumentParser(description="Plan and generate FSS keys for a given model.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--num_inferences",
        type=int,
        default=1,
        help="Number of inferences to generate keys for.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the dummy input.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=128,
        help="Sequence length for the dummy input.",
    )
    args = parser.parse_args()

    # 2. 加载明文 PyTorch 模型
    print(f"Loading plaintext model: {args.model_name_or_path}...")
    # 这里的 config 需要根据实际模型调整，对于 BERT 来说通常是这样
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=2)
    pytorch_model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    pytorch_model.eval() # 确保是评估模式
    print("Model loaded successfully.")

    # 3. 创建形状正确的虚拟输入 (dummy_input)
    print(f"Creating dummy input with shape: ({args.batch_size}, {args.seq_length})")
    # 对于 BERT 类型的模型，输入通常是 (input_ids, attention_mask, token_type_ids)
    # 我们用整数张量来模拟 input_ids
    dummy_ids = torch.randint(0, config.vocab_size, (args.batch_size, args.seq_length))
    dummy_mask = torch.ones(args.batch_size, args.seq_length, dtype=torch.long)
    dummy_token_ids = torch.zeros(args.batch_size, args.seq_length, dtype=torch.long)
    
    # 将它们打包成元组，因为 from_pytorch 接受元组作为多输入
    dummy_input = (dummy_ids, dummy_mask, dummy_token_ids)
    print("Dummy input created.")

    # 4. 调用我们的核心规划和生成函数
    plan_and_generate_relu_keys(
        pytorch_model, 
        dummy_input, 
        FastSecNetReLUKey,
        num_inferences=args.num_inferences
    )

if __name__ == "__main__":
    main()