# crypten/mpc/provider.py

import os
import pickle
import threading

from crypten.utils.generate_model_plan import DCFKey
class ParamProvider:
    """
    为 CrypTen/SHAFT 实现的、基于缓冲池和指针的参数提供者。
    它的行为模式与 NssMPClib 中的 ParamProvider 非常相似。
    """

    def __init__(self, param_type, party_id, saved_name,data_path):
        """
        初始化 Provider。

        Args:
            param_type (type): 要管理的 Parameter 类 (例如 FSSCompareKey).
            party_id (int): 当前 party 的 rank (0 或 1).
            saved_name (str): 密钥文件的基础名 (例如 "FSSR_keys").
        """
        self.param_type = param_type
        self.party_id = party_id
        self.saved_name = saved_name
        self._lock = threading.Lock()  # 保证线程安全

        # 初始时，参数池为空，需要手动加载
        self.param_pool = None
        self.total_count = 0
        self.ptr = 0  # 消耗指针

        # 在初始化时就直接加载
        self._load_params(data_path)

    def _load_params(self,data_path):
        """
        从磁盘加载完整的、批处理过的参数文件。
        """# e.g., '~/.crypten/data/'
        file_name = f"{self.saved_name}_{self.party_id}.pkl"
        file_path = os.path.join(data_path, file_name)

        try:
            with open(file_path, 'rb') as f:
                self.param_pool = list(pickle.load(f).values())
                for i in self.param_pool:
                    i.dcf_key = DCFKey.from_dic(i.dcf_key)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Key file not found: {file_path}. "
                f"Please run the offline 'plan_and_generate' script first."
            )

        # 假设 param_pool 是一个 Parameter 对象，它实现了 __len__
        self.total_count = len(self.param_pool)
        self.ptr = 0  # 每次重新加载都重置指针

        print(f"Provider for {self.param_type.__name__} initialized for party {self.party_id}. "
              f"Loaded a pool of {self.total_count} keys.")

    def get_parameters(self, number_of_params):
        """
        核心方法：从池中获取指定数量的参数。
        """
        with self._lock:  # 保证对指针的修改是原子操作
            if self.param_pool is None:
                raise RuntimeError("Parameters have not been loaded. Call _load_params() first.")

            start_idx = self.ptr
            end_idx = start_idx + number_of_params

            if end_idx > self.total_count:
                #TODO joker need to work here
                self.ptr = 0
                start_idx = self.ptr
                end_idx = start_idx + number_of_params
                # raise ValueError(
                #     f"Not enough pre-computed parameters in the pool for this request. "
                #     f"Requested: {number_of_params}, "
                #     f"Available: {self.total_count - start_idx}."
                # )

            # 使用 Parameter 类自己的 __getitem__ (切片) 功能
            # 这会返回一个新的、尺寸正确的 Parameter 对象
            params_for_this_call = self.param_pool[start_idx:end_idx]

            # 更新消耗指针
            self.ptr = end_idx

            return params_for_this_call

    def reset_for_new_inference(self):
        """
        重置消耗指针，以便可以对新一批数据进行推理。
        这假设离线生成的密钥总量足够多次推理。
        """
        with self._lock:
            self.ptr = 0
            print(f"Provider for {self.param_type.__name__} has been reset for a new inference run.")