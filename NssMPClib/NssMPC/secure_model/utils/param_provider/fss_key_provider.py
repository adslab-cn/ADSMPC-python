# NssMPC/secure_model/utils/param_provider/fss_key_provider.py

from ._base_param_provider import BaseParamProvider
from NssMPC.config import DEBUG_LEVEL

class FSSKeyProvider(BaseParamProvider):
    def __init__(self, param_type, saved_name=None, param_tag=None, root_path=None):
        super().__init__(param_type, saved_name, param_tag, root_path)
        self.param = []
        self._key_queue = [] # 使用队列更直观

    def load_params(self):
        if not isinstance(self.param, list):
            raise TypeError(f"{self.param_type.__name__} provider expects a list of keys.")
        
        self._key_queue = self.param.copy()
        print(f"Loaded {len(self._key_queue)} key objects for {self.param_type.__name__}")

    def get_parameters(self, num_elements: int):
        """
        从队列的头部获取下一个FSS密钥。
        """
        if not self._key_queue:
            if DEBUG_LEVEL:
                print(f"Warning: FSS key queue for {self.param_type.__name__} is empty. Re-loading for DEBUG.")
                self.load_params() # 在DEBUG模式下重新加载
            else:
                raise IndexError(f"Ran out of FSS key objects for {self.param_type.__name__}")

        key_to_return = self._key_queue.pop(0)

        if len(key_to_return) != num_elements:
            raise ValueError(
                f"FSS Key size mismatch for {self.param_type.__name__}. "
                f"Requested: {num_elements}, but the pre-generated key has size: {len(key_to_return)}. "
                "Ensure dummy_model and inference run on the same shapes."
            )
            
        return key_to_return