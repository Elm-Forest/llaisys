import json
import mmap
import struct
import ctypes
import numpy as np
from pathlib import Path
from typing import Sequence, List, Dict, Any

from ..libllaisys import LIB_LLAISYS, DeviceType, DataType
from ..libllaisys.models import LlaisysQwen2Meta, LlaisysQwen2Weights
from ..tensor import Tensor

class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        self.device = device
        # 保存 Tensor 引用防止被 Python GC 回收，导致 C++ 指针悬空
        self._tensor_refs = []

        # 1. 加载 Config
        config_path = self.model_path / "config.json"
        if not config_path.exists():
             # 尝试递归查找
            candidates = list(self.model_path.rglob("config.json"))
            if candidates:
                config_path = candidates[0]
            else:
                raise FileNotFoundError(f"config.json not found in {self.model_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 2. 准备 Meta
        # 注意：我们将所有权重在 Python 端转换为 F32，所以告诉 C++ 我们使用的是 F32
        self.meta = LlaisysQwen2Meta()
        self.meta.dtype = DataType.F32 
        self.meta.nlayer = int(config["num_hidden_layers"])
        self.meta.hs = int(config["hidden_size"])
        self.meta.nh = int(config["num_attention_heads"])
        self.meta.nkvh = int(config.get("num_key_value_heads", self.meta.nh))
        self.meta.dh = self.meta.hs // self.meta.nh
        self.meta.di = int(config["intermediate_size"])
        self.meta.maxseq = int(config.get("max_position_embeddings", 2048))
        self.meta.voc = int(config["vocab_size"])
        self.meta.epsilon = float(config["rms_norm_eps"])
        self.meta.theta = float(config.get("rope_theta", 10000.0))
        self.meta.end_token = int(config.get("eos_token_id", 151643))

        # 3. 创建 C 模型
        device_ids = (ctypes.c_int * 1)(0)
        self.handle = LIB_LLAISYS.llaisysQwen2ModelCreate(
            ctypes.byref(self.meta), 
            self.device.value, 
            device_ids, 
            1
        )
        if not self.handle:
            raise RuntimeError("Failed to create C++ model instance")
        
        # 4. 获取权重指针结构体
        self.weights_struct = LIB_LLAISYS.llaisysQwen2ModelWeights(self.handle).contents

        # 5. 加载权重 (使用手动解析 header + mmap 方式)
        self._load_weights()

    def _load_weights(self):
        files = sorted(self.model_path.glob("*.safetensors"))
        if not files:
            files = sorted(self.model_path.rglob("*.safetensors"))
        
        if not files:
            print(f"Warning: No safetensors found in {self.model_path}")
            return

        print(f"Loading weights from {len(files)} safetensors files...")
        for file in files:
            self._load_safetensors_file(file)

    def _load_safetensors_file(self, file_path: Path):
        """
        手动解析 safetensors 文件，绕过 numpy 对 bfloat16 的限制。
        """
        with open(file_path, "rb") as f:
            # 1. 读取头部长度 (8字节 uint64)
            header_size_bytes = f.read(8)
            if len(header_size_bytes) != 8:
                return
            header_size = struct.unpack("<Q", header_size_bytes)[0]
            
            # 2. 读取并解析头部 JSON
            header_json = f.read(header_size).decode("utf-8")
            header = json.loads(header_json)
            
            # 数据区开始的绝对偏移量
            data_start = 8 + header_size

            # 3. 使用 mmap 映射文件数据区
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                for name, info in header.items():
                    if name == "__metadata__":
                        continue
                    
                    dtype_str = info["dtype"]
                    shape = info["shape"]
                    offsets = info["data_offsets"]
                    
                    # 获取原始字节
                    start = data_start + offsets[0]
                    end = data_start + offsets[1]
                    raw_bytes = mm[start:end]

                    # 4. 关键：将数据转换为 Float32 Numpy Array
                    # 无论原始是 F16 还是 BF16，我们都转为 F32 传给 C++
                    # 这样避免了依赖特定 numpy 版本或 torch
                    np_data = self._convert_to_f32(raw_bytes, dtype_str)
                    
                    if np_data is not None:
                        self._dispatch_weight(name, np_data, shape)

    def _convert_to_f32(self, raw_bytes: bytes, dtype_str: str) -> np.ndarray:
        """
        将原始字节转换为 float32 数组。
        处理 BF16 的黑魔法就在这里。
        """
        if dtype_str == "BF16":
            # 读取为 uint16
            raw_u16 = np.frombuffer(raw_bytes, dtype=np.uint16)
            # 核心技巧：BF16 是 FP32 的高16位。
            # 将 uint16 转为 uint32，左移 16 位，然后 view 为 float32
            # [BF16_bits] -> [BF16_bits | 0000000000000000] (in FP32 format)
            arr_f32 = (raw_u16.astype(np.uint32) << 16).view(np.float32)
            return arr_f32
        
        elif dtype_str == "F16":
            # 标准 F16，numpy 通常支持读取，然后转为 F32
            return np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)
        
        elif dtype_str == "F32":
            return np.frombuffer(raw_bytes, dtype=np.float32)
        
        return None

    def _dispatch_weight(self, name: str, data: np.ndarray, shape: List[int]):
        """
        将转换好的 F32 数据加载到 C++ 对应的 Tensor 中。
        """
        # 辅助：创建 Tensor 并加载数据
        def load_to_ptr(c_tensor_ptr):
            if not c_tensor_ptr:
                return # C++ 端没有初始化这个层（例如 layer index 超出）
            
            # 使用 libllaisys.tensor 的 Tensor 类包装 C 指针
            # 注意：我们这里不需要 create 新 tensor，因为 C++ 已经在 ModelCreate 时分配了内存
            # 我们只需要把数据 memcpy 进去。
            
            # 获取目标 Tensor 的封装
            t = Tensor(tensor=c_tensor_ptr)
            
            # 检查形状是否匹配（可选，但推荐）
            # if t.shape != shape: print(f"Shape mismatch {name}")
            
            # 加载数据
            t.load(data.ctypes.data)

        # 权重映射逻辑
        if name == "model.embed_tokens.weight":
            load_to_ptr(self.weights_struct.in_embed)
        elif name == "model.norm.weight":
            load_to_ptr(self.weights_struct.out_norm_w)
        elif name == "lm_head.weight":
            load_to_ptr(self.weights_struct.out_embed)
        elif name.startswith("model.layers."):
            parts = name.split(".")
            try:
                idx = int(parts[2])
            except ValueError:
                return
            
            if idx >= self.meta.nlayer:
                return

            suffix = ".".join(parts[3:])
            w = self.weights_struct

            if suffix == "input_layernorm.weight":
                load_to_ptr(w.attn_norm_w[idx])
            elif suffix == "post_attention_layernorm.weight":
                load_to_ptr(w.mlp_norm_w[idx])
            
            # Attention
            elif suffix == "self_attn.q_proj.weight":
                load_to_ptr(w.attn_q_w[idx])
            elif suffix == "self_attn.q_proj.bias":
                load_to_ptr(w.attn_q_b[idx])
            elif suffix == "self_attn.k_proj.weight":
                load_to_ptr(w.attn_k_w[idx])
            elif suffix == "self_attn.k_proj.bias":
                load_to_ptr(w.attn_k_b[idx])
            elif suffix == "self_attn.v_proj.weight":
                load_to_ptr(w.attn_v_w[idx])
            elif suffix == "self_attn.v_proj.bias":
                load_to_ptr(w.attn_v_b[idx])
            elif suffix == "self_attn.o_proj.weight":
                load_to_ptr(w.attn_o_w[idx])
            
            # MLP
            elif suffix == "mlp.gate_proj.weight":
                load_to_ptr(w.mlp_gate_w[idx])
            elif suffix == "mlp.up_proj.weight":
                load_to_ptr(w.mlp_up_w[idx])
            elif suffix == "mlp.down_proj.weight":
                load_to_ptr(w.mlp_down_w[idx])

    def __del__(self):
        if hasattr(self, "handle") and self.handle:
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self.handle)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        generated = list(inputs)
        curr_len = len(generated)
        
        # 1. Prefill
        in_arr = (ctypes.c_int64 * curr_len)(*generated)
        next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.handle, in_arr, curr_len)
        generated.append(next_token)
        
        # 2. Decode
        for _ in range(max_new_tokens - 1):
            if next_token == self.meta.end_token:
                break
                
            in_arr = (ctypes.c_int64 * 1)(next_token)
            next_token = LIB_LLAISYS.llaisysQwen2ModelInfer(self.handle, in_arr, 1)
            generated.append(next_token)
            
        return generated