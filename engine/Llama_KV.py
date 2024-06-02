from typing import List, Optional

import torch
import torch.nn.functional as F
from transformers import LlamaConfig


class KV_Cache:
    def __init__(
        self,
        config: LlamaConfig,
        batch_size: int = 1,
        max_length: int = 256,
        device: str = "cuda:0",
        dtype=torch.float16,
    ) -> None:
        self.config = config
        self.max_length = max_length
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype,
        )

        self.v_cache = torch.zeros(
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_length,
            config.hidden_size // config.num_attention_heads,
            device=self.device,
            dtype=self.dtype,
        )
        self.num_layers = config.num_hidden_layers
        self.kv_offset = 0

    def initialize_kv(self, k_cache: torch.Tensor, v_cache: torch.Tensor, kv_len: int):

        self.k_cache[..., :kv_len, :] = k_cache[..., :kv_len, :]
        self.v_cache[..., :kv_len, :] = v_cache[..., :kv_len, :]

        self.kv_offset = kv_len

    def gather_kv(self, indices: List[int]):

        self.k_cache[..., : len(indices), :] = self.k_cache[..., indices, :]
        self.v_cache[..., : len(indices), :] = self.v_cache[..., indices, :]

        self.k_cache[..., len(indices) :, :] = 0.0
        self.v_cache[..., len(indices) :, :] = 0.0

        self.kv_offset = len(indices)

    def reorder_cache_tokens(self, source_token_idxs: torch.Tensor, dest_token_idxs: Optional[torch.Tensor] = None):
        if source_token_idxs.dtype == torch.bool:
            pad_size = self.k_cache.shape[-2] - source_token_idxs.shape[-1]
            source_token_idxs = torch.nn.functional.pad(input=source_token_idxs, pad=(0, pad_size), mode="constant", value=False)

        if dest_token_idxs is None:
            dest_size = source_token_idxs.sum() if source_token_idxs.dtype == torch.bool else source_token_idxs.shape[-1]
            dest_token_idxs = torch.arange(dest_size)

        self.k_cache[..., dest_token_idxs.to(self.k_cache.device), :] = self.k_cache[..., source_token_idxs.to(self.k_cache.device), :]
        self.v_cache[..., dest_token_idxs.to(self.v_cache.device), :] = self.v_cache[..., source_token_idxs.to(self.v_cache.device), :]

        right_edge = torch.where(dest_token_idxs)[0].max() if dest_token_idxs.dtype == torch.bool else dest_token_idxs.max()

        # setting the rest of the cache to zeros to ensure that get_seq_length() works right
        self.k_cache[0, 0, 0, right_edge.to(self.k_cache.device) :, :].zero_()

    def gather_kv_incremental(self, indices: List[int], offset: int):

        self.k_cache[..., offset : offset + len(indices), :] = self.k_cache[..., indices, :]
        self.v_cache[..., offset : offset + len(indices), :] = self.v_cache[..., indices, :]

        self.k_cache[..., offset + len(indices) :, :] = 0.0
        self.v_cache[..., offset + len(indices) :, :] = 0.0

        self.kv_offset = offset + len(indices)

    def update_kv_cache(self, new_k_cache: torch.Tensor, new_v_cache: torch.Tensor, layer_idx: int, storage_ids: torch.LongTensor, debug: bool = False):

        input_length = len(storage_ids)
        if debug:
            assert input_length == new_k_cache.shape[-2]
            assert input_length == new_v_cache.shape[-2]

        self.k_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_k_cache)
        self.v_cache[layer_idx].index_copy_(dim=-2, index=storage_ids, source=new_v_cache)

        if layer_idx == self.num_layers - 1:
            self.kv_offset += input_length
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    @torch.inference_mode
    def clear(self):
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.kv_offset = 0

    def get_usable_length(self, layer_idx: int, input_length: int):
        if layer_idx == self.num_layers - 1:
            return self.kv_offset
        else:
            return self.kv_offset + input_length

    def set_kv_len(self, kv_len: int):
        self.kv_offset = kv_len

    @property
    def kv_len_used(self):
        return (self.k_cache[0, 0, 0, :, 0] != 0).sum()

    def set_max_len(self, new_max_len):
        if self.kv_len_used > new_max_len:
            raise ValueError(f"Current cache size {self.kv_len_used()} is greater than new `max_len` {new_max_len}.")

        if new_max_len <= self.k_cache.shape[-2]:
            self.k_cache = self.k_cache[..., :new_max_len, :]
            self.v_cache = self.v_cache[..., :new_max_len, :]
        else:
            self.k_cache = F.pad(input=self.k_cache, pad=(0, 0, 0, new_max_len - self.k_cache.shape[-2]), mode="constant", value=0)
            self.v_cache = F.pad(input=self.v_cache, pad=(0, 0, 0, new_max_len - self.v_cache.shape[-2]), mode="constant", value=0)
