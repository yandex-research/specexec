from typing import Dict, Optional, Tuple, Any  # noqa: F401

# import numpy as np
import torch
import torch.nn.functional as F
import transformers


class EngineStatic:
    def __init__(self, model_name, max_len, dtype=torch.float16, device="cuda:0"):
        self.model_name = model_name
        self.max_len = max_len
        self.device = device
        if isinstance(model_name, str):
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        else:
            self.model = model_name
        self.config = self.model.config
        self.dtype = self.model.dtype
        self.model._setup_cache(StaticCachePlus, 1, max_cache_len=self.max_len)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: torch.LongTensor = None,
    ):

        output = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )
        return output.logits

    @property
    def kv_len_used(self):
        # was: return self.model.model.layers[0].self_attn.past_key_value.get_seq_length().item()
        for layer in self.model.model.layers:
            break  # compatible with offloading code
        cache_corner = layer.self_attn.past_key_value.key_cache[0, 0, :, 0] != 0.0
        result = 0 if not torch.any(cache_corner) else torch.where(cache_corner)[0].max(dim=-1).indices + 1  # avoiding get_seq_length()
        return result
        # return layer.self_attn.past_key_value.get_seq_length().item()

    @torch.inference_mode()
    def clear_kv(self):
        for layer in self.model.model.layers:
            layer.self_attn.past_key_value.key_cache.zero_()
            layer.self_attn.past_key_value.value_cache.zero_()

    def reorder_cache_tokens(self, source_token_idxs: torch.tensor, dest_token_idxs: torch.tensor = None):
        """Applies indices mask to KV cache or truncates it"""

        if dest_token_idxs is None:  # assumed that destination starts from cache beginning
            dest_size = source_token_idxs.sum() if source_token_idxs.dtype == torch.bool else source_token_idxs.shape[-1]
            dest_token_idxs = torch.arange(dest_size, device=self.device)

        if (source_token_idxs.dtype == torch.bool) and (source_token_idxs.shape[-1] < self.max_len):
            source_token_idxs = F.pad(input=source_token_idxs, pad=(0, self.max_len - source_token_idxs.shape[-1]), mode="constant", value=False)
        for layer in self.model.model.layers:
            layer.self_attn.past_key_value.reorder_cache_tokens(source_token_idxs=source_token_idxs, dest_token_idxs=dest_token_idxs)

    def set_max_len(self, new_max_len):
        if self.kv_len_used > new_max_len:
            raise ValueError(f"Current cache size {self.kv_len_used()} is greater than new `max_len` {new_max_len}.")
        for layer in self.model.model.layers:
            layer.self_attn.past_key_value.resize(new_max_len)

        self.max_len = new_max_len

    def _forward(
        model,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: torch.LongTensor = None,
    ):
        output = model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )
        return output.logits


# def capture_graph_compiled(engine, decoding_seqlen: int = 1):
#     device = engine.device
#     dtype = engine.dtype
#     static_input_ids = torch.full((1, decoding_seqlen), 0, dtype=torch.long, device=device)
#     static_position_ids = torch.full((1, decoding_seqlen), 0, dtype=torch.long, device=device)
#     static_storage_ids = torch.arange(decoding_seqlen, dtype=torch.long, device=device)
#     static_attn_mask = torch.full((decoding_seqlen, engine.max_len), 0, dtype=dtype, device=device)
#     static_attn_mask = static_attn_mask[None, None, :, :]


# forward_compiled = torch.compile(_forward, mode="reduce-overhead", fullgraph=True)

# graph = torch.cuda.CUDAGraph()
# with torch.cuda.graph(graph, pool=mempool):
#     static_logits = engine.forward(
#         input_ids=static_input_ids, cache_position=static_storage_ids, position_ids=static_position_ids, attention_mask=static_attn_mask
#     )

# def run(input_ids, storage_ids, position_ids, attn_mask):
#     static_input_ids.copy_(input_ids)
#     static_storage_ids.copy_(storage_ids)
#     static_position_ids.copy_(position_ids)
#     static_attn_mask.copy_(attn_mask)
#     graph.replay()
#     return static_logits.clone()

# return run


class EngineStaticCompiled(EngineStatic):
    @torch.inference_mode()
    def __init__(self, model_name=None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        decoding_seqlens = [128, self.max_len]
        self.compilelds = {}
        for decoding_seqlen in decoding_seqlens:
            pass
            # self.compilelds[decoding_seqlen] = capture_graph_compiled(engine=self.engine, decoding_seqlen=decoding_seqlen)

        def _forward(
            model,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cache_position: torch.LongTensor = None,
        ):
            output = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
            )
            return output.logits

        self.forward_compiled = torch.compile(_forward, mode="reduce-overhead", fullgraph=True)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: torch.LongTensor = None,
    ):
        logits = self.forward_compiled(
            model=self.model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )
        return logits


class StaticCachePlus(transformers.StaticCache):

    def reorder_cache_tokens(self, source_token_idxs: torch.LongTensor, dest_token_idxs=None):
        """
        reorders cache along axis=2 (tokens)
        assumes batch_size==1  
        accepts int and bool indices
        if source_token_idx.dtype == torch.bool assumes source_token_idx.shape == self.key_cache.shape[2]
        resets setting {{ the rest of cache OR .key_cache[0, 0] }} to zeros to ensure that get_seq_length() works right
        """
        if (source_token_idxs.dtype == torch.bool) and (source_token_idxs.shape[-1] < self.key_cache.shape[-2]):
            source_token_idxs = F.pad(input=source_token_idxs, pad=(0, self.key_cache.shape[-2] - source_token_idxs.shape[-1]), mode="constant", value=False)

        if dest_token_idxs is None:
            dest_size = source_token_idxs.sum() if source_token_idxs.dtype == torch.bool else source_token_idxs.shape[-1]
            dest_token_idxs = torch.arange(dest_size)

        self.key_cache[:, :, dest_token_idxs.to(self.key_cache.device), :] = self.key_cache[:, :, source_token_idxs.to(self.key_cache.device), :]
        self.value_cache[:, :, dest_token_idxs.to(self.value_cache.device), :] = self.value_cache[:, :, source_token_idxs.to(self.value_cache.device), :]

        right_edge = (torch.where(dest_token_idxs)[0].max() if dest_token_idxs.dtype == torch.bool else dest_token_idxs.max()) + 1

        # setting the rest of the cache to zeros to ensure that get_seq_length() works right
        self.key_cache[0, 0, right_edge.to(self.key_cache.device) :, :].zero_()

    def resize(self, new_len):
        if new_len <= self.key_cache.shape[-2]:
            self.key_cache = self.key_cache[:, :, :new_len, :]
            self.value_cache = self.value_cache[:, :, :new_len, :]
        else:
            self.key_cache = F.pad(input=self.key_cache, pad=(0, 0, 0, new_len - self.key_cache.shape[-2]), mode="constant", value=0)
            self.value_cache = F.pad(input=self.value_cache, pad=(0, 0, 0, new_len - self.value_cache.shape[-2]), mode="constant", value=0)
        self.max_cache_len = new_len
