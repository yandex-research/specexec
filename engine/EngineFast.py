"""
based on streamlined Llama implementation from Sequoia repo (https://github.com/Infini-AI-Lab/Sequoia)
"""

import gc
import logging

# from .Llama_modules import LlamaAttention_FI, LlamaAttention_TG
from typing import List, Optional, Tuple, Union  # noqa: F401s

import accelerate
import torch
import torch.nn.functional as F
import transformers

from .Llama_KV import KV_Cache
from .Llama_model import LlamaForCausalLM_FI, LlamaForCausalLM_TG

from specdec import utils

if "logger" not in globals():
    logger = utils.get_logger()


class InferenceEngine:
    def __init__(self, model_name: str, max_len: int, dtype=torch.float16, device="cuda:0") -> None:

        self.device = device
        self.dtype = dtype
        self.max_len = max_len

        self.model = LlamaForCausalLM_FI.from_pretrained(model_name, torch_dtype=dtype, device_map=device)
        self.model.eval()
        self.config = self.model.config

        self.kv_cache = KV_Cache(config=self.model.config, max_length=max_len, device=device, dtype=dtype)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        debug: bool = False,
    ):
        # if attention_mask.min() == 0:
        attention_mask = (1 - attention_mask) * torch.finfo(attention_mask.dtype).min  # inverting attn mask
        if debug:
            _, input_length = input_ids.shape
            assert cache_position.shape[0] == input_length
            assert attention_mask.shape[-2] == input_length
            assert attention_mask.shape[-1] == self.max_len
            assert position_ids.shape[1] == input_length

        logits = self.model(
            input_ids=input_ids,
            max_length=self.max_len,
            storage_ids=cache_position,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=self.kv_cache,
            debug=debug,
        )
        return logits

    def clear_kv(self) -> None:
        self.kv_cache.clear()

    def initialize_kv(self, k_cache: torch.Tensor, v_cache: torch.Tensor, kv_len: int):
        self.kv_cache.initialize_kv(k_cache, v_cache, kv_len)

    def reorder_cache_tokens(self, source_token_idxs: torch.LongTensor, dest_token_idxs=None):
        if source_token_idxs.numel():
            self.kv_cache.reorder_cache_tokens(source_token_idxs=source_token_idxs, dest_token_idxs=dest_token_idxs)

    def get_kv_cache(self, in_place=False):
        if not in_place:
            return self.kv_cache.k_cache.clone(), self.kv_cache.v_cache.clone()
        else:
            return self.kv_cache.k_cache, self.kv_cache.v_cache

    @property
    def kv_len_used(self):
        return self.kv_cache.kv_len_used

    def set_max_len(self, new_max_len):
        self.kv_cache.set_max_len(new_max_len)
        self.max_len = new_max_len


class InferenceEngineCompiled(InferenceEngine):

    @torch.inference_mode()
    def __init__(self, model_name=None, **kwargs):
        super().__init__(model_name=model_name, **kwargs)

        def _forward(model, input_ids: torch.LongTensor, max_length, attention_mask, position_ids, storage_ids, kv_cache):
            attention_mask = (1 - attention_mask) * torch.finfo(attention_mask.dtype).min

            logits = model(
                input_ids=input_ids,
                max_length=max_length,
                storage_ids=storage_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
            return logits

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
            max_length=self.max_len,
            storage_ids=cache_position,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=self.kv_cache,
        )
        return logits


class InferenceEngineTG(InferenceEngine):
    def __init__(self, max_len: int, model_name_or_path: str, dtype=torch.float16, device="cuda:0", offloading=False) -> None:
        super().__init__(max_len, model_name_or_path, dtype, device)
        if offloading:
            self.model = LlamaForCausalLM_TG.from_pretrained(model_name_or_path, torch_dtype=dtype)
            self.model.eval()
            self.model = accelerate.cpu_offload(self.model, execution_device=self.device)
        else:
            self.model = LlamaForCausalLM_TG.from_pretrained(model_name_or_path, torch_dtype=dtype, device_map=device)
            self.model.eval()

    def set_kv_len(self, kv_len: int):
        self.kv_cache.set_kv_len(kv_len)


def capture_graph(engine: InferenceEngine, decoding_seqlen: int = 1, mempool=None, n_warmups: int = 3):
    device = engine.device
    dtype = engine.dtype
    static_input_ids = torch.full((1, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_position_ids = torch.full((1, decoding_seqlen), 0, dtype=torch.long, device=device)
    static_storage_ids = torch.arange(decoding_seqlen, dtype=torch.long, device=device)
    static_attn_mask = torch.full((decoding_seqlen, engine.max_len), 0, dtype=dtype, device=device)
    static_attn_mask = static_attn_mask[None, None, :, :]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmups):
            static_logits = engine.forward(
                input_ids=static_input_ids, cache_position=static_storage_ids, position_ids=static_position_ids, attention_mask=static_attn_mask
            )
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        static_logits = engine.forward(
            input_ids=static_input_ids, cache_position=static_storage_ids, position_ids=static_position_ids, attention_mask=static_attn_mask
        )

    def run(input_ids, storage_ids, position_ids, attn_mask):
        static_input_ids.copy_(input_ids)
        static_storage_ids.copy_(storage_ids)
        static_position_ids.copy_(position_ids)
        static_attn_mask.copy_(attn_mask)
        graph.replay()
        return static_logits.clone()

    return run


class GraphInferenceEngine:
    def __init__(self, max_len: int, model_name_or_path: str, dtype=torch.float16, device="cuda:0") -> None:

        self.device = device
        self.dtype = dtype
        self.max_len = max_len
        self.engine = InferenceEngine(max_len=max_len, model_name=model_name_or_path, dtype=dtype, device=device)
        self.callables = {}
        self.mempool = None
        self.config = self.engine.config

    @torch.inference_mode()
    def initialize_cuda_graph(self, decoding_seqlens: List[int], n_warmups=3):
        gc.collect()
        self.mempool = torch.cuda.graphs.graph_pool_handle()
        for decoding_seqlen in sorted(decoding_seqlens):
            if decoding_seqlen not in self.callables and decoding_seqlen <= self.max_len:
                self.callables[decoding_seqlen] = capture_graph(engine=self.engine, decoding_seqlen=decoding_seqlen, mempool=self.mempool, n_warmups=n_warmups)
        self.engine.clear_kv()

    @torch.inference_mode()
    def graph_inference(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        storage_ids: Optional[torch.LongTensor] = None,
        debug: bool = False,
    ):

        dec_length = input_ids.shape[1]
        if debug:
            assert input_ids.shape[0] == 1
            assert storage_ids.shape[0] == dec_length
            assert position_ids.shape[0] == 1
            assert position_ids.shape[1] == dec_length
            assert attention_mask.shape[2] == dec_length
            assert attention_mask.shape[3] == self.engine.max_len
            assert attention_mask.shape[0] == 1
            assert attention_mask.shape[1] == 1
            assert attention_mask.device == self.device
            assert storage_ids.device == self.device
            assert position_ids.device == self.device
            assert input_ids.device == self.device
            assert dec_length in self.callables

        logits = self.callables[dec_length](input_ids, storage_ids, position_ids, attention_mask)
        return logits

    def clear_kv(self):
        self.engine.clear_kv()

    def initialize_kv(self, k_cache: torch.Tensor, v_cache: torch.Tensor, kv_len: int):
        self.engine.initialize_kv(k_cache, v_cache, kv_len)

    def get_kv_cache(self, in_place=False):
        return self.engine.get_kv_cache(in_place=in_place)

    def gather_kv(self, indices: List[int]):
        self.engine.reorder_cache_tokens(indices)

    @torch.inference_mode()
    def inference(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        storage_ids: torch.LongTensor = None,
    ):

        return self.engine.forward(input_ids=input_ids, cache_position=storage_ids, attention_mask=attention_mask, position_ids=position_ids)

    @property
    def kv_len_used(self) -> int:
        return self.engine.kv_len_used

    def reorder_cache_tokens(self, source_token_idxs: torch.LongTensor, dest_token_idxs: Optional[torch.LongTensor] = None) -> None:
        self.engine.reorder_cache_tokens(source_token_idxs=source_token_idxs, dest_token_idxs=dest_token_idxs)

    def set_max_len(self, new_max_len):
        self.engine.set_max_len(new_max_len)
        self.max_len = new_max_len
        self.callables = {}
        self.initialize_cuda_graph([*self.decoding_seqlens, new_max_len])

    @property
    def model(self):
        return self.engine.model


class GraphInferenceEngineTG:
    def __init__(self, max_len: int, model_name_or_path: str, dtype=torch.float16, device="cuda:0", offloading=False) -> None:

        self.device = device
        self.dtype = dtype
        self.max_len = max_len
        self.engine = InferenceEngineTG(max_len=max_len, model_name_or_path=model_name_or_path, dtype=dtype, device=device, offloading=offloading)

    def clear_kv(self):
        self.engine.clear_kv()

    def initialize_kv(self, k_cache: torch.Tensor, v_cache: torch.Tensor, kv_len: int):
        self.engine.initialize_kv(k_cache, v_cache, kv_len)

    def get_kv_cache(self, in_place=False):
        return self.engine.get_kv_cache(in_place=in_place)

    def gather_kv(self, indices: List[int]):
        self.engine.reorder_cache_tokens(indices)

    def set_kv_len(self, kv_len: int):
        self.engine.set_kv_len(kv_len)

    @torch.no_grad()
    def inference(
        self,
        input_ids: torch.LongTensor,
        storage_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):

        return self.engine.forward(input_ids=input_ids, cache_position=storage_ids, attention_mask=attention_mask, position_ids=position_ids)


class InferenceEnginePadded(GraphInferenceEngine):

    # @torch.inference_mode()
    def __init__(self, model_name=None, seqlens=[], **kwargs):
        self.config = transformers.AutoConfig.from_pretrained(model_name)
        assert self.config.model_type == 'llama', f"Only Llama family models are supported by {self.__class__.__name__}."

        super().__init__(model_name_or_path=model_name, **kwargs)
        typical_batchsizes = [1, 4, 16, 64, 128, 256, 1024]
        self.decoding_seqlens = list(set([*typical_batchsizes, *seqlens, self.max_len]))
        self.decoding_seqlens.sort()
        with utils.Timing(synchronize=True) as t:
            self.initialize_cuda_graph(self.decoding_seqlens)
        logger.info(f"InferenceEnginePadded graph init complete in {t.elapsed:.3f} sec.")

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: torch.LongTensor = None,
    ):
        for limit in self.callables.keys():  # finding lowest limit not less than b
            if input_ids.shape[-1] <= limit:
                break
        # print(f"Padded amask shape {attention_mask.shape}")
        static_input_ids = torch.full((1, limit), 0, dtype=torch.long, device=self.device)
        static_attn_mask = torch.zeros((1, 1, limit, self.max_len), dtype=self.dtype, device=self.device)
        static_position_ids = torch.full((1, limit), 0, dtype=torch.long, device=self.device)

        static_input_ids[:, : input_ids.shape[-1]] = input_ids
        static_attn_mask[:, :, : attention_mask.shape[-2], : attention_mask.shape[-1]] = attention_mask
        static_position_ids[:, : position_ids.shape[-1]] = position_ids
        static_storage_ids = F.pad(input=cache_position, pad=(0, limit - cache_position.shape[-1]), mode="constant", value=limit - 1)

        with utils.Timing(synchronize=(logger.level <= logging.DEBUG)) as t:
            logits = self.graph_inference(
                input_ids=static_input_ids,
                attention_mask=static_attn_mask,
                position_ids=static_position_ids,
                storage_ids=static_storage_ids,
            )
        logger.debug(f"graph_inference call {t.elapsed:.4f}  {input_ids.shape=} {limit=}")

        logits1 = logits[:, : input_ids.shape[-1], :]

        if False:  # testing logits match
            logits0 = self.inference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                storage_ids=cache_position,
            )
            k0 = logits0[0, :, :].topk(5, dim=-1)
            k1 = logits1[0, :, :].topk(5, dim=-1)
            # print(k0.values.cpu(), k1.values.cpu())
            # print(k0.indices.cpu(), k1.indices.cpu())
            if not torch.equal(k0.indices, k1.indices):
                pass
            # assert torch.equal(k0.indices, k1.indices)

        return logits1
