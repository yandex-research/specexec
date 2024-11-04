from typing import Dict, Optional, Tuple, Any  # noqa: F401

import numpy as np
import torch
import torch.nn.functional as F
import transformers

try:
    from EngineSequoia.Engine import InferenceEngine as InferenceEngineSequoia
except ModuleNotFoundError:
    pass
from specdec import utils
from tqdm import tqdm


class EngineRegular:
    """wrapper for regular transformers model with regular cache"""

    def __init__(self, model_name, max_len, dtype=torch.float16, device="cuda:0"):
        self.model_name = model_name
        self.max_len = max_len
        self.device = device
        if isinstance(model_name, str):
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
        else:
            self.model = model_name
        self.config = self.model.config
        self.kv_cache = DynamicCachePlus()

        # removing artifacts of StaticCache if previously used with the model
        for i, layer in enumerate(self.model.model.layers):
            if hasattr(layer.self_attn, "past_key_value"):
                delattr(layer.self_attn, "past_key_value")

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: torch.LongTensor = None,
    ):
        # assert torch.equal(cache_position, torch.arange(cache_position[0], cache_position[-1] + 1, device=cache_position.device)), "reconsider use of cache_position in amask slicing"
        attention_mask = attention_mask[..., : cache_position.max() + 1]
        assert attention_mask.shape[-2] == input_ids.shape[-1]

        cache_position_models = ["llama"]

        if self.model.config.model_type in cache_position_models:
            output = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.kv_cache,
                cache_position=cache_position,
            )
        else:
            assert torch.equal(cache_position, torch.arange(cache_position[0], cache_position[0] + cache_position.numel(), device=cache_position.device))

            output = self.model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=self.kv_cache,
            )

        self.kv_cache = output.past_key_values

        return output.logits

    @property
    def kv_len_used(self):
        if isinstance(self.kv_cache, transformers.DynamicCache):
            return self.kv_cache.get_seq_length()  # pass the call to DynamicCache
        else:  # if cache is in legacy form
            return 0 if self.kv_cache is None else self.kv_cache[0][0].shape[2]

    def clear_kv(self):
        self.kv_cache = DynamicCachePlus()

    def reorder_cache_tokens(self, source_token_idxs: torch.tensor, dest_token_idxs: torch.tensor = None):
        """Applies indices mask to KV cache or truncates it"""

        cache_size = self.kv_cache[0][0].shape[-2]  # replace with self.kv_len_used
        if source_token_idxs.dtype == torch.bool:
            source_token_idxs = torch.where(source_token_idxs)[0]

        left_edge = dest_token_idxs.min() if dest_token_idxs is not None else 0

        if source_token_idxs.max() >= cache_size:  # source includes elements outside of cache
            source_token_idxs = source_token_idxs[source_token_idxs < cache_size]
            dest_token_idxs = torch.arange(left_edge, left_edge + source_token_idxs.shape[-1], device=self.device)

        if dest_token_idxs is None:  # assumed that destination starts from cache beginning
            dest_token_idxs = torch.arange(source_token_idxs.shape[-1], device=self.device)

        new_cache = []
        for layer_cache_k, layer_cache_v in self.kv_cache:
            new_cache.append(
                (
                    torch.cat([layer_cache_k[:, :, :left_edge, :], layer_cache_k[:, :, source_token_idxs, :]], dim=-2),
                    torch.cat([layer_cache_v[:, :, :left_edge, :], layer_cache_v[:, :, source_token_idxs, :]], dim=-2),
                )
            )
        self.kv_cache = DynamicCachePlus.from_legacy_cache(tuple(new_cache))

    def set_max_len(self, new_max_len):
        if self.kv_cache is not None and self.kv_len_used > new_max_len:
            raise ValueError(f"Current cache size {self.kv_len_used()} is greater than new `max_len` {new_max_len}.")
        self.max_len = new_max_len


class EngineRegularDummy(EngineRegular):
    """special class for testing, with limited and controlled tokens probas"""

    sample_model_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"

    def __init__(self, model_name=None, probs=None, **kwargs):
        if model_name is None:
            model_name = transformers.AutoModelForCausalLM.from_pretrained(self.sample_model_name).to(0)
        super().__init__(model_name=model_name, **kwargs)

        self.logits = torch.ones(self.model.config.vocab_size, dtype=torch.float32, device=self.device) * torch.finfo(torch.float32).min

        probs = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]) if probs is None else probs
        self.logits = self.get_logits_from_probs(probs).to(self.device)

    def get_logits_from_probs(self, probs):
        """converts probs into logits; assumes probs tensors starts from index 0"""
        if probs.sum() != 1:
            probs = F.normalize(probs, dim=-1, p=1)
        probs[probs == 0] = torch.finfo(torch.float32).eps
        logits = torch.ones(self.model.config.vocab_size, dtype=torch.float32) * torch.finfo(torch.float32).min
        logits[: probs.shape[-1]] = torch.log(probs)
        logits[logits < -10] = torch.finfo(torch.float32).min
        assert torch.allclose(F.softmax(logits, dim=-1)[: probs.shape[-1]], probs, atol=1e-6)
        return logits

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: torch.LongTensor = None,
    ):
        logits = super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, cache_position=cache_position)
        adj_logits = self.logits.tile(*logits.shape[:-1], 1)
        return adj_logits


class EngineSequoiaIE:
    """wrapper for the original Sequoia InferenceEngine engine class"""

    def __init__(self, model_name, max_len, dtype=torch.float16, device="cuda:0"):

        self.i_engine = InferenceEngineSequoia(model_name_or_path=model_name, max_length=max_len, dtype=dtype, device=device)

        self.model_name = model_name
        self.max_len = max_len
        self.device = device
        self.model = self.i_engine.model
        self.config = self.model.config

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        cache_position: torch.LongTensor = None,
    ):
        attention_mask_inverted = (1 - attention_mask) * torch.finfo(attention_mask.dtype).min

        logits = self.i_engine.model_run(
            input_ids=input_ids,
            attention_mask=attention_mask_inverted,
            position_ids=position_ids,
            storage_ids=cache_position,
        )
        return logits

    def clear_kv(self):
        self.i_engine.clear_kv()

    @property
    def kv_len_used(self):
        return (self.i_engine.kv_cache.k_cache[0, 0, 0, :, 0] != 0).sum()

    def reorder_cache_tokens(self, source_token_idxs: torch.LongTensor, dest_token_idxs=None):
        """
        reorders cache along axis=2 (tokens)
        assumes batch_size==1  
        accepts int and bool indices
        if source_token_idxs.dtype == torch.bool assumes source_token_idxs.shape == self.key_cache.shape[2]
        resets setting {{ the rest of cache OR .key_cache[0, 0] }} to zeros to ensure that get_seq_length() works right
        """
        _cache = self.i_engine.kv_cache

        if dest_token_idxs is None:  # assumed that destination starts from cache beginning
            dest_size = source_token_idxs.sum() if source_token_idxs.dtype == torch.bool else source_token_idxs.shape[-1]
            dest_token_idxs = torch.arange(dest_size, device=self.device)

        if (source_token_idxs.dtype == torch.bool) and (source_token_idxs.shape[-1] < _cache.k_cache.shape[-2]):
            source_token_idxs = F.pad(input=source_token_idxs, pad=(0, _cache.k_cache.shape[-2] - source_token_idxs.shape[-1]), mode="constant", value=False)

        _cache.k_cache[:, :, :, dest_token_idxs.to(_cache.device), :] = _cache.k_cache[:, :, :, source_token_idxs.to(_cache.device), :]
        _cache.v_cache[:, :, :, dest_token_idxs.to(_cache.device), :] = _cache.v_cache[:, :, :, source_token_idxs.to(_cache.device), :]

        right_edge = torch.where(dest_token_idxs)[0].max() if dest_token_idxs.dtype == torch.bool else dest_token_idxs.max()

        # setting the rest of the cache to zeros to ensure that get_seq_length() works right
        _cache.k_cache[0, 0, 0, right_edge.to(_cache.device) :, :].zero_()


def benchmark_engine(engine, input_sizes=[256], repeats=5, max_len=1024):
    # repeats=10, min_pow=0, max_pow=13
    _max_len = getattr(engine, "max_len", max_len)

    with torch.inference_mode():
        result = {
            "model": engine.config._name_or_path,
            "device": torch.cuda.get_device_name(engine.device).replace("NVIDIA ", ""),
            "engine": engine.__class__.__name__,
            "max_len": _max_len,
        }

        _ = benchmark_single(engine, input_size=1, repeats=1)  # warmup

        pbar = tqdm(input_sizes, desc="benchmarking...")  # (total=(max_pow + 1) * repeats)
        stats = []
        for n in pbar:
            try:
                s = benchmark_single(engine, input_size=n, repeats=repeats, max_len=_max_len)
                # results.append({**stats, **common_stats})
                stats.append(s)
                pbar.desc = f"benchmarked {n=}"
            except RuntimeError:
                pass
        stats.sort(key=lambda x: x["size"])
        for entry in stats:
            result[entry["size"]] = entry["latency"]
        for entry in stats:
            result[f"m_{entry['size']}"] = entry["mem_use"]

    # df = pd.DataFrame(data=results, columns=["n", "t"])
    # df = df.groupby('n').agg('median')
    # stats = pd.concat([df.t, pd.Series(config_stats), pd.Series(mem_usage)])
    return result


def benchmark_single(engine, input_size, repeats, max_len=None):
    _max_len = getattr(engine, "max_len", max_len)
    timings = []
    torch.cuda.reset_peak_memory_stats()
    init_mem = torch.cuda.max_memory_allocated()
    attention_mask = torch.zeros(1, 1, input_size, _max_len, device=engine.device)
    _causal_mask = torch.tril(torch.ones(input_size, input_size, device=engine.device))
    attention_mask[..., :input_size, :input_size] = _causal_mask
    position_ids = torch.arange(input_size, device=engine.device).unsqueeze(0)
    cache_position = torch.arange(input_size, device=engine.device)

    for r in range(repeats):
        input_ids = torch.randint(2, 10000, (1, input_size), device=engine.device)
        with utils.Timing(synchronize=True) as t:
            _ = engine.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
            )
        timings.append(t.elapsed)
        # pbar.update()
    mem_use = (torch.cuda.max_memory_allocated() - init_mem) / 2**30
    stats = dict(
        size=input_size,
        latency=round(np.median(timings) * 1000, 3),
        mem_use=round(mem_use, 3),
    )
    return stats


# -------------------  CACHES  -------------------ß


class DynamicCachePlus(transformers.DynamicCache):
    """A better version of DynamicCache with persistence and reorder"""

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if (cache_kwargs is not None) and ("cache_position" in cache_kwargs) and (len(self.key_cache) > layer_idx):
            cache_position = cache_kwargs["cache_position"]
            pad_size = cache_position.max().item() + 1 - self.key_cache[layer_idx].shape[-2]
            self.key_cache[layer_idx] = F.pad(input=self.key_cache[layer_idx], pad=(0, 0, 0, pad_size), mode="constant", value=0)
            self.key_cache[layer_idx][:, :, cache_position, :] = key_states
            self.value_cache[layer_idx] = F.pad(input=self.value_cache[layer_idx], pad=(0, 0, 0, pad_size), mode="constant", value=0)
            self.value_cache[layer_idx][:, :, cache_position, :] = value_states

        elif len(self.key_cache) <= layer_idx:
            # called from from_legacy_cache() - assumes empty cache
            cache_position = torch.arange(key_states.shape[-2])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)

        else:
            # expanding cache by number of extra positionssss
            transformers.DynamicCache.update(
                self,
                key_states=key_states,
                value_states=value_states,
                layer_idx=layer_idx,
                cache_kwargs=cache_kwargs,
            )

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache_tokens(self, source_token_idxs: torch.tensor, dest_token_idxs: torch.tensor = None):
        """Applies indices mask to KV cache or truncates it"""

        cache_shape = self.key_cache[0].shape

        if source_token_idxs.dtype == torch.bool:
            source_token_idxs = torch.where(source_token_idxs)[0]

        left_edge = dest_token_idxs.min() if dest_token_idxs is not None else 0

        if source_token_idxs.max() >= cache_shape[-2]:  # source includes elements outside of cache
            source_token_idxs = source_token_idxs[source_token_idxs < cache_shape[-2]]
            dest_token_idxs = torch.arange(left_edge, left_edge + source_token_idxs.shape[-1], device=self.device)

        if dest_token_idxs is None:  # assumed that destination starts from cache beginning
            dest_token_idxs = torch.arange(source_token_idxs.shape[-1], device=self.device)

        for layer_cache_k, layer_cache_v in zip(self.key_cache, self.value_cache):
            layer_cache_k = torch.cat([layer_cache_k[:, :, :left_edge, :], layer_cache_k[:, :, source_token_idxs, :]], dim=-2)
            layer_cache_v = torch.cat([layer_cache_v[:, :, :left_edge, :], layer_cache_v[:, :, source_token_idxs, :]], dim=-2)

    def to_legacy_cache(self):
        return self

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCachePlus":
        if isinstance(past_key_values, cls):
            return past_key_values
        else:
            return super().from_legacy_cache(past_key_values)
