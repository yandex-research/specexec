# from copy import deepcopy

import copy

import accelerate
import optimum
import safetensors
import torch
import transformers
from auto_gptq.modeling._utils import autogptq_post_init
from auto_gptq.utils.exllama_utils import exllama_set_max_input_length
from optimum.gptq import GPTQQuantizer
from optimum.gptq.quantizer import ExllamaVersion
from optimum.gptq.utils import get_device
from tqdm.auto import trange

# from transformers.modeling_utils import _add_variant, cached_file, SAFE_WEIGHTS_NAME, SAFE_WEIGHTS_INDEX_NAME, get_checkpoint_shard_files  # noqa: F401
from transformers.modeling_utils import _load_state_dict_into_meta_model
from transformers.utils.quantization_config import GPTQConfig

from offloading.model_loader import Loader
from offloading.offload_engine import OffoadingCache
from offloading.storage_wrapper import ModuleWithStorage

from specdec import utils

if "logger" not in globals():
    logger = utils.get_logger()


def load_offloaded_model(model_name, device_size=3, main_device=torch.device("cuda"), main_dtype=torch.float16):

    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.layers": "meta",
        "model.norm": "cuda:0",
        "lm_head": "cuda:0",
        "model.rotary_emb": "cuda:0",
    }

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, low_cpu_mem_usage=True, torch_dtype=torch.float16)

    # load config
    model_config = transformers.AutoConfig.from_pretrained(model_name)
    # model_config = transformers.LlamaModel._autoset_attn_implementation(model_config)  # fix config._attn_implementation  # IS IT NEEDED W/O GPTQ ?
    # should be 'sdpa' if enabled / available, see model._check_and_enable_sdpa(...) also transformers.utils.is_torch_sdpa_available()

    loader = Loader(model_name)

    def make_module():
        with torch.device(main_device):
            original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float16)
            module = type(model.model.layers[0])(model_config, layer_idx=0)
            torch.set_default_dtype(original_dtype)

        # module = deepcopy(sample_layer)
        module.layer_idx = None
        return ModuleWithStorage(module.to(device=main_device, dtype=main_dtype))

    with utils.Timing(synchronize=True) as t:
        cache = OffoadingCache(make_module, device_size=device_size)  # <-- keep :device_size: modules on device
    logger.debug(f"OffoadingCache created in {t.elapsed:.6f}")

    # offsets, storage_size_bytes = cache.get_storage_offsets()
    offsets = cache.all_device_buffers[0].offsets
    logger.debug("offsets, storage_size_bytes", offsets)

    for layer_idx in trange(model.config.num_hidden_layers, desc="Filling CPU modules storage"):
        logger.debug(f"start loading layer {layer_idx} {'.' * 48}")
        with accelerate.init_empty_weights():
            layer_ = type(model.model.layers[0])(model_config, layer_idx)
        logger.debug(f"created empty layer {layer_idx}")

        loader.fill_layer(layer_, layer_idx)
        logger.debug(f"filled layer {layer_idx}")

        module = ModuleWithStorage(layer_.to(dtype=main_dtype), offsets=offsets)
        logger.debug(f"MWS layer {layer_idx}")

        cache.add_module(uid=layer_idx, module=module)
        logger.debug(f"added layer {layer_idx}")

        del layer_
        logger.debug(f"done with layer {layer_idx}")

    model.model.layers = OffloadedLayerIter(cache)

    return model


class OffloadedLayerIter(torch.nn.Module):
    def __init__(self, cache):
        super().__init__()
        self.cache = cache  # this module owns cache

    def __iter__(self):
        for layer_idx, module in self.cache.load_modules(*range(len(self.cache.offloaded_storages))):
            module.module.self_attn.layer_idx = layer_idx
            yield module

    def __getitem__(self, idx):
        return self.cache.all_device_buffers[idx]


def load_transformer_layer_gptq(
    layer_idx: int, model_file_path, model_config, quantizer, device: torch.device
) -> transformers.models.llama.modeling_llama.LlamaDecoderLayer:
    """
    Loads a single Llama layer from a model checkpoint in GPTQ format
    returns layer, that still requires post_init() call
    """

    layer_prefix = f"model.layers.{layer_idx}."

    # Load tensors whose keys start with the layer_prefix
    loaded_layer_state_dict = {}
    with safetensors.safe_open(model_file_path, framework="pt", device=device) as f:
        for key in f.keys():
            if key.startswith(layer_prefix):
                loaded_layer_state_dict[key] = f.get_tensor(key)

    # get empty Llama layer
    with torch.device(device):
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)

        if model_config.model_type == "llama":
            layer_class = transformers.models.llama.modeling_llama.LlamaDecoderLayer
        elif model_config.model_type == "mixtral":
            layer_class = transformers.models.mixtral.modeling_mixtral.MixtralDecoderLayer

        with accelerate.init_empty_weights():
            layer1 = layer_class(model_config, layer_idx)

        torch.set_default_dtype(original_dtype)

    # convert the layer to GPTQ
    sublayers_to_be_replaced = optimum.gptq.utils.get_layers(layer1)

    # config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    if hasattr(model_config, "quantization_config"):
        modules_in_block_to_quantize = model_config.quantization_config.modules_in_block_to_quantize

        if modules_in_block_to_quantize is not None:
            layers_to_keep = sum(modules_in_block_to_quantize, [])
            for name in list(sublayers_to_be_replaced.keys()):
                if not any(name.endswith(layer) for layer in layers_to_keep):
                    # print(f"Quantization disabled for {name} (only modules_in_block_to_quantize={modules_in_block_to_quantize} are quantized)")
                    del sublayers_to_be_replaced[name]

    if quantizer is not None:
        quantizer._replace_by_quant_layers(layer1, names=sublayers_to_be_replaced)

    # select layer's items from the state_dict and fix prefices
    loaded_layer_state_dict = {k.replace(layer_prefix, ""): v for k, v in loaded_layer_state_dict.items() if k.startswith(layer_prefix)}

    # insert the loaded tensors into their places
    new_error_msgs, offload_index, state_dict_index = _load_state_dict_into_meta_model(
        model=layer1,
        state_dict=loaded_layer_state_dict,
        loaded_state_dict_keys=list(loaded_layer_state_dict.keys()),
        start_prefix="",
        expected_keys=list(layer1.state_dict().keys()),
        device_map={"": device},
        unexpected_keys=None,  # passing `unexpected` for cleanup from quantization items
    )
    return layer1


# GPTQ-related code:


def load_gptq_offloaded_model(model_name, device_size=1, main_device=torch.device("cuda:0"), max_input_length=None):

    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.layers": "meta",
        "model.norm": "cuda:0",
        "lm_head": "cuda:0",
    }

    # load config
    config = transformers.AutoConfig.from_pretrained(model_name)
    config = transformers.LlamaModel._autoset_attn_implementation(config)  # fix config._attn_implementation
    # should be 'sdpa' if enabled / available, see model._check_and_enable_sdpa(...) also transformers.utils.is_torch_sdpa_available()

    assert (
        hasattr(config, "quantization_config") and config.quantization_config is not None
    ), "this  model is not quantized, use different function for offloaded model creation"
    q_config_dict = config.quantization_config
    if not isinstance(q_config_dict, dict):
        q_config_dict = q_config_dict.to_dict_optimum()
    q_config_dict["max_input_length"] = max_input_length
    q_config = GPTQConfig.from_dict(q_config_dict)
    config.quantization_config = q_config

    # load model with layers to meta, use patch to postpone post_init q4 creation
    post_init_model_original = optimum.gptq.quantizer.GPTQQuantizer.post_init_model
    optimum.gptq.quantizer.GPTQQuantizer.post_init_model = post_init_model_patched  # PATCH
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, low_cpu_mem_usage=True, quantization_config=q_config)
    optimum.gptq.quantizer.GPTQQuantizer.post_init_model = post_init_model_original  # RESTORE

    quantizer = GPTQQuantizer.from_dict(q_config_dict)

    main_dtype = next(model.model.layers[0].parameters()).dtype

    resolved_archive_file = transformers.utils.hub.cached_file(model_name, filename="model.safetensors")

    # layer template for make_module()
    sample_layer = load_transformer_layer_gptq(layer_idx=0, model_file_path=resolved_archive_file, model_config=config, quantizer=quantizer, device="cpu")

    def make_module():
        module = copy.deepcopy(sample_layer)  # noqa  # no need to deepcopy if only 1 layer
        module.layer_idx = None
        return ModuleWithStorage(module.to(device=main_device, dtype=main_dtype))

    cache = OffoadingCache(make_module, device_size=device_size)  # <-- keep :device_size: modules on device
    del sample_layer

    for layer_idx in trange(model.config.num_hidden_layers):
        layer = load_transformer_layer_gptq(layer_idx=layer_idx, model_file_path=resolved_archive_file, model_config=config, quantizer=quantizer, device="cpu")
        module = ModuleWithStorage(layer.to(dtype=main_dtype))
        cache.add_module(uid=layer_idx, module=module)
        del module
        del layer

    # complete post_init(), create q4 objects for loaded_device_module_buffers
    for module, *_ in cache.loaded_device_module_buffers:
        module.module = quantizer.post_init_model(module.module)

    model.model.layers = OffloadedLayerIter(cache)

    return model


def post_init_model_patched(self, model):
    """
    # monkey patch to postpone q4 creation
    based on optimum.gptq.quantizer.GPTQQuantizer.post_init_model
    version for cpu
    """
    # print(f"{self.bits=}, {self.disable_exllama=}")
    if self.bits == 4 and not self.disable_exllama:
        patched_mode = True
        if get_device(model) == torch.device("cpu") or (hasattr(model, "hf_device_map") and any(d in model.hf_device_map for d in ["cpu", "disk"])):
            # raise ValueError(
            #     "Found modules on cpu/disk. Using Exllama or Exllamav2 backend requires all the modules to be on GPU."
            #     "You can deactivate exllama backend by setting `disable_exllama=True` in the quantization config object"
            # )
            patched_mode = True

    class StoreAttr(object):
        pass

    model.quantize_config = StoreAttr()
    model.quantize_config.desc_act = self.desc_act
    if not patched_mode:
        model = autogptq_post_init(model, use_act_order=self.desc_act)
    if (
        self.desc_act
        and (not self.disable_exllama and self.exllama_version == ExllamaVersion.ONE)
        and (self.max_input_length is not None)
        and getattr(model.config.quantization_config, "act_order", False)
        and (self.max_input_length is not None and self.max_input_length > model.config.max_length)
    ):
        model = exllama_set_max_input_length(model, self.max_input_length)
    return model
