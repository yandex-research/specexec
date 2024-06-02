import torch
import transformers
from tqdm import tqdm
from tqdm.auto import trange
from copy import deepcopy

from offloading.offload_engine import OffoadingCache
from offloading.storage_wrapper import ModuleWithStorage


def test_wrapper():
    model_name = "meta-llama/Llama-2-7b-hf"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype='auto', device_map='cpu')

    main_device = torch.device('cuda')
    main_dtype = next(model.model.layers[0].parameters()).dtype

    def make_module():
        module = deepcopy(model.model.layers[0])
        module.layer_idx = None
        return ModuleWithStorage(module.to(device=main_device, dtype=main_dtype))

    cache = OffoadingCache(make_module, device_size=3)  # <-- keep :device_size: modules on device
    for layer_idx in trange(model.config.num_hidden_layers, desc='Populating offloaded buffer'):
        module = ModuleWithStorage(deepcopy(model.model.layers[layer_idx]).to(dtype=main_dtype))
        cache.add_module(uid=layer_idx, module=module)

    with torch.no_grad():
        for i in range(10):
            x = torch.randn(1, 4, 4096, dtype=main_dtype, device=main_device)
            y_ref = x
            for layer_idx in trange(model.config.num_hidden_layers, desc="Naive offloading"):
                layer = deepcopy(model.model.layers[layer_idx]).to(dtype=main_dtype, device=main_device)
                y_ref, = layer(y_ref)

            y = x
            for layer_idx, module in tqdm(cache.load_modules(*range(model.config.num_hidden_layers)),
                                          total=model.config.num_hidden_layers, desc='Buffered offloading'):
                assert module.module.layer_idx is None
                module.module.layer_idx = layer_idx
                y, = module(y)
                module.module.layer_idx = None
            assert torch.allclose(y, y_ref)
