import os
from itertools import chain

import safetensors
import torch
from accelerate.utils.modeling import set_module_tensor_to_device
from transformers.modeling_utils import find_submodule_and_param_name, get_checkpoint_shard_files
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME, cached_file  # noqa: F401


class Loader:
    """loads transformer style model on demand"""

    def __init__(self, pretrained_model_name_or_path):

        resolved_archive_file = cached_file(
            pretrained_model_name_or_path,
            SAFE_WEIGHTS_INDEX_NAME,
        )

        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
        )
        self.resolved_archive_file_dict = {os.path.basename(fn): fn for fn in resolved_archive_file}

        self.weight_map = sharded_metadata["weight_map"]
        # start_prefix = 'model.layers.'
        # self.offload_index = {k: v for k, v in weight_map.items() if k.startswith(start_prefix)}
        # self.offload_index = weight_map

    def __getitem__(self, item_name, device="cpu"):
        if item_name in self.weight_map:
            checkpoint_file = self.weight_map[item_name]
            archive_file = self.resolved_archive_file_dict[checkpoint_file]
        else:
            raise IndexError(f"item {item_name} not found")

        with safetensors.safe_open(archive_file, framework="pt", device=device) as f:
            loaded_tensor = f.get_tensor(item_name)
        return loaded_tensor

    def fill_layer(self, layer, layer_index, device=torch.device("cpu")):
        """loads weights into layer parameters and buffers"""
        layer_prefix = f"model.layers.{layer_index}."

        for k, v in chain(layer.named_parameters(), layer.named_buffers()):
            if layer_prefix + k in self.weight_map:
                loaded_tensor = self[layer_prefix + k]
                submodule, param_name = find_submodule_and_param_name(layer, k, start_prefix="")
                set_module_tensor_to_device(module=submodule, tensor_name=param_name, device=device, value=loaded_tensor)
