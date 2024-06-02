"""Wrapper for a module that puts all tensors into a shared storage; based on original work by dvmazur@ and lavawolfiee@"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
from itertools import chain

from specdec import utils

if "logger" not in globals():
    logger = utils.get_logger()


class ModuleWithStorage(nn.Module):
    """
    Wraps a module and puts all its parameters and buffers to a shared storage (torch.UntypedStorage).
    WARNING: this wrapper modifies the input module in-place so that it can no longer change device or be trained.
    """

    def __init__(self, module: nn.Module, offsets=None):
        super().__init__()
        self.offsets = self.get_storage_offsets(module) if offsets is None else offsets
        self.storage, self.module = self.put_on_storage_inplace_(module, offsets=self.offsets)

    @staticmethod
    def put_on_storage_inplace_(module: nn.Module, offsets: Dict[str, Tuple[int, int]]) -> Tuple[torch.UntypedStorage, nn.Module]:
        """Modify module so that every parameter and buffer is a pointer to a pre-allocated storage"""
        device = next(module.parameters()).device
        storage_size_bytes = max([x[-1] for x in offsets.values()])

        storage = torch.UntypedStorage(storage_size_bytes, device=device)
        logger.debug(f"storage created, size={storage_size_bytes:,} bytes")

        for name, x in chain(module.named_parameters(), module.named_buffers()):
            start, end = offsets[name]
            assert isinstance(x, torch.Tensor)
            storage_view = torch.as_tensor(storage[start:end], dtype=x.dtype, device=device).view(x.shape)
            storage_view.copy_(x)
            assert storage_view.data_ptr() == storage.data_ptr() + start
            x.data = storage_view  # <-- replace parameter/buffer with a pointer to storage

        logger.debug("storage filled")
        for k, v in module.state_dict().items():
            assert storage.data_ptr() <= v.data_ptr() <= storage.data_ptr() + storage.nbytes(), k
        return storage, module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def __getattr__(self, name: str) -> torch.Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    @staticmethod
    def get_storage_offsets(module):
        offsets = {}
        edge_offset = 0

        for name, module in chain(module.named_parameters(), module.named_buffers()):
            assert isinstance(module, torch.Tensor)
            offsets[name] = (edge_offset, edge_offset + module.nbytes)
            edge_offset += module.nbytes

        return offsets
