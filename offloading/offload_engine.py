"""Engine for speculative offloading; based on original work by dvmazur@ and lavawolfiee@"""
import sys
from collections import deque
# from itertools import chain
from typing import Any, Callable, Deque, Dict, Iterator, Optional, Tuple

import torch

from offloading.storage_wrapper import ModuleWithStorage

from specdec import utils
if "logger" not in globals():
    logger = utils.get_logger()

ModuleUID = Any


class OffoadingCache(torch.nn.Module):
    def __init__(self, make_device_module: Callable[[], ModuleWithStorage], device_size: int):
        """
        A memory manager that dynamically loads an array of modules with identical hyperparameters
        :param make_device_module: a function that creates a new module instance on the main device (e.g. cuda)
          - each call to make_module must return a new instance of module; cannot reuse pre-existing ones
          - the device of the created module will dictate the device of this cache
        :param device_size: number of modules that can be loaded to CUDA simultaneously
        """
        super().__init__()
        assert device_size > 0
        self.module_type = self.module_size = self.device = None
        self.active = False
        self.all_device_buffers = torch.nn.ModuleList([self._check_module(make_device_module()) for _ in range(device_size)])
        self.loaded_device_module_buffers: Deque[Tuple[ModuleWithStorage, Optional[ModuleUID], torch.cuda.Stream]] = deque(
            [(module_buffer, None, torch.cuda.Stream()) for module_buffer in self.all_device_buffers]
        )
        self.offloaded_storages: Dict[ModuleUID, torch.UntypedStorage] = {}
        assert self.module_size is not None

    # def get_storage_offsets(self):
    #     offsets = {}
    #     edge_offset = 0

    #     for name, module in chain(self.all_device_buffers[0].named_parameters(), self.all_device_buffers[0].named_buffers()):
    #         assert isinstance(module, torch.Tensor)
    #         offsets[name] = (edge_offset, edge_offset + module.nbytes)
    #         edge_offset += module.nbytes

    #     return offsets, edge_offset

    def _check_module(self, module: ModuleWithStorage):
        assert isinstance(module.storage, torch.UntypedStorage)
        if self.module_type is None:
            self.module_type = type(module)
            self.module_size = len(module.storage)
            self.device = module.storage.device
        else:
            assert isinstance(module, self.module_type)
            assert len(module.storage) == self.module_size
            assert module.storage.device == self.device
        return module

    def add_module(self, uid: ModuleUID, module: ModuleWithStorage):
        """Register a new module to the cache and associate it with uid"""
        assert self.module_type is not None
        assert isinstance(module, self.module_type)
        return self.add_module_by_storage(uid, module.storage)

    def add_module_by_storage(self, uid: ModuleUID, storage: torch.UntypedStorage):
        """Register a new module defined by its storage"""
        assert uid is not None, "uid cannot be None"
        assert uid not in self.offloaded_storages, f"Module UID {uid} already registered"
        assert isinstance(storage, torch.UntypedStorage)
        assert len(storage) == self.module_size
        self.offloaded_storages[uid] = storage.cpu().pin_memory(self.device)

    def load_modules(self, *uids_to_load: ModuleUID) -> Iterator[Tuple[ModuleUID, ModuleWithStorage]]:
        """
        Loads modules with the specified UIDs, returns an iterator over requested modules.
        :note: The loaded module is only valid for one iteration, i.e. before the next module is loaded.
        :param uids_to_load: iterate over the specified module uids. Same uids as in add_module
        :returns: an iterator that yields (layer_uid, module) pairs, only usable inside the for loop
        :example:
        >>> for layer_index, layer in module_cache.load_modules(range(num_layers)):
        >>>     hidden_states, = layer(hidden_states)
        """
        assert len(set(uids_to_load)) == len(uids_to_load)
        assert not self.active, "already loading experts; buffers are busy"

        try:
            self.active = True
            index_of_next_uid_to_load = 0  # index of the next info that is supposed to be loaded
            for _ in range(len(self.loaded_device_module_buffers)):
                loaded_module_buffer, loaded_uid, load_stream = self.loaded_device_module_buffers.popleft()
                next_uid_to_load = uids_to_load[index_of_next_uid_to_load]
                if loaded_uid != next_uid_to_load:
                    with torch.cuda.stream(load_stream):
                        loaded_module_buffer.storage.copy_(self.offloaded_storages[next_uid_to_load], non_blocking=True)
                self.loaded_device_module_buffers.append((loaded_module_buffer, next_uid_to_load, load_stream))
                index_of_next_uid_to_load += 1

            for uid_to_yield in uids_to_load:
                loaded_module_buffer, loaded_uid, load_stream = self.loaded_device_module_buffers.popleft()
                assert loaded_uid == uid_to_yield
                torch.cuda.default_stream(self.device).wait_stream(load_stream)
                yield uid_to_yield, loaded_module_buffer

                next_uid_to_load = uids_to_load[index_of_next_uid_to_load % len(uids_to_load)]
                #  ^-- note: when index_of_next_uid_to_load >= len(uids_to_load), this will pre-load first few layers
                load_stream.wait_stream(torch.cuda.default_stream(self.device))
                with torch.cuda.stream(load_stream):
                    loaded_module_buffer.storage.copy_(self.offloaded_storages[next_uid_to_load], non_blocking=True)
                self.loaded_device_module_buffers.append((loaded_module_buffer, next_uid_to_load, load_stream))
                index_of_next_uid_to_load += 1
        except Exception as e:
            print(e)
            pass
        finally:
            if len(self.loaded_device_module_buffers) != len(self.all_device_buffers):  # error handling
                print(f"Recovering {len(self.all_device_buffers) - len(self.loaded_device_module_buffers)} buffers", file=sys.stderr)
                self.loaded_device_module_buffers: Deque[Tuple[ModuleWithStorage, Optional[ModuleUID]]] = deque(
                    [(module_buffer, None, torch.cuda.Stream()) for module_buffer in self.all_device_buffers]
                )
            self.active = False
