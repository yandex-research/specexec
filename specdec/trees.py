"""
Better tree. Version 4b
"""

import logging
import torch
import torch.nn.functional as F

from typing import Dict, Optional  # noqa: F401
from specdec import utils

# global constats to address Tree.data rows
TOKENS = 0
POSITIONS = 1
PARENTS = 2
STATUS = 3

if "logger" not in globals():
    logger = utils.get_logger()


class Tree:

    def __init__(self, prefix_tokens, draft_engine, tokenizer=None):
        self.device = draft_engine.device
        self.end = 0
        self.engine = draft_engine
        self.engine.clear_kv()
        self.tokenizer = tokenizer
        self.max_len = self.engine.max_len

        # codes for the token states
        self.PROMPT = torch.tensor(10, device=self.device)
        self.GENERATED = torch.tensor(15, device=self.device)
        self.PROCESSED = torch.tensor(20, device=self.device)
        self.CANDIDATE = torch.tensor(30, device=self.device)

        # create empty tensors for the tree data
        self.data = torch.zeros(4, self.max_len, dtype=torch.int64, device=self.device)
        self.log_probs = torch.zeros(self.max_len, dtype=torch.float32, device=self.device)
        self.amask = torch.zeros((1, 1, self.max_len, self.max_len), dtype=torch.float16, device=self.device)
        self.q_probs: Dict[int, torch.tensor] = {}  # used by SpecInfer and related methods

        # fill the tree tensors with prefix data
        if isinstance(prefix_tokens, list):
            prefix_tokens = torch.tensor(prefix_tokens, device=self.device)
        self.prefix_len = prefix_tokens.flatten().shape[-1]
        self.data[TOKENS, : self.prefix_len] = prefix_tokens.flatten()
        self.data[POSITIONS, : self.prefix_len] = torch.arange(self.prefix_len, device=self.device)
        self.data[PARENTS, 1 : self.prefix_len] = torch.arange(self.prefix_len - 1, device=self.device)  # first token parent idx stays zero
        self.data[STATUS, : self.prefix_len - 1] = self.PROMPT
        self.data[STATUS, self.prefix_len - 1] = self.CANDIDATE  # last token is the candidate for the next children generation
        self.end = self.prefix_len
        self.reset_amask_to_prefix_len(self.prefix_len)

    def reset_amask_to_prefix_len(self, px_len=None):
        px_len = self.prefix_len if px_len is None else px_len
        self.amask.zero_()
        _causal_mask = torch.tril(torch.ones(px_len, px_len, dtype=self.amask.dtype, device=self.amask.device))
        self.amask[..., :px_len, :px_len] = _causal_mask

    @torch.inference_mode()
    def process_candidates(self, lim=None, pick_best=False):

        candidate_idxs = torch.where(self.data[STATUS, : self.end] == self.CANDIDATE)[0]
        assert candidate_idxs.numel() > 0, "this tree has no valid candidates left"

        if lim is not None:
            if pick_best and candidate_idxs.numel() > lim:
                cprobs = self.log_probs[candidate_idxs]
                top_lim_indices = cprobs.topk(k=lim, sorted=False).indices
                candidate_idxs = candidate_idxs[top_lim_indices]
                candidate_idxs, _ = candidate_idxs.sort()
            else:
                candidate_idxs = candidate_idxs[:lim]

        if self.engine.kv_len_used < candidate_idxs.min():
            input_idxs = torch.cat([torch.arange(self.engine.kv_len_used, candidate_idxs.min(), device=self.device), candidate_idxs])
        else:
            input_idxs = candidate_idxs

        input_ids = self.data[TOKENS, input_idxs].unsqueeze(0)
        position_ids = self.data[POSITIONS, input_idxs].unsqueeze(0)
        attention_mask = self.amask[..., input_idxs, :]

        with utils.Timing(synchronize=(logger.level <= logging.DEBUG)) as t:
            logits = self.engine.forward(
                input_ids=input_ids,
                attention_mask=self.invert_mask(attention_mask, dtype=self.engine.model.dtype),
                position_ids=position_ids,
                cache_position=input_idxs,
            )
        logger.debug(f"Draft Engine forward() call {t.elapsed:.4f}  {input_ids.shape=}")

        self.data[STATUS, candidate_idxs] = self.PROCESSED
        beam_scores = self.log_probs[candidate_idxs]
        beam_positions = self.data[POSITIONS, candidate_idxs]
        out_logits = logits[0, -candidate_idxs.shape[-1] :, :]

        return (
            out_logits,
            candidate_idxs,
            beam_scores,
            beam_positions,
        )  # assumes that the needed logits are in the end of the logits tensor

    def add(self, token_ids, positions, parent_idxs, log_probs, new_status=None):
        """adds new tokens with attributes to the tree"""

        new_status = self.CANDIDATE if new_status is None else new_status

        add_size = token_ids.numel()

        if add_size > self.max_len - self.end:
            raise ValueError(f"required addition size {add_size} exceeds remaining tree capacity of {self.max_len - self.end}.")
        assert add_size == positions.numel() == parent_idxs.numel() == log_probs.numel()

        self.data[TOKENS, self.end : self.end + add_size] = token_ids.view(-1)
        self.data[POSITIONS, self.end : self.end + add_size] = positions.view(-1)
        self.data[PARENTS, self.end : self.end + add_size] = parent_idxs.view(-1)
        self.data[STATUS, self.end : self.end + add_size] = new_status

        self.log_probs[self.end : self.end + add_size] = log_probs.view(-1)

        amask_draft = self.amask[..., parent_idxs.view(-1), : self.end]
        new_amask_eye = torch.eye(add_size, device=self.device)[None, None, ...]
        amask_draft = torch.cat((amask_draft, new_amask_eye), dim=-1)
        self.amask[..., self.end : self.end + add_size, : self.end + add_size] = amask_draft

        self.end += add_size

    @property
    def size(self):
        return self.end

    @property
    def h(self):
        return (self.data[POSITIONS].max() - self.prefix_len + 1).item()

    @property
    def tokens(self):
        return self.data[TOKENS, : self.end]

    @property
    def positions(self):
        return self.data[POSITIONS, : self.end]

    @property
    def parents(self):
        return self.data[PARENTS, : self.end]

    @property
    def status(self):
        return self.data[STATUS, : self.end]

    def draw(self, start=None, tokenizer=None, add_prob=False, return_repr=False):
        from anytree import Node, RenderTree

        tokenizer = self.tokenizer if tokenizer is None else tokenizer

        if self.size == 0:
            return "[ ! empty_tree ! ]"
        if start is None:
            start = self.prefix_len - 1

        indices = torch.arange(start, self.end)
        parents = self.data[PARENTS, indices]

        def get_token_repr(idx):
            if tokenizer:
                out = tokenizer.convert_ids_to_tokens(self.data[TOKENS, idx].item())
            else:
                out = str(self.data[TOKENS, idx].item())
            if add_prob:
                out += f" ({self.log_probs[idx]:.2f})"
            return out

        tokens = [get_token_repr(i) for i in indices]
        nodes = [None] * len(tokens)

        # Iterate through the tokens and parents to build the tree
        for i, (idx, token, parent_idx) in enumerate(zip(indices, tokens, parents)):
            if i == 0:  # Root node
                nodes[i] = Node(token)
            else:  # Non-root node
                nodes[i] = Node(token, parent=nodes[parent_idx - start])
        # Print the tree
        tree_repr = [f"{pre}{node.name}" for (pre, fill, node) in RenderTree(nodes[0])]
        tree_repr = "\n".join(tree_repr)
        if return_repr:
            return tree_repr
        else:
            print(tree_repr)

    def trim_budget(self, *, budget=None, min_log_prob=None):
        if budget is not None:
            source_idxs = self.log_probs[self.prefix_len : self.end].topk(budget, sorted=False).indices + self.prefix_len
        elif min_log_prob is not None:
            source_idxs = torch.where(self.log_probs[self.prefix_len : self.end] >= min_log_prob)[0] + self.prefix_len
        else:
            raise ValueError("no argument provided to `Tree.trim_budget()`")

        dest_idxs = torch.arange(self.prefix_len, self.prefix_len + source_idxs.shape[-1], device=self.device)
        return self.gather_tree(source_idxs, dest_idxs)

    def gather_tree(self, source_idxs, dest_idxs):

        if source_idxs.dtype == torch.bool:
            source_idxs = torch.where(source_idxs)[0]
        if source_idxs.numel() == 0:
            return

        interim_idx = torch.arange(self.end, device=self.device)  # helper index for parents indices conversion
        interim_idx[source_idxs] = dest_idxs
        self.data[PARENTS, source_idxs] = interim_idx[self.data[PARENTS, source_idxs]]

        self.data[:, dest_idxs] = self.data[:, source_idxs]
        self.log_probs[dest_idxs] = self.log_probs[source_idxs]
        self.amask[:, :, dest_idxs, :] = self.amask[:, :, source_idxs, :]
        self.amask[:, :, :, dest_idxs] = self.amask[:, :, :, source_idxs]

        self.end = dest_idxs.max().item() + 1

        self.amask[:, :, self.end :, :].zero_()  # zeroing unused mask areas
        self.amask[:, :, :, self.end :].zero_()

        self.engine.reorder_cache_tokens(source_token_idxs=source_idxs, dest_token_idxs=dest_idxs)
        return interim_idx

    def set_max_len(self, new_max_len):
        if self.end > new_max_len:
            raise ValueError(f"Tree resize failed. Current tree used size {self.end} is greater than new `max_len` {new_max_len}.")
        if new_max_len > self.max_len:
            pad = new_max_len - self.max_len
            self.data = F.pad(input=self.data, pad=(0, pad), mode="constant", value=0)
            self.log_probs = F.pad(input=self.log_probs, pad=(0, pad), mode="constant", value=0.0)
            self.amask = F.pad(input=self.amask, pad=(0, pad, 0, pad), mode="constant", value=0.0)
        elif new_max_len < self.max_len:
            self.data = self.data[..., :new_max_len]
            self.log_probs = self.log_probs[..., :new_max_len]
            self.amask = self.amask[..., :new_max_len, :new_max_len]

        self.max_len = new_max_len
        self.engine.set_max_len(new_max_len)

    def reset_to_sequence(self, sequence_mask, target_engine=None):
        assert torch.all(sequence_mask[: self.prefix_len]), "Prefix is not fully retained"
        sequence_idxs = torch.where(sequence_mask != 0)[0]

        # truncate caches
        if target_engine:
            target_engine.reorder_cache_tokens(sequence_mask)
        self.engine.reorder_cache_tokens(sequence_mask[: self.engine.kv_len_used])  # note: draft model's KV cache is shorter

        new_len = sum(sequence_mask).item()
        if torch.any(sequence_mask[self.prefix_len :]):
            source_idxs = sequence_idxs[sequence_idxs >= self.prefix_len]
            dest_idxs = torch.arange(self.prefix_len, new_len, device=self.device)
            self.data[TOKENS, dest_idxs] = self.data[TOKENS, source_idxs]
            self.data[POSITIONS, dest_idxs] = dest_idxs
            self.data[PARENTS, dest_idxs] = dest_idxs - 1
            self.data[STATUS, dest_idxs] = self.GENERATED

        self.end = new_len
        self.prefix_len = self.end
        self.data[STATUS, : self.prefix_len] = self.PROMPT
        self.log_probs[self.end :].zero_()
        self.data[:, self.end :].zero_()
        self.reset_amask_to_prefix_len()
        self.q_probs = {}
        return

    @torch.inference_mode()
    def get_best_min_prob(self, budget):
        # gets prob limit based on tree and heap together. returns the lowest negative if budget not reached.
        if self.end - self.prefix_len < budget:
            return torch.finfo(self.log_probs.dtype).min
        else:
            min_prob = self.log_probs.topk(budget).values.min()
            return min_prob

    @torch.inference_mode()
    def get_expected_contrib(self, decay_factor=0.95):
        """math expectation of num accepted tokens"""
        cum_probs = torch.exp(self.log_probs[self.prefix_len : self.end])
        new_positions = self.positions[self.prefix_len :] - self.prefix_len + 1
        decay_tensor = torch.pow(decay_factor, new_positions)

        return (cum_probs * decay_tensor).sum()

    @staticmethod
    def invert_mask(mask, dtype=torch.float32):
        if mask.min() == 0:
            min_dtype = torch.finfo(dtype).min
            mask = (mask.eq(0.0)).to(dtype=dtype) * min_dtype
        return mask
