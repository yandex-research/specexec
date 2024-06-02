"""
SpecExec, version 4b
"""

import numpy as np
import torch
import logging
import math

from . import utils
from .spec_base import SpecBase
from .trees import TOKENS, POSITIONS, PARENTS, STATUS  # noqa: F401

logger = utils.get_logger()


class SpecExecBase(SpecBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_engines(self, prefix_len, max_budget, max_new_tokens, **kwargs):
        super().reset_engines()
        safety_margin = 64  # account for warmup tokens and excess generation
        draft_max_len = prefix_len + max_new_tokens + max_budget + safety_margin + 256
        target_max_len = prefix_len + max_new_tokens + max_budget + safety_margin
        if hasattr(self, "tree"):
            self.tree.set_max_len(draft_max_len)
        else:
            self.draft_engine.set_max_len(draft_max_len)

        self.target_engine.set_max_len(target_max_len)
        logger.info(f"Max_len reset: {draft_max_len=}, {target_max_len=}")

    @torch.inference_mode()
    def grow_tree(self, max_budget, max_beam_len, max_n_beams=32, max_branch_width=16, min_log_prob=-10, decay_factor=0.95, **kwargs):
        """grows speculative tree
        Args:
            Engines, tokenizer,
            max_tokens (_type_): maximum tree size in tokens
            max_beam_length (_type_): number of growth iterations
            max_n_beams (int, optional): number of tokens considered at each iteration
            max_branch_width (int, optional): max number of children per branch.
            min_log_prob: limit for proba of added tokens. Defaults to -10.
        Returns:
            statistics and tree
        """

        logger.debug(f"=================  G R O W  {self.__class__.__name__}  ================================================== ")
        self.decay_factor_log = math.log(decay_factor)
        expected_ratios = []
        input_tokens_count = []  # for logging
        self.max_budget = max_budget
        pick_best = self.draft_engine.config.model_type in ["llama"]  # list models that support `cache_position` argument

        for next_position in range(self.tree.prefix_len, self.tree.prefix_len + max_beam_len):
            logger.debug(f"{next_position=} ----------------------- {self.tree.end=}")

            draft_logits, parent_indices, parent_scores, parent_positions = self.tree.process_candidates(lim=max_n_beams, pick_best=pick_best)

            logger.debug(f"after process_candidates {draft_logits.shape=}, {self.tree.end=}")

            best_child_token_ids, best_child_positions, best_parent_idxs, cum_beam_scores = self.get_next_beams(
                draft_logits,
                parent_pos=parent_positions,
                parent_idxs=parent_indices,
                beam_scores=parent_scores,
                num_beams=max_budget,
                decay_factor_log=self.decay_factor_log,
                max_branch_width=max_branch_width,
            )

            input_tokens_count.append(best_child_token_ids.numel())

            if best_child_token_ids.shape[-1] == 0:  # no new beams
                logger.debug("no children offered")
                # break if no child generated (possibly due to min_log_prob too high for the budget)
                break
            # break if new tokens are not in top budget by log prob
            if self.tree.end - self.tree.prefix_len >= max_budget:
                lowest_tree_log_prob = self.tree.log_probs[self.tree.prefix_len : self.tree.end].topk(k=max_budget, dim=-1, sorted=False).values.min()
                best_new_log_prob = cum_beam_scores.max()
                if best_new_log_prob <= lowest_tree_log_prob:
                    logger.debug(f"early stop: pos {next_position}: best_new={best_new_log_prob:.2f} <= lowest_tree_prob={lowest_tree_log_prob:.2f}")
                    break
            self.tree.add(best_child_token_ids, best_child_positions, best_parent_idxs, cum_beam_scores, new_status=self.tree.CANDIDATE)

        if self.tree.end - self.tree.prefix_len > max_budget:
            logger.debug(f"Have to trim: Draft token count: {self.tree.end - self.tree.prefix_len} > max_budget {max_budget}")
            self.tree.trim_budget(budget=max_budget)

        stats = {
            "tree_w": np.unique(self.tree.positions.tolist(), return_counts=True)[1].max(),
            "tree_h": self.tree.positions.max().item() - self.tree.prefix_len + 1,
            "tree_size": self.tree.size - self.tree.prefix_len,  # tree size net of prefix len
            "input_len_0": sum(input_tokens_count),
            "draft_iters": next_position - self.tree.prefix_len + 1 if "next_position" in locals() else 0,
            "lowest_cum_log_prob": round(self.tree.log_probs[: self.tree.end].min().item(), 4),
        }
        logger.debug(f"input_tokens_count: {sum(input_tokens_count)}, {input_tokens_count}")
        logger.debug(
            f"tree layer sizes: {torch.unique(self.tree.positions[self.tree.prefix_len:], return_counts=True)[1].tolist()}"
        )  # Tree nodes counts by level
        logger.info(f"{stats}")

        return stats

    @torch.inference_mode()
    def validate_tree(self, temperature=1.0, top_p=1.0, **kwargs):
        """validation of the generated sequences with Target model"""
        logger.debug(f"=================  V A L I D A T E   {self.__class__.__name__}   ============================")
        target_token_map_bool = self.tree.status[: self.tree.end] >= self.tree.PROCESSED  # tokens generated in the current iteration
        target_token_map_bool[: self.tree.prefix_len] = False  # addresses problem of the last prefix token status
        target_token_idxs = torch.where(target_token_map_bool)[0]
        target_parent_idxs = self.tree.parents[: self.tree.end][target_token_map_bool]

        input_token_map_bool = target_token_map_bool.clone()  # tokens needed as target_engine forward inputs
        input_token_map_bool[target_parent_idxs] = True  # inputs for fwd
        if self.target_engine.kv_len_used == 0:
            input_token_map_bool[: self.tree.prefix_len] = True

        if self.tree.end > self.target_engine.max_len:
            logger.info(f"target_engine max_len expands from {self.target_engine.set_max_len} to {self.tree.end}")
            self.target_engine.set_max_len(self.tree.end)

        input_ids = self.tree.tokens[input_token_map_bool].unsqueeze(0)
        cache_position = torch.where(input_token_map_bool)[0]
        amask_target = self.tree.amask[:, :, cache_position, : self.target_engine.max_len]  # clipping to target max_len
        position_ids = self.tree.positions[input_token_map_bool].unsqueeze(0)
        logger.info(f"VAL {input_ids.shape=}, {amask_target.shape=}, {self.target_engine.kv_len_used=}, {self.tree.prefix_len=}, {self.tree.end=}")

        target_logits = self.target_engine.forward(
            input_ids=input_ids,
            attention_mask=self.tree.invert_mask(amask_target, dtype=self.target_engine.model.dtype),
            position_ids=position_ids,
            cache_position=cache_position,
        )
        target_logits = target_logits.squeeze(0)  # remove batch dim
        all_target_token_choices, all_target_token_logprobs = self.sampler_from_logits(logits=target_logits, temperature=temperature, top_p=top_p)

        # Matching target and draft choices to find the longest accepted sequence
        interim_t = torch.ones_like(self.tree.tokens)
        interim_t[input_token_map_bool] = all_target_token_choices

        draft_token_choices = self.tree.tokens[target_token_map_bool]
        target_token_choices = interim_t[target_parent_idxs]

        # get accept_mask
        accept_flags = draft_token_choices == target_token_choices  # flags of positions where draft & target match in <target_token_idxs space>
        accept_idxs = target_token_idxs[accept_flags]  # indices of positions where draft & target match

        accept_mask = torch.zeros(self.tree.end, device=self.device)  # mask for selecting rows from amask
        accept_mask[: self.tree.prefix_len] = 1  # assume whole prefix accepted
        accept_mask[accept_idxs] = 1  # add accepted idxs based on draft==target
        accepted_amask = amask_target[0, 0, :, : self.tree.end] * accept_mask

        mask_row_sums = amask_target[0, 0, :, : self.tree.end].sum(axis=1)

        # get the best sequence
        seq_lengths = accepted_amask.sum(axis=1)
        best_sequence_index = (mask_row_sums * (mask_row_sums == seq_lengths)).argmax()
        best_sequence_mask = amask_target[0, 0, best_sequence_index, : self.tree.end].to(torch.bool)

        fresh_token_idxs = torch.where(best_sequence_mask[self.tree.prefix_len :])[0] + self.tree.prefix_len
        fresh_token_ids = self.tree.tokens[fresh_token_idxs].tolist()

        last_accepted_token_position = fresh_token_idxs[-1] if fresh_token_idxs.numel() else self.tree.prefix_len - 1
        self.tree.reset_to_sequence(best_sequence_mask, target_engine=self.target_engine)

        # Generate one extra token based on target model logits
        extra_token_id = interim_t[last_accepted_token_position]  # all_target_token_choices[last_accepted_token_position - logits_offset]
        self.tree.add(
            token_ids=extra_token_id,
            positions=self.tree.positions[self.tree.size - 1] + 1,
            parent_idxs=torch.tensor([self.tree.size - 1], device=self.device),
            log_probs=torch.tensor(0.0, device=self.device),
            new_status=self.tree.CANDIDATE,
        )
        self.tree.prefix_len = self.tree.end
        self.tree.data[STATUS, : self.tree.prefix_len - 1] = self.tree.PROMPT

        fresh_token_ids.append(extra_token_id.item())

        if logger.level <= logging.DEBUG:
            logger.debug(f"{extra_token_id=}, '{self.tokenizer.convert_ids_to_tokens(extra_token_id.item())}'")
            logger.debug(f"sampled {len(fresh_token_ids)} tokens: {fresh_token_ids} {self.tokenizer.convert_ids_to_tokens(fresh_token_ids)}")
        stats = {"input_len_1": input_ids.shape[-1], "cache_len_1": self.target_engine.kv_len_used, "accepted_tokens": len(fresh_token_ids)}

        return stats, fresh_token_ids

    @staticmethod
    @torch.inference_mode()
    def sampler_from_logits(logits, temperature=1.0, top_p=0.9, min_tokens_to_keep=1):
        """
        Performs token sampling from logits using top-p (nucleus) sampling or deterministic selection.
        Args:
            logits (torch.Tensor): Logits from a language model.
            temperature (float): Adjusts distribution sharpness (higher = more random);  0 for greedy.
            top_p (float): Cumulative probability threshold for top-p sampling.
            min_tokens_to_keep (int): Minimum tokens to keep regardless of top_p.
        Returns: Tuple[torch.Tensor, torch.Tensor]: Indices and log probabilities of selected tokens.
        """

        if temperature > 0:
            if temperature != 1:
                scores = logits / temperature  # Apply temperature scaling
            else:
                scores = logits

            if top_p != 1.0:
                # Sort scores in descending order for top-p sampling
                sorted_logits, sorted_indices = torch.sort(scores, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

                # Create a mask to remove logits not in the top-p
                sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
                sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0  # Keep at least min_tokens_to_keep tokens

                # Scatter the indices to the original order and mask the logits
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                scores.masked_fill_(indices_to_remove, -float("inf"))

            # Sampling from the filtered logits
            probs = torch.softmax(scores, dim=-1)
            _ = torch.multinomial(probs, 1)[:, 0]  # warmup
            selection = torch.multinomial(probs, 1)[:, 0]

            # Compute log probabilities
            logprobs = torch.log_softmax(scores, dim=-1)
            logprobs = torch.index_select(logprobs, 1, selection).diag()

        else:
            # Greedy selection
            selection = torch.argmax(logits, dim=-1)
            logprobs = torch.zeros_like(selection)

        return selection.to(logits.device), logprobs.to(logits.device)

    @torch.inference_mode()
    def get_next_beams(self, logits, parent_pos, parent_idxs, beam_scores, num_beams=None, min_log_prob=None, decay_factor_log=0, max_branch_width=4, **kwargs):
        """
        produces up to num_beams top beams by cumulative log_prob
        with log_prob >= min_log_prob limit
        decay_factor_log - accounts for uncertainty in acceptance of the draft tokens
        """

        logprobs = torch.log_softmax(logits, dim=-1)  # shape: [n_beams, voc_size]
        logprobs_k = logprobs.topk(k=max_branch_width, dim=-1, sorted=False)
        leaves_ids = logprobs_k.indices
        leaves_p = logprobs_k.values

        flat_incoming_probs = (beam_scores.unsqueeze(-1) + decay_factor_log + leaves_p).flatten()
        flat_incoming_ids = leaves_ids.flatten()
        sorted_incoming_probs = flat_incoming_probs.sort(descending=True)
        flat_sorted_log_probs = sorted_incoming_probs.values
        flat_sorted_indices = sorted_incoming_probs.indices

        joint_probs = torch.concat(
            [self.tree.log_probs[self.tree.prefix_len : self.tree.end], flat_sorted_log_probs]
        )  # existing + new probs, for finding threshold

        if joint_probs.shape[-1] > num_beams or joint_probs.shape[-1] + (self.tree.end - self.tree.prefix_len) > self.tree.max_len:
            min_joint_prob = joint_probs.topk(k=num_beams, sorted=False, dim=-1).values.min()

            flat_best_mask = torch.where(flat_sorted_log_probs >= min_joint_prob)[0]
            flat_best_probs = flat_sorted_log_probs[flat_best_mask]
            flat_best_indices = flat_sorted_indices[flat_best_mask]
            best_child_token_ids = flat_incoming_ids[flat_best_indices]

            if flat_best_indices.shape[-1] + self.tree.end > self.tree.max_len:
                logger.debug(f"get_next_beams: trimming draft from {self.tree.end - self.tree.prefix_len} to {self.max_budget=} tokens; {self.tree.end}")
                interim_idx = self.tree.trim_budget(min_log_prob=min_joint_prob)
                parent_idxs = interim_idx[parent_idxs]
        else:
            flat_best_probs = flat_sorted_log_probs
            flat_best_indices = flat_sorted_indices
            best_child_token_ids = flat_incoming_ids[flat_sorted_indices]

        best_hypo_ids = flat_best_indices // max_branch_width

        best_parent_idxs = parent_idxs[best_hypo_ids]
        best_child_pos = parent_pos[best_hypo_ids] + 1

        return best_child_token_ids, best_child_pos, best_parent_idxs, flat_best_probs


class SpecExecBeams(SpecExecBase):
    @torch.inference_mode()
    def grow_tree(self, max_n_beams, max_beam_len, min_log_prob=None, **kwargs):

        logger.debug(f"=================  G R O W  {self.__class__.__name__}  ==================================================")

        input_tokens_count = []  # for logging
        n_beams = 1

        for next_position in range(self.tree.prefix_len, self.tree.prefix_len + max_beam_len):
            draft_logits, parent_indices, parent_scores, parent_positions = self.tree.process_candidates(lim=None)

            best_child_token_ids, best_child_positions, best_parent_idxs, cum_beam_scores = self.get_next_beams(
                draft_logits,
                parent_pos=parent_positions,
                parent_idxs=parent_indices,
                beam_scores=parent_scores,
                num_beams=max_n_beams,
                min_log_prob=min_log_prob,
            )

            n_beams = best_child_token_ids.shape[-1]
            input_tokens_count.append(best_child_token_ids.numel())

            if n_beams == 0:
                logging.debug(f"beams exhausted after {next_position - self.tree.prefix_len} steps")
                break

            self.tree.add(best_child_token_ids, best_child_positions, best_parent_idxs, cum_beam_scores, new_status=self.tree.CANDIDATE)

        logger.debug(f"generated: n_beams={n_beams}, n_tokens={self.tree.size - self.tree.prefix_len}")

        stats = {
            "tree_w": np.unique(self.tree.positions.tolist(), return_counts=True)[1].max(),
            "tree_h": self.tree.positions.max().item() - self.tree.prefix_len + 1,
            "tree_size": self.tree.size - self.tree.prefix_len,  # tree size net of prefix len
            "input_len_0": sum(input_tokens_count),
            "draft_iters": next_position - self.tree.prefix_len + 1 if "next_position" in locals() else 0,
            "lowest_cum_log_prob": round(self.tree.log_probs[: self.tree.end].min().item(), 4),
        }
        return stats

    @torch.inference_mode()
    def get_next_beams(self, logits, parent_pos, parent_idxs, beam_scores, num_beams=None, min_log_prob=None, **kwargs):
        """
        produces up to num_beams top beams by cumulative log_prob
        with log_prob >= min_log_prob limit
        """
        logprobs = torch.log_softmax(logits, dim=-1)  # shape: [n_beams, voc_size]

        flat_log_probs = (beam_scores.unsqueeze(-1) + logprobs).flatten()
        flat_best = flat_log_probs.topk(k=num_beams, largest=True)

        if min_log_prob is not None:
            flat_best_mask = torch.where(flat_best.values > min_log_prob)[0]
            flat_best_values = flat_best.values[flat_best_mask]
            flat_best_indices = flat_best.indices[flat_best_mask]
        else:
            flat_best_values = flat_best.values
            flat_best_indices = flat_best.indices

        best_hypo_ids = flat_best_indices // self.tokenizer.vocab_size
        best_child_token_ids = flat_best_indices % self.tokenizer.vocab_size

        best_parent_idxs = parent_idxs[best_hypo_ids]
        best_child_pos = parent_pos[best_hypo_ids] + 1

        return best_child_token_ids, best_child_pos, best_parent_idxs, flat_best_values
