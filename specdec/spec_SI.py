""""
SpecInfer reproduction.
n_beams sequences are generated starting form the root, with no subsequent branching
"""

import logging
from typing import Tuple

import numpy as np
import torch

from . import utils
from .spec_base import SpecBase
from .trees import TOKENS, POSITIONS, PARENTS, STATUS  # noqa: F401
from .utils import kv_cache_mask_apply

logger = utils.get_logger()


class SpecInfer(SpecBase):
    @torch.inference_mode()
    def grow_tree(
        self,
        max_beam_len,
        max_n_beams=None,
        replacement=False,
        repack=False,
        min_log_prob=None,
        max_budget=None,
        temperature=1.0,
        top_p=1.0,
        draft_temperature=None,
        **kwargs,
    ):
        """Builds up newly created sampling tree."""

        if repack and (replacement is not True):
            raise ValueError("Non-False option `repack` requires `replacement=True`")

        if draft_temperature is None:
            if temperature >= 0.1:
                draft_temperature = temperature
            else:
                draft_temperature = 1.0

        logger.debug(f"=================  G R O W  {self.__class__.__name__}  ==================================================")
        logger.debug(f"prefix text: {repr(self.tokenizer.decode(self.tree.tokens[-32:]))}")

        input_tokens_count = []  # for logging
        best_hypo_ids = torch.tensor([0], device=self.device)
        n_beams = 1  # = best_hypo_ids.shape[0]

        next_position = self.tree.prefix_len
        while True:
            logger.debug(f"Grow position: {next_position} --------------------------")

            draft_logits, parent_indices, parent_scores, parent_positions = self.tree.process_candidates(lim=max_n_beams, pick_best=False)
            logger.debug(f"after process_candidates {draft_logits.shape=}, {self.tree.end=}")

            draft_probs = torch.softmax(draft_logits, dim=1)

            best_child_token_ids, best_hypo_ids, best_child_probs = self.select_best_children(
                probs=draft_probs,
                beams_cum_log_probas=self.tree.log_probs[parent_indices],  # incoming cumulative probabilities, shaped (cohort_size)
                max_n_beams=max_n_beams,
                replacement=replacement,
                min_log_prob=min_log_prob,
                step=next_position - self.tree.prefix_len,
                **kwargs,
            )
            best_parents_idxs = parent_indices[best_hypo_ids]
            best_child_positions = parent_positions[best_hypo_ids] + 1

            assert draft_logits.shape[0] == parent_indices.shape[0]
            for i, pi in enumerate(parent_indices):
                if pi in best_parents_idxs:
                    self.tree.q_probs[pi.item()] = draft_probs[i]

            input_tokens_count.append(best_child_token_ids.numel())

            self.tree.add(best_child_token_ids, best_child_positions, best_parents_idxs, best_child_probs, new_status=self.tree.CANDIDATE)

            n_beams = best_hypo_ids.shape[0]

            # early cycle termination check; relevant with min_log_prob parameter or fixed tree structure
            if n_beams == 0:
                logging.debug(f"beams exhausted after {next_position - self.tree.prefix_len} steps")
                break

            child_cum_log_probs = (self.tree.log_probs[parent_indices][best_hypo_ids] + torch.log(best_child_probs)).flatten()

            # Logging:
            if logger.level <= logging.DEBUG:  # the next line takes ~17-25 ms
                decoded_best_token_ids = [self.tokenizer.decode(t) for t in best_child_token_ids.flatten()]  # only used for logging
                logger.debug(
                    f"new tokens:{best_child_token_ids.tolist()}, {decoded_best_token_ids}; "
                    f"cum_log_probs:{[round(clp.item(), 1) for clp in child_cum_log_probs]}"
                )
                logger.trace(f"Tokens:{decoded_best_token_ids}; cum_log_probs:{[round(p.item(), 2) for p in child_cum_log_probs]}")

            next_position += 1

            if self.levels is not None:
                # limiting number of delivered tokens to the set budget for the classes with fixed trees
                if self.tree.size - self.tree.prefix_len >= max_budget:
                    self.tree = self.tree.trimmed(self.tree.prefix_len + max_budget)
                    break
            else:
                if next_position - self.tree.prefix_len >= max_beam_len:
                    break

        if logger.level <= logging.TRACE:
            # drawing beams tree
            self.tree.draw(tokenizer=self.tokenizer)

        logger.debug(f"generated: n_beams={n_beams}, n_tokens={self.tree.size - self.tree.prefix_len}")

        if repack:
            # trees with replacement are always repacked to combine stems, matching on prefixes, into branched trees
            raise NotImplementedError
            self.tree = self.tree.repacked()  # Not implemented
            logger.debug(f"repacked: n_beams={n_beams}, n_tokens={self.tree.size - self.tree.prefix_len}")

        stats = {
            "tree_w": np.unique(self.tree.positions.tolist(), return_counts=True)[1].max(),
            "tree_h": self.tree.positions.max().item() - self.tree.prefix_len + 1,
            "tree_size": self.tree.size - self.tree.prefix_len,  # tree size net of prefix len
            "input_len_0": sum(input_tokens_count),
            "draft_iters": next_position - self.tree.prefix_len + 1 if "next_position" in locals() else 0,
        }

        return stats

    @torch.inference_mode()
    def select_best_children(
        self,
        probs,
        beams_cum_log_probas,
        max_n_beams,
        replacement,
        **kwargs,
    ):
        prev_n_beams = beams_cum_log_probas.shape[-1]
        samples_per_beam = max_n_beams // prev_n_beams

        best_hypo_ids = torch.repeat_interleave(
            torch.arange(prev_n_beams, device=probs.device, dtype=torch.int64),
            samples_per_beam,
        )
        best_child_token_ids = torch.multinomial(probs, num_samples=samples_per_beam, replacement=replacement).reshape(1, -1)

        best_child_probs = torch.gather(probs, dim=-1, index=best_child_token_ids)
        return best_child_token_ids, best_hypo_ids, best_child_probs

    @torch.inference_mode()
    def validate_tree(self, temperature, top_p, **kwargs):
        """validation of the generated sequences with Target model"""
        logger.debug(f"=================  V A L I D A T E   {self.__class__.__name__}   ============================")

        initial_kv_len = self.target_engine.kv_len_used

        target_token_map_bool = self.tree.status[: self.tree.end] >= self.tree.PROCESSED  # tokens generated in the current iteration
        target_token_map_bool[: self.tree.prefix_len] = False  # addresses problem of the last prefix token status
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
        logger.debug(f"Target fwd inputs: {input_ids.shape=}, {initial_kv_len=}, {amask_target.shape=}")

        target_logits = self.target_engine.forward(
            input_ids=input_ids,
            attention_mask=self.tree.invert_mask(amask_target, dtype=self.target_engine.model.dtype),
            position_ids=position_ids,
            cache_position=cache_position,
        )
        target_logits = target_logits.squeeze(0)  # remove batch dim   # LOGITS_COORDS
        num_unprocessed_tokens = self.tree.prefix_len - initial_kv_len
        logits_offset = self.tree.prefix_len - num_unprocessed_tokens  # offset between tree index and model logits index
        target_softmax_probas = utils.get_sampling_probas(target_logits, temperature, top_p)  # LOGITS_COORDS

        logger.debug(f"{target_logits.shape=}, {num_unprocessed_tokens=}")

        # GOING DOWN THE TREE TO DECODE
        current_parent_index = self.tree.prefix_len - 1
        fresh_token_ids = []
        cache_indices = []  # for pruning target model cache after the iteration
        extra_token_allowed = True

        for position in self.tree.positions[self.tree.prefix_len :].unique():
            logger.trace(f"Verify position: {position} --------------------------")

            candidates_mask = torch.logical_and(self.tree.parents == current_parent_index, self.tree.positions == position)  # ABS_COORDS for parent ids
            if not torch.any(candidates_mask):
                logger.debug(f"End of beam at {position=}, need to sample extra token.")
                break
            candidates_indices = torch.where(candidates_mask)[0]
            debug_line = f"{position}, candidates:{str(candidates_indices.tolist()):<12}  "

            p = target_softmax_probas[self.tree.parents[candidates_indices[0]] - logits_offset, :]
            q = self.tree.q_probs[self.tree.parents[candidates_indices[0]].item()]

            chosen_candidate_index, chosen_token_id = self._inference_step(p, q, candidates_indices, candidate_token_ids=self.tree.tokens[candidates_indices])

            if chosen_candidate_index is not None:  # accepted
                fresh_token_ids.append(chosen_token_id.item())
                p_accept = (p[chosen_token_id] / q[chosen_token_id]).clamp_max(1.0)
                debug_line += f"accepted, p={p_accept:.3f}  "
                logger.debug(debug_line + f"chosen:{chosen_candidate_index}, token:{chosen_token_id} ({repr(self.tokenizer.decode(chosen_token_id))})")
                cache_indices.append(chosen_candidate_index.item())
                current_parent_index = chosen_candidate_index

            else:  # rejected
                # chosen_token_id will be added after tree compacting
                assert chosen_token_id not in self.tree.tokens[candidates_indices]
                debug_line += f"sampled from p_adj: token={chosen_token_id} {repr(self.tokenizer.decode(chosen_token_id))}; "
                logger.debug(debug_line + "It was from outside of the tree.")
                extra_token_allowed = False
                break  # exit the "for position_id" loop

        # compacting the tree and the caches
        logger.info(f"{cache_indices=}")
        best_sequence_mask = torch.zeros_like(target_token_map_bool)
        best_sequence_mask[: self.tree.prefix_len] = 1
        best_sequence_mask[cache_indices] = 1
        self.tree.reset_to_sequence(best_sequence_mask, target_engine=self.target_engine)

        if extra_token_allowed:
            # adding extra token in the end if previous was sampled from the tree (otherwise see break above)
            extra_token_pos_in_target = chosen_candidate_index - logits_offset
            extra_token_id = torch.multinomial(target_softmax_probas[extra_token_pos_in_target], num_samples=1)
            logger.debug(f"Extra token after complete beam: {extra_token_id} ({self.tokenizer.convert_ids_to_tokens(extra_token_id)})")
        else:
            extra_token_id = chosen_token_id

        self.tree.add(
            token_ids=extra_token_id,
            positions=self.tree.positions[self.tree.size - 1] + 1,
            parent_idxs=torch.tensor([self.tree.size - 1], device=self.device),
            log_probs=torch.tensor(0.0, device=self.device),
            new_status=self.tree.CANDIDATE,
        )
        fresh_token_ids.append(extra_token_id.item())

        self.tree.prefix_len = self.tree.end
        self.tree.data[STATUS, : self.tree.prefix_len - 1] = self.tree.PROMPT
        self.tree.data[STATUS, self.tree.prefix_len - 1] = self.tree.CANDIDATE

        logger.debug(f"sampled {len(fresh_token_ids)} tokens: {fresh_token_ids} {self.tokenizer.convert_ids_to_tokens(fresh_token_ids)}")
        stats = {"input_len_1": input_ids.shape[-1], "cache_len_1": logits_offset, "accepted_tokens": len(fresh_token_ids)}

        return stats, fresh_token_ids 

    @staticmethod
    @torch.no_grad()
    def _inference_step(p, q, candidate_idxs, candidate_token_ids) -> Tuple[int, int]:
        """return chosen candidate_id, chosen candidate_token_id. If chosen_candidate_id is None, all cands were rejected"""
        p_adj = p.clone()
        q = q.clone()

        chosen_token_id = None
        for candidate_idx, candidate_token_id in zip(candidate_idxs, candidate_token_ids):
            p_adj_i = p_adj[candidate_token_id]
            q_i = q[candidate_token_id]

            r = torch.rand(1, device=p.device)
            p_accept = (p_adj_i / q_i).clamp_max(1.0)
            if r <= p_accept:
                chosen_token_id = candidate_token_id
                return candidate_idx, chosen_token_id
            else:
                p_adj = (p_adj - q).clip(min=0)
                p_adj = torch.nn.functional.normalize(p_adj, dim=0, p=1)

                q[candidate_token_id] = 0
                q = torch.nn.functional.normalize(q, dim=0, p=1)

        assert chosen_token_id is None  # we did not accept any token => sample from p_adj
        try:
            chosen_token_id = torch.multinomial(p_adj, num_samples=1)
        except RuntimeError:
            raise RuntimeError("problem with p_adj!" f"{sum(p_adj > 0).item()}, {candidate_idx, candidate_idxs, candidate_token_ids}")
        # note that at this point, chosen_token_idx can never be in tree
        return None, chosen_token_id

    @torch.no_grad()
    def _prune_target_model_kv_cache(self, cache_indices, prefix_len):
        """retains on KV cache only elements related to the prefix and to the selected tokens"""
        target_cache_keep_mask = torch.cat((torch.arange(prefix_len), torch.tensor(cache_indices))).int().to(self.device)
        self.target_model_outputs["past_key_values"] = kv_cache_mask_apply(self.target_model_outputs["past_key_values"], mask=target_cache_keep_mask)
        kv_len = self.target_model_outputs["past_key_values"][0][0].shape[2]
        logger.trace(f"Pruned target KV cache len={kv_len}, mask={target_cache_keep_mask.int().tolist()}")

    def reset_engines(self, prefix_len, max_new_tokens, max_beam_len, max_n_beams, **kwargs):
        super().reset_engines()
        safety_margin = 64  # account for warmup tokens and excess generation
        draft_max_len = prefix_len + max_new_tokens + max_n_beams * max_beam_len + safety_margin + 256
        target_max_len = prefix_len + max_new_tokens + max_n_beams * max_beam_len + safety_margin
        if hasattr(self, "tree"):
            self.tree.set_max_len(draft_max_len)
        else:
            self.draft_engine.set_max_len(draft_max_len)

        self.target_engine.set_max_len(target_max_len)
        logger.info(f"Max_len reset: {draft_max_len=}, {target_max_len=}")
