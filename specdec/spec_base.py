"""
Base class for SpecInfer, SpecExec and whatever comes next
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

from .trees import Tree
from . import utils

if "logger" not in globals():
    logger = utils.get_logger()


class SpecBase(ABC):
    def __init__(self, draft_engine, target_engine, tokenizer):
        self.draft_engine = draft_engine
        self.target_engine = target_engine
        self.tokenizer = tokenizer
        self.device = self.draft_engine.device

    def generate(self, *args, **kwargs):
        """wrapper around generator"""
        for _ in self.generate_and_yield(*args, **kwargs):
            pass
        return self.prefix_tokens

    @torch.inference_mode()
    def generate_and_yield(
        self,
        prompt,
        max_new_tokens,
        seed=None,
        verbose_output=False,
        **kwargs,
    ):
        if kwargs:
            logger.debug(f"Found unused {kwargs=}")

        self.prefix_tokens = self.tokenizer.encode(prompt)
        self.original_num_tokens = len(self.prefix_tokens)

        logger.info(f"{self.__class__.__name__} starting generation.")
        logger.debug(f"Prompt: '{prompt}'")

        self.history = []
        self.log = []
        self.summary = {
            **kwargs,
            "draft_model_name": self.draft_engine.config._name_or_path,
            "target_model_name": self.target_engine.config._name_or_path,
            "prompt_len": len(self.prefix_tokens),
            "prompt_text": prompt,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
        }

        utils.set_seed(seed)
        torch.cuda.reset_peak_memory_stats()

        if hasattr(self, "tree"):
            del self.tree
        self.reset_engines(prefix_len=len(self.prefix_tokens), max_new_tokens=max_new_tokens, **kwargs)

        self.tree = Tree(prefix_tokens=self.prefix_tokens, draft_engine=self.draft_engine, tokenizer=self.tokenizer)
        self.levels = self.get_tree_levels(**kwargs)  # in case the child class works with fixed trees

        # warmup:
        logger.debug("=====  W A R M U P  ========")
        with utils.Timing(synchronize=True) as tw0:
            stats0 = self.grow_tree(prefix_tokens=self.prefix_tokens, **kwargs)
        with utils.Timing(synchronize=True) as tw1:
            stats1, warmup_tokens = self.validate_tree(**kwargs)
        torch.cuda.empty_cache()

        logger.debug(f"warmup time={tw0.elapsed + tw1.elapsed:.3f}; generated {len(warmup_tokens)} tokens.")
        self.prefix_tokens.extend(warmup_tokens)

        # main generation cycle
        iter = 1
        test_time = 0
        eos_flag = False

        while len(self.prefix_tokens) < max_new_tokens + self.original_num_tokens + len(warmup_tokens) and not eos_flag:
            logger.debug(f"=====  I T E R  {iter}  ========")

            with utils.Timing(synchronize=True) as t0:
                stats0 = self.grow_tree(prefix_tokens=self.prefix_tokens, **kwargs)
            with utils.Timing(synchronize=True) as t1:
                stats1, fresh_tokens = self.validate_tree(**kwargs)
            test_time += t0.elapsed + t1.elapsed
            torch.cuda.empty_cache()

            logger.info(
                f"{iter:>3}.  "
                + f"Draft: {t0.elapsed:.3f}s, {stats0['tree_w']}w/{stats0['tree_h']}h/{stats0['tree_size']}size;  "
                + f"Target: {t1.elapsed:.3f}s, +{len(fresh_tokens)} tokens: {self.tokenizer.convert_ids_to_tokens(fresh_tokens)};  inp1:{stats1['input_len_1']}"
            )

            if self.tokenizer.eos_token_id in fresh_tokens:
                fresh_tokens = fresh_tokens[: fresh_tokens.index(self.tokenizer.eos_token_id)]
                eos_flag = True
            self.prefix_tokens.extend(fresh_tokens)

            log1 = {
                "iter": iter,
                "t0": round(t0.elapsed, 2),
                "t1": round(t1.elapsed, 2),
                "new_tokens": len(fresh_tokens),
            }
            self.log.append({**log1, **stats0, **stats1})
            iter += 1
            yield fresh_tokens

        self.summary.update(
            {
                "iters": len(self.log),
                "new_tokens": len(self.prefix_tokens) - self.original_num_tokens - len(warmup_tokens),
                "tree_h": round(np.mean([x.get("tree_h") for x in self.log]), 1),
                "tree_w": int(np.mean([x.get("tree_w") for x in self.log])),
                "tree_size": int(np.mean([x.get("tree_size") for x in self.log])),
                "t0": round(sum([x.get("t0", 0) for x in self.log]) / len(self.log), 4),
                "t1": round(sum([x.get("t1", 0) for x in self.log]) / len(self.log), 4),
                "tft": round(tw0.elapsed + tw1.elapsed, 4),
                "input_0": int(sum([x.get("input_len_0", 0) for x in self.log]) / len(self.log)),
                "input_1": int(sum([x.get("input_len_1", 0) for x in self.log]) / len(self.log)),
                "min_CLP": round(np.mean([x.get("lowest_cum_log_prob", 0) for x in self.log]), 2),
                "draft_iters": round(np.mean([x.get("draft_iters", 0) for x in self.log]), 1),
                "mem_use": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
        )
        self.summary["gen_rate"] = round(self.summary["new_tokens"] / self.summary["iters"], 1)
        self.summary["speed"] = round(self.summary["new_tokens"] / test_time, 2)
        logger.debug(f"\nResult tokens: {self.prefix_tokens}\n string:  {repr(self.tokenizer.decode(self.prefix_tokens))}")

        if verbose_output:
            print("Prompt:", "." * 80)
            print(utils.colored(prompt, "GREEN"))
            print("Generated", "." * 80)
            print(utils.colored(repr(self.tokenizer.decode(self.prefix_tokens[self.original_num_tokens :])), "CYAN"))
            print("=" * 80)

    @abstractmethod
    def grow_tree(self, tree, **kwargs):
        pass

    @abstractmethod
    def validate_tree(self, **kwargs):
        pass

    def get_tree_levels(self, **kwargs):
        # sets self.levels to None unless a child class overrides this method
        pass

    def reset_engines(self, **kwargs):
        self.draft_engine.clear_kv()
        self.target_engine.clear_kv()
