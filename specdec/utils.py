import json
import logging
import random
import sys
import time

import numpy as np
import torch


def get_logger():
    # adding custom log level

    def trace(self, message, *args, **kwargs):
        if self.isEnabledFor(TRACE_LEVEL_NUM):
            self._log(TRACE_LEVEL_NUM, message, args, **kwargs)

    TRACE_LEVEL_NUM = 5
    logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
    logging.TRACE = TRACE_LEVEL_NUM
    logging.Logger.trace = trace

    # initializing logger

    class MillisecondFormatter(logging.Formatter):
        def formatTime(self, record, datefmt=None):
            created = time.strftime("%H:%M:%S", time.localtime(record.created))
            millis = f"{int(record.msecs):03d}"
            return f"{created}.{millis}"

    logger = logging.getLogger()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = MillisecondFormatter("[%(asctime)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


class Timing:
    """
    Time measuring context manager
        with Timing() as t:
            time.sleep(0.1)
            print("Interim timing =", t.elapsed)
            time.sleep(0.1)
        print("Timing =", t.elapsed)
    """

    def __init__(self, device="cuda:0", synchronize=False):
        self.device = device
        self.synchronize = synchronize
        self._elapsed = None

    def __enter__(self):
        if self.synchronize:
            torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()
        return self

    @property
    def elapsed(self):
        if self._elapsed is None:
            return time.perf_counter() - self.start
        return self._elapsed

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.synchronize:
            torch.cuda.synchronize(self.device)
        self._elapsed = time.perf_counter() - self.start


def set_seed(seed=0):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)


def kv_cache_mask_apply(kv_cache, mask=None, truncate=None):
    """Applies mask to KV cache or truncates it"""
    kv_cache_len = kv_cache[0][0].shape[2]
    if mask is None:
        mask = torch.ones(truncate).to(kv_cache[0][0].device).bool()
    if (mask.dtype == torch.bool) and (mask.shape[-1] < kv_cache_len):
        mask = torch.nn.functional.pad(mask, (0, kv_cache_len - mask.shape[-1]), "constant", False)
    try:
        return tuple(
            (
                past_keys_i[:, :, mask, :],
                past_values_i[:, :, mask, :],
            )
            for (past_keys_i, past_values_i) in kv_cache
        )
    except IndexError:
        print(f"kv_cache_mask_apply ERROR; {mask.dtype=}, {kv_cache_len=}")
        print(mask)


def canary(immune=True):
    """checks whether cuda is still functioning."""
    try:
        torch.ones(1).cuda()
    except Exception as e:
        print("CANARY JUST DIED!")  # Set breakpoint here
        if immune:
            pass
        else:
            raise e


def get_dataset(dataset_name, path_to_data="./data"):
    """
    loads prompts dataset from json file in standard location({repo_root}/data), as list of tuples (id, string)
    returns list of strings
    assumes data created as list of tuples (id, string), saved as json
    """
    try:
        file_path = f"{path_to_data}/{dataset_name}.json"
        with open(file_path, "r") as f:
            dataset = json.load(f)
        dataset = [x[1] for x in dataset]
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing `data/{dataset_name}.json` file.")

    return dataset


@torch.no_grad()
def build_amask_slow(parent_indices, start=0):
    """
    building 4D mask
    parent_indices parents indices.
    start: first input_ids token index
    THIS IS A QUITE SLOW IMPLEMENTATION. USE FOR DEBUG ONLY (~3s with 1k tokens)
    """
    mask = torch.zeros(parent_indices.shape[-1] - start, parent_indices.shape[-1], dtype=torch.int64, device=parent_indices.device)
    for i in range(start, parent_indices.shape[-1]):
        pos = torch.tensor([i])
        mask_i = []
        while pos != -1:
            mask_i.append(pos)
            pos = parent_indices[pos]
        mask[i - start, mask_i] = 1
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask.bool()


@torch.no_grad()
def get_sampling_probas(logits, temperature=1.0, top_p=1.0):
    """
    Computes sampling probabilities from logits using temperature scaling and top-p filtering.

    Args:
        logits (torch.Tensor): The logits from a model's output.
        temperature (float): Scaling factor for logits, default is 1.0.
        top_p (float): Proportion of top logits to consider, default is 1.0 (all logits).

    Returns:
        torch.Tensor: The softmax probabilities after applying temperature scaling and top-p filtering.
    """
    if temperature == 0:
        max_indices = torch.argmax(logits, dim=-1)
        probas = torch.zeros_like(logits, dtype=torch.float32)
        probas[torch.arange(logits.shape[0]), max_indices] = 1
        return probas

    if temperature > 0:
        logits = logits / temperature  # Apply temperature scaling

    if top_p < 1.0:
        # Sort scores in descending order for top-p sampling
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Create a mask to remove logits not in the top-p
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        min_tokens_to_keep = 1  # technical constant
        sorted_indices_to_remove[..., -min_tokens_to_keep:] = 0  # Keep at least min_tokens_to_keep tokens

        # Scatter the indices to the original order and mask the logits
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits.masked_fill_(indices_to_remove, -float("inf"))  # this is the slowest part of this fn
        assert (logits > -float("inf")).sum(dim=1).min() >= 1  # check for failure in top-p

    softmax_probas = torch.softmax(logits, dim=-1)
    return softmax_probas


# escape color codes for use with colored and standalone
# example: print (f"{COLORS['HIGHLIGHTED_PURPLE']}this is HIGHLIGHTED_PURPLE{COLORS['DEFAULT']}")
# example: print(colored("this is blue", "BLUE"), colored("this is green", "GREEN"))

COLORS = {
    "DEFAULT": "\x1b[0m",
    "BOLD": "\x1b[1m",
    "ITALIC": "\x1b[3m",
    "UNDERLINE": "\x1b[4m",
    "UNDERLINE_THICK": "\x1b[21m",
    "HIGHLIGHTED": "\x1b[7m",
    "HIGHLIGHTED_BLACK": "\x1b[40m",
    "HIGHLIGHTED_RED": "\x1b[41m",
    "HIGHLIGHTED_GREEN": "\x1b[42m",
    "HIGHLIGHTED_YELLOW": "\x1b[43m",
    "HIGHLIGHTED_BLUE": "\x1b[44m",
    "HIGHLIGHTED_PURPLE": "\x1b[45m",
    "HIGHLIGHTED_CYAN": "\x1b[46m",
    "HIGHLIGHTED_GREY": "\x1b[47m",
    "HIGHLIGHTED_GREY_LIGHT": "\x1b[100m",
    "HIGHLIGHTED_RED_LIGHT": "\x1b[101m",
    "HIGHLIGHTED_GREEN_LIGHT": "\x1b[102m",
    "HIGHLIGHTED_YELLOW_LIGHT": "\x1b[103m",
    "HIGHLIGHTED_BLUE_LIGHT": "\x1b[104m",
    "HIGHLIGHTED_PURPLE_LIGHT": "\x1b[105m",
    "HIGHLIGHTED_CYAN_LIGHT": "\x1b[106m",
    "HIGHLIGHTED_WHITE_LIGHT": "\x1b[107m",
    "STRIKE_THROUGH": "\x1b[9m",
    "MARGIN_1": "\x1b[51m",
    "MARGIN_2": "\x1b[52m",
    "BLACK": "\x1b[30m",
    "RED_DARK": "\x1b[31m",
    "GREEN_DARK": "\x1b[32m",
    "YELLOW_DARK": "\x1b[33m",
    "BLUE_DARK": "\x1b[34m",
    "PURPLE_DARK": "\x1b[35m",
    "CYAN_DARK": "\x1b[36m",
    "GREY_DARK": "\x1b[37m",
    "BLACK_LIGHT": "\x1b[90m",
    "RED": "\x1b[91m",
    "GREEN": "\x1b[92m",
    "YELLOW": "\x1b[93m",
    "BLUE": "\x1b[94m",
    "PURPLE": "\x1b[95m",
    "CYAN": "\x1b[96m",
    "WHITE": "\x1b[97m",
}


def colored(input_str: str, code: str):
    """
    wraps string into coloring escape sequences for colored printout to terminal
    based on https://stackoverflow.com/a/75054413/10396469
    usage: print(colored("this is blue", "BLUE"), colored("this is green", "GREEN"))
    for more options and cross-platform use consider colorama or yachalk
    """
    assert code.upper() in COLORS, f"invalid color code {code}"
    return COLORS[code.upper()] + input_str + COLORS["DEFAULT"]
