from tqdm import tqdm
from specdec import utils
import torch
import numpy as np


def benchmark_engine(engine, input_sizes=[256], repeats=5, max_len=1024, verbose=False):
    _max_len = getattr(engine, "max_len", max_len)

    with torch.inference_mode():
        result = {
            "model": engine.config._name_or_path,
            "device": torch.cuda.get_device_name(engine.device).replace("NVIDIA ", ""),
            "engine": engine.__class__.__name__,
            "max_len": _max_len,
        }

        _ = benchmark_single(engine, input_size=1, repeats=1, max_len=_max_len)  # warmup

        pbar = tqdm(input_sizes, desc="...", ncols=120) if verbose else input_sizes
        stats = []
        for n in pbar:
            if n <= max_len:
                if verbose:
                    pbar.desc = f"working: cls={engine.__class__.__name__} ML={engine.max_len} {n=}"
                try:
                    s = benchmark_single(engine, input_size=n, repeats=repeats, max_len=_max_len)
                    stats.append(s)
                except RuntimeError:
                    pass
                if verbose:
                    pbar.desc = f"cls={engine.__class__.__name__} ML={engine.max_len} {n=}"
        stats.sort(key=lambda x: x["size"])
        for entry in stats:
            result[entry["size"]] = entry["latency"]
        for entry in stats:
            result[f"m_{entry['size']}"] = entry["mem_use"]
    return result


def benchmark_single(engine, input_size, repeats=5, max_len=None):
    _max_len = getattr(engine, "max_len", max_len)
    timings = []
    torch.cuda.reset_peak_memory_stats()
    init_mem = torch.cuda.max_memory_allocated()
    attention_mask = torch.zeros(1, 1, input_size, _max_len, device=engine.device)
    _causal_mask = torch.tril(torch.ones(input_size, input_size, device=engine.device))
    attention_mask[..., :input_size, :input_size] = _causal_mask

    min_dtype = torch.finfo(engine.model.dtype).min
    attention_mask = (attention_mask == 0.0) * min_dtype
    position_ids = torch.arange(input_size, device=engine.device).unsqueeze(0)
    cache_position = torch.arange(input_size, device=engine.device)

    for r in range(repeats):
        input_ids = torch.randint(2, 10000, (1, input_size), device=engine.device)
        with utils.Timing(synchronize=True) as t:
            _ = engine.forward(
                input_ids=input_ids,
                attention_mask=attention_mask.to(torch.float16),
                position_ids=position_ids,
                cache_position=cache_position,
            )
        timings.append(t.elapsed)
    mem_use = (torch.cuda.max_memory_allocated() - init_mem) / 2**30
    stats = dict(
        size=input_size,
        latency=round(np.median(timings) * 1000, 3),
        mem_use=round(mem_use, 3),
    )
    return stats


def get_max_batchsize(bb, r_tol=0.10):
    """
    Get the maximum batch size that can be used while maintaining a relative tolerance.
    Parameters:
    bb (dict): A dictionary mapping batch sizes to their corresponding times.
    r_tol (float): The relative tolerance for the batch size. Default is 0.10.
    Returns:
    tuple: A tuple containing the maximum batch size and its corresponding time.
    """
    pow2 = [2**i for i in range(20)]
    bb = [(k, v) for k, v in bb.items() if k in pow2]
    bb.sort(key=lambda x: x[0])

    best_n, best_t = bb[0]
    for i, (n, t) in enumerate(bb[1:], start=1):
        if t / best_t - 1 <= r_tol:
            best_n, best_t = n, t
        else:
            break

    return best_n, best_t
