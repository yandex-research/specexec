import argparse
import datetime
import json
import logging
import os
import socket
import subprocess
from itertools import product
from pathlib import Path

import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm

from offloading.offload_model import load_gptq_offloaded_model, load_offloaded_model
from specdec import SpecExecBeams, SpecExecBase, SpecInfer, utils
import engine
from specdec.utils import colored

device = torch.device("cuda:0")
_DEFAULT_DEVICE_SIZE = 2
DISPLAY_WIDTH = 160
pd.set_option("display.width", DISPLAY_WIDTH)
pd.set_option("display.max_columns", 32)


def create_spec_generator(
    model_name_0,
    model_name_1,
    draft_engine_class,
    gen_type="SX",
    offload=False,
    device_size=_DEFAULT_DEVICE_SIZE,
    check_tokenizer=False,
):
    """Creates a SpecGenerator object for different generation types.

    This function loads draft and target pre-trained language models specified by their names
    and creates a SpecBase subclass object based on the provided generation type.
    It also handles several configuration options like device placement and tokenizer verification.

    Args:
        model_name_0 (str): Name of the draft model.
        model_name_1 (str): Name of the target model.
        gen_type (str, optional): Generation type. Defaults to "SX" (SpecExec).
            Valid options include:
                - "SpecExecBase", : SpecExec generator
                - "SI", "spec_infer", "specinfer": SpecInfer generator
        offload (bool, optional): Whether to offload model 1 using offloading library. Defaults to False.
        device_size (int, optional): Device size for offloading. Defaults to `_DEFAULT_DEVICE_SIZE`.
        check_tokenizer (bool, optional): Whether to verify if both models have the same tokenizer. Defaults to False.

    Returns:
        SpecGenerator: An instance of a SpecBase subclass object based on the provided parameters.

    Raises:
        ValueError: If an invalid `gen_type` is provided.
    """

    if len(model_name_0.split("::")) == 2:
        model_name_0, rev_0 = model_name_0.split("::")
    else:
        rev_0 = "main"  # default in `from_pretrained()`

    if len(model_name_1.split("::")) == 2:
        model_name_1, rev_1 = model_name_1.split("::")
    else:
        rev_1 = "main"  # default in `from_pretrained()`

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_0, legacy=False)

    if check_tokenizer:
        # verify that the two models have the same tokenizer
        tokenizer_1 = transformers.AutoTokenizer.from_pretrained(model_name_1, legacy=False)
        vv0 = tokenizer.get_vocab()
        vv1 = tokenizer_1.get_vocab()

        ignored_tokens = ["[PAD]"]  # disregard these tokens when comparing the cokonizers' vocabs
        assert set(vv0.keys()).difference(ignored_tokens) == set(vv1.keys()).difference(ignored_tokens)
        for k in set(vv0.keys()).difference(ignored_tokens):
            assert vv0[k] == vv1[k]
        del tokenizer_1, vv0, vv1

    logger.info(f"Loading Model 0: `{model_name_0}`, {draft_engine_class=}")
    if draft_engine_class.lower() in ("es", "static", "enginestatic"):
        model_0 = transformers.AutoModelForCausalLM.from_pretrained(model_name_0, device_map=device, torch_dtype=torch.float16, revision=rev_0)
        draft_engine = engine.EngineStatic(model_0, max_len=args.tree_max_len)
    # elif draft_engine_class.lower() in ("esc", "staticcompiled", "enginestaticcompiled"):
    #     model_0 = transformers.AutoModelForCausalLM.from_pretrained(model_name_0, device_map=device, torch_dtype=torch.float16, revision=rev_0)
    #     draft_engine = engine.EngineStaticCompiled(model_0, max_len=args.tree_max_len)
    # elif draft_engine_class.lower() in ("ie", "inferenceengine"):
    #     draft_engine = engine.InferenceEngine(model_name_0, max_len=args.tree_max_len)
    elif draft_engine_class.lower() in ("padded", "inferenceenginepadded"):
        draft_engine = engine.InferenceEnginePadded(model_name_0, max_len=args.tree_max_len)
    elif draft_engine_class.lower() in ("er", "regular", "engineregular"):
        draft_engine = engine.EngineRegular(model_name_0, max_len=args.tree_max_len)
    else:
        raise ValueError(f"Unsupported engine class: {draft_engine_class} !")

    logger.info(f"Loading Model 1: `{model_name_1}`")
    gptq_max_input_length = 16384  # constant for GPTQ models

    if offload:
        if "gptq" in model_name_1.lower():
            model_1 = load_gptq_offloaded_model(model_name_1, device_size=device_size, main_device=device, max_input_length=gptq_max_input_length)
        else:
            model_1 = load_offloaded_model(model_name_1, device_size=device_size, main_device=device)

    else:
        model_1 = transformers.AutoModelForCausalLM.from_pretrained(model_name_1, device_map=device, torch_dtype=torch.float16, revision=rev_1)

        if "gptq" in model_name_1.lower():
            model_1_config = transformers.AutoConfig.from_pretrained(model_name_1)
            if getattr(model_1_config.quantization_config, "act_order", False) and (model_1_config.config.max_length < 16384):
                try:
                    from auto_gptq import exllama_set_max_input_length

                    model_1 = exllama_set_max_input_length(model_1, gptq_max_input_length)
                    print("set `exllama_set_max_input_length` OK")
                except (AttributeError, ValueError, ImportError):
                    # AttributeError may happen if GPTQ-quantized model has no attribute 'device_to_buffers'
                    # could be fixed by using code from post_init()
                    # ImportError resembles https://github.com/open-mmlab/mmdetection3d/issues/1152
                    logger.warning("Failed to set `exllama_set_max_input_length`")

    # target_engine = EngineStatic(model_1, max_len=args.tree_max_len)
    target_engine = engine.EngineRegular(model_1, max_len=args.tree_max_len)

    if gen_type.lower() in ("sx_base", "base", "sx2", "spec_exec_base", "specexecbase"):
        spec_generator = SpecExecBase(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("spec_exec_beams", "specexecbeams", "sx_beams"):
        spec_generator = SpecExecBeams(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("sa", "a", "spec_adaptive", "specadaptive"):
        spec_generator = SpecAdaptive(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("sf", "f", "spec_fixed", "specfixed"):
        spec_generator = SpecFixed(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("si", "spec_infer", "specinfer"):
        spec_generator = SpecInfer(draft_engine, target_engine, tokenizer)
    elif gen_type.lower() in ("sis", "spec_infer_stems", "specinferstems"):
        spec_generator = SpecInferStems(draft_engine, target_engine, tokenizer)
    else:
        raise ValueError(f"unknown {gen_type=}")

    logger.info(f"Created spec_generator of type {gen_type}; Models: {model_name_0}, {model_name_1}")
    return spec_generator


def run_tests(
    spec_generator,
    dataset,
    args,
    max_budget=None,
    max_n_beams=None,
    max_beam_len=None,
    max_branch_width=None,
    min_log_prob=None,
    **kwargs,
):
    """runs uniform experiments from dataset using same set of parameters"""
    test_logs = []

    for i in range(args.dataset_start_index, min(args.dataset_start_index + args.n_tests, len(dataset))):
        prompt = dataset[i]
        _ = spec_generator.generate(
            prompt,
            max_n_beams=max_n_beams,
            max_beam_len=max_beam_len,
            max_new_tokens=args.max_new_tokens,
            branching=args.branching,
            max_budget=max_budget,
            max_branch_width=max_branch_width,
            replacement=args.replacement,
            verbose=args.verbose,
            temperature=args.temperature,
            draft_temperature=args.draft_temperature,
            top_p=args.top_p,
            min_log_prob=min_log_prob,
            seed=args.seed,
            tree_max_len=args.tree_max_len,
            **kwargs,
        )

        test_logs.append(spec_generator.summary)
        generated_text = spec_generator.tokenizer.decode(spec_generator.prefix_tokens[spec_generator.original_num_tokens :]).__repr__().strip("'")

        excl_keys = ["ver", "model_name_0", "model_name_1"]
        log1 = {k: v for k, v in spec_generator.summary.items() if k not in excl_keys}
        log1 = {"run": i, **log1, "text": generated_text[:32]}
        log1["prompt_text"] = log1["prompt_text"].replace(r" [\INST] ", "")[-32:]  # last 32 prompt chars

        stdout_whitelist = (
            "run",
            "prompt_len",
            "iters",
            "new_tok",
            "tree_h",
            "tree_w",
            "tree_size",
            "t0",
            "t1",
            "input_0",
            "input_1",
            "min_CLP",
            "gen_rate",
            "speed",
            "mem_use",
        )
        log_one_line(log1, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="summary", stdout_whitelist=stdout_whitelist)

    df = pd.DataFrame(test_logs)

    exp_summary = dict(
        max_n_beams=max_n_beams,
        max_beam_len=max_beam_len,
        min_log_prob=min_log_prob,
        max_budget=max_budget,
        max_branch_width=max_branch_width,
        gen_rate=round((df.new_tokens / df.iters).mean(), 2),  # macro-averaged generation rate
        gen_rate_micro=round(df.new_tokens.sum() / df.iters.sum(), 2),
        gen_speed=round(df.speed.mean(), 3),
        gen_speed_micro=round(df.new_tokens.sum() / (df.new_tokens / df.speed).sum(), 3),
        t0=round(df.t0.mean(), 4),
        t1=round(df.t1.mean(), 4),
        input_0=round(df.input_0.mean(), 1),
        input_1=round(df.input_1.mean(), 1),
        tree_size=round(df.tree_size.mean(), 1),
        tree_w=round(df.tree_w.mean(), 1),
        tree_h=round(df.tree_h.mean(), 1),
        prompt_len=round(df.prompt_len.mean(), 1),
        min_CLP=round(df.min_CLP.mean(), 2),
        mem_use=round(df.mem_use.max(), 2),
    )

    torch.cuda.empty_cache()
    return exp_summary, test_logs


def log_one_line(data_dict, verbose, save_dir=None, exp_name=None, msg_type=None, stdout_whitelist=None):
    """
    Logs key-value pairs from a dictionary to both the console (as a single line) and a JSONL file,
    with optional filtering for certain keys and conditional logging based on verbosity.

    Args:
        data_dict (dict): A dictionary containing the data to be logged.
        verbose (bool): If True, logs to stdout regardless of logger level.
        save_dir (str): Path to the directory where the log file will be saved.
        exp_name (str): Name of the experiment, used for the log file name.
        msg_type (str, optional): A message type to be included in the log file. Defaults to None.
    """
    stdout_blacklist = ["prompt_text", "text"]
    message_colors = {"exp": "GREEN", "summary": "WHITE", "config": "YELLOW_DARK", "zero": "blue", "info": "GREEN_DARK"}

    if verbose or (logger.level >= logging.INFO):
        if stdout_whitelist:
            log_line = "  ".join([f"{k}:{v}" for k, v in data_dict.items() if k in stdout_whitelist and v is not None])
        else:
            log_line = "  ".join([f"{k}:{v}" for k, v in data_dict.items() if k not in stdout_blacklist and v is not None])

        print(colored(log_line, message_colors.get(msg_type, "WHITE")))

    # logging to file
    if (msg_type is not None) and (save_dir is not None) and (exp_name is not None):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        log_filename = save_path / f"{exp_name}.jsonl"
        with log_filename.open("a") as file:
            file.write(json.dumps({"msg_type": msg_type, **data_dict}) + "\n")


def arg_to_list(args, arg):
    """
    Converts a command-line argument value to a list of appropriate types.
    Handles different value formats (single value, comma-separated values, "None"),
    converts to integers or floats as needed, and returns a list of parsed values.

    Args:
        args: An object containing command-line arguments (e.g., argparse.Namespace).
        arg (str): The name of the argument to retrieve and convert.

    Returns:
        list: A list of parsed values from the argument.
    """
    arg_value = getattr(args, arg)
    float_args = ["min_log_prob"]
    if arg_value is None:
        return [None]

    def from_str(s):
        """
        Parses a string value into an integer, float, or None.
        Args:  s (str): The string to parse.
        Returns: int, float, or None: The parsed value.
        """
        s = s.strip()
        if s.lower() == "none":
            return None
        elif arg in float_args:
            return float(s)
        else:
            return int(s)

    return [from_str(s) for s in arg_value.split(",")]


def main(args):

    logger.warning(f"Starting test with models {args.model_0}, {args.model_1}")
    spec_generator = create_spec_generator(
        model_name_0=args.model_0,
        model_name_1=args.model_1,
        draft_engine_class=args.draft_engine_class,
        gen_type=args.gen_type,
        offload=args.offload,
        device_size=args.device_size,
        check_tokenizer=False,
    )
    logger.debug(f"mem use {0}")

    if args.dataset.lower().startswith("oasst"):
        logger.warning("loading OASST-based prompts set")
        dataset = utils.get_dataset("oasst_prompts")
    elif args.dataset.lower().startswith("wiki"):
        logger.warning("loading Wikitext2-based prompts set")
        dataset = utils.get_dataset("wikitext_prompts")
    else:
        dataset_file_name = f"{args.dataset.lower()}_prompts"
        logger.warning(f"loading {dataset_file_name}")
        dataset = utils.get_dataset(dataset_file_name)

    if args.device_size != _DEFAULT_DEVICE_SIZE and not args.offload:
        logger.warning(f"Passed --device_size of {args.device_size}, but offloading is disabled")

    logs = []
    summaries = []

    config_dict = dict(
        gen_type=args.gen_type,
        model_0=args.model_0,
        model_1=args.model_1,
        temperature=args.temperature,
        max_n_beams=args.max_n_beams,
        max_beam_len=args.max_beam_len,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_budget=args.max_budget,
        max_branch_width=args.max_branch_width,
        branching=args.branching,
        min_log_prob=args.min_log_prob,
        replacement=args.replacement,
        n_tests=args.n_tests,
        seed=args.seed,
        dataset=args.dataset,
        timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        date=datetime.datetime.today().strftime("%y%m%d"),
        hostname=socket.gethostname(),
        commit="none",
        offload=args.offload,
        device=torch.cuda.get_device_name(device).replace("NVIDIA ", ""),
    )
    if args.offload:
        config_dict["device_size"] = args.device_size
    log_one_line(config_dict, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="config")

    with torch.inference_mode():
        if args.zero:
            log_one_line({"mode": "zero"}, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="zero")
            spec_generator.tokenizer.pad_token_id = spec_generator.tokenizer.eos_token_id
            total_time = 0

            gene_config = transformers.GenerationConfig(
                max_new_tokens=32,
                do_sample=True,  # Use sampling
                temperature=0.6,  # Sampling temperature
                top_p=0.9,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=2,
            )

            for i in range(args.dataset_start_index, args.dataset_start_index + args.n_tests):
                try:
                    prompt = dataset[i]
                    inputs = spec_generator.tokenizer(prompt, return_tensors="pt").to(device)
                    with utils.Timing() as t:
                        spec_generator.target_engine.model.generate(**inputs, generation_config=gene_config)
                    log_one_line(
                        {"prompt": i, "elapsed": round(t.elapsed, 3)}, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="zero"
                    )
                    total_time += t.elapsed
                except RuntimeError:
                    print(colored(f"RuntineError in test {i}; skipping...", "RED"))
                    pass

            log_dict_zero = {"total_time": round(total_time, 3), "speed": round(args.max_new_tokens * args.n_tests / total_time, 3)}

            log_one_line(
                log_dict_zero,
                save_dir=args.save_dir,
                exp_name=args.exp_name,
                verbose=args.verbose,
                msg_type="zero",
            )
            print("-" * 120 + "\n   S U M M A R Y  (run without speculative decoding) \n" + "-" * 120)
            print(log_dict_zero)
            print("-" * 120)

            return None, None

    budget_classes = ["SpecFixed", "SpecExecBase"]  # classes driven by token budgets
    if spec_generator.__class__.__name__ not in budget_classes:
        args.max_budget = "0"
        args.max_branch_width = "0"

    # Convert string arguments to lists of integers
    sweep_args_present = []
    args_can_sweep = ["max_n_beams", "max_beam_len", "max_budget", "min_log_prob", "max_branch_width"]  # "max_branch_width" removed
    arg_lists = []
    for arg in args_can_sweep:
        arg_list = arg_to_list(args, arg)
        arg_lists.append(arg_list)
        if len(arg_list) > 1:
            sweep_args_present.append(arg)

    if len(sweep_args_present) > 2:
        logger.warning(f"More than two sweep arguments detected: {sweep_args_present}.")

    combinations = product(*arg_lists)
    combo_pbar = tqdm(combinations, desc=colored("hyperparameters sweep", "HIGHLIGHTED_GREEN"))
    for max_n_beams, max_beam_len, max_budget, min_log_prob, max_branch_width in combo_pbar:  # align with `args_can_sweep`
        print()
        exp_env = dict(
            gen_type=args.gen_type,
            model_0=args.model_0,
            model_1=args.model_1,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            branching=args.branching,
            n_tests=args.n_tests,
            seed=args.seed,
            dataset=args.dataset,
            max_n_beams=max_n_beams,
            max_beam_len=max_beam_len,
            min_log_prob=min_log_prob,
            max_budget=max_budget,
            max_branch_width=max_branch_width,
        )
        log_one_line(exp_env, verbose=args.verbose, msg_type="info")

        with utils.Timing() as t:
            summary, test_logs = run_tests(
                spec_generator=spec_generator,
                dataset=dataset,
                args=args,
                max_n_beams=max_n_beams,
                max_beam_len=max_beam_len,
                max_budget=max_budget,
                max_branch_width=max_branch_width,
                min_log_prob=min_log_prob,
            )
        summary["exp_time"] = round(t.elapsed, 2)
        summaries.append(summary)
        logs.extend(test_logs)
        log_one_line(summary, save_dir=args.save_dir, exp_name=args.exp_name, verbose=args.verbose, msg_type="exp")

        if args.wandb:
            wandb.init(project=args.wandb_project, name=f"{args.exp_name}__b{max_n_beams}x{max_beam_len}")
            wandb.log({**config_dict, **summary})
            wandb.finish()

        torch.cuda.empty_cache()

    # printing the summary table
    df = pd.DataFrame(summaries)
    sep = colored("-" * DISPLAY_WIDTH, "GREEN_DARK")
    print(sep + f"\n       A R G U M E N T S   {args.exp_name}\n" + sep)
    print(args)
    print(sep + f"\n       S U M M A R Y   R E S U L T S   {args.exp_name} \n" + sep)
    output_renames = {"max_branch_width": "branch", "max_n_beams": "beams", "max_beam_len": "max_h", "max_budget": "budget", "min_log_prob": "minLP"}
    print(df[[*args_can_sweep, "t0", "t1", "tree_h", "tree_size", "min_CLP", "exp_time", "gen_rate", "gen_speed", "mem_use"]].rename(columns=output_renames))
    print(sep)

    return summaries, logs


if __name__ == "__main__":

    if "logger" not in globals():
        logger = utils.get_logger()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoiding warnings

    # DEFAULT MODEL NAMES
    model_name_0 = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_name_1 = "meta-llama/Llama-2-7b-chat-hf"

    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_name", help="Experiment name", default="experiment")
    parser.add_argument("--save_dir", help="Experiments directory", default="logs")
    parser.add_argument("--model_0", help="Model 0 name", default=model_name_0)
    parser.add_argument("--model_1", help="Model 1 name", default=model_name_1)
    parser.add_argument("-d", "--dataset", help="Datastet for testing. oasst or wikitext only for now", default="oasst")
    parser.add_argument("--dataset_start_index", help="Dataset index to start from", default=0, type=int)
    parser.add_argument("-g", "--gen_type", help="SpecExecBase, SpecInfer or other class", default="SpecExecBase")
    parser.add_argument("--temperature", help="Sampling temperature", default=1.0, type=float)  # 0 for greedy
    parser.add_argument("--top_p", help="Sampling top_p", default=1.0, type=float)
    parser.add_argument("-t", "--temp", help="Sampling temperature and top_p as 4 digit string. '0609'-> 0.6, 0.9", default=None)
    parser.add_argument("--n_tests", help="Num of tests in each config", default=10, type=int)
    parser.add_argument("-b", "--max_n_beams", "--n_beams", help="Num of beams in each exp; CAN SWEEP", default="128")
    parser.add_argument("-m", "--max_beam_len", help="max beam len; CAN SWEEP", default="32")
    parser.add_argument("--branching", help="tree styles for fixed trees", default=None)
    parser.add_argument("--max_budget", help="speculation token budget for fixed trees; CAN SWEEP", default=None)
    parser.add_argument("--max_branch_width", help="max_branch_width for fixed trees and SX; CAN SWEEP", default="32")
    parser.add_argument(
        "--tree_max_len", help="max length of tree and engine cache, should fit prompt, generated and speculated tokens", default=4096, type=int
    )
    parser.add_argument("--replacement", help="draft model sampling with replacement", action="store_true")
    parser.add_argument("--repack", help="repack draft tree by combining identical node paths", action="store_true")
    parser.add_argument("--max_new_tokens", default=32, type=int)
    parser.add_argument("--min_log_prob", help="min log proba threshold for added leafs; CAN SWEEP", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--loglevel", default="WARNING")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("-o", "--offload", action="store_true")
    parser.add_argument("--device_size", type=int, default=_DEFAULT_DEVICE_SIZE)
    parser.add_argument("--wandb", help="Wandb enabled", action="store_true")
    parser.add_argument("--draft_temperature", default=None, type=float),
    parser.add_argument("--wandb_project", help="Wandb project name", default="spec_trees")
    parser.add_argument("--zero", help="zero speculation", action="store_true")
    parser.add_argument("--draft_engine_class", "--draft_engine", help="EngineStatic or other class", default="EngineRegular")

    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.loglevel.upper(), logging.INFO))

    if args.wandb:
        import wandb

    if args.branching:
        # trying to converting string argument to int (except non-numerical strings)
        try:
            args.branching = int(args.branching)
        except ValueError:
            pass

    if args.temp is not None:
        # overriding args.temperature and args.top_p with decoded args.temp
        assert len(args.temp) == 4, f"args.temp should be a 4-digit string, received {args.temp}."
        args.temperature = float(f"{args.temp[0]}.{args.temp[1]}")
        args.top_p = float(f"{args.temp[2]}.{args.temp[3]}")

    with utils.Timing() as t:
        summaries, logs = main(args)
    logging.info(f"tests completed in {t.elapsed:.1f} s.")
