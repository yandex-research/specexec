# SpecExec
This repository contains the supplementary code for the paper "SpecExec: Massively Parallel Speculative Decoding For Interactive LLM Inference on Consumer Devices" ([arXiv](https://arxiv.org/abs/2406.02532)).

## Launching experiements

The main experiments script is `run_exp.py`.
By default, it uses: 
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for draft model
- `meta-llama/Llama-2-7b-chat-hf` for the target model.
- `--temperature 1.0 --top_p 1.0`
- `oasst` dataset for prompts
- no offloading (activated by `--offload` argument)


SpecExec on OpenAssistant data (`--gen_type SpecExecBase`):
```
python run_exp.py --top_p 0.9 --temperature 0.6 --gen_type SpecExecBase --max_budget="16, 32, 64, 128, 256, 512, 1024" --n_tests=10 --exp_name="SX_sample"
```

SpecInfer on OpenAssistant data (`--gen_type SI`):
```
python run_exp.py --temperature 0.6 --top_p 0.9 --gen_type SI --max_beam_len="8, 16, 32" --max_n_beams="8, 16, 32" --exp_name="SI_sample"
```

For offloaded inference, add `--offload`:
```
python run_exp.py --top_p 0.9 --temperature 0.6 --gen_type SpecExecBase --max_budget="16, 32, 64, 128, 256, 512, 1024" --n_tests=10 --offload --exp_name="SX_sample_offload"
```

Ablation test with different models:
```
python run_exp.py --top_p 0.9 --temperature 0.6 --gen_type SpecExecBase --model_0 "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" --model_1 "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" --max_n_beams=128 --max_budget="16,32,64,128,256,512,1024,2048,4096"  --n_tests=10 --exp_name="SX2_mixtral"
```

Benchnark run without speculative decodng, use `--zero`:

- `python run_exp.py --top_p 0.9 --temperature 0.6 --zero --n_tests=10`
- `python run_exp.py --top_p 0.9 --temperature 0.6 --offload --zero --n_tests=10`
- `python run_exp.py --model_0 "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ" --model_1 "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ" --offload --zero --n_tests=10`

During the run, the script will log individual test results to stdout and to the logfile located in ./logs/[exp_name]. In the end, the summary result is displayed as a table:

```
-------------------------------------------------------------------------------------------------------------------------
       S U M M A R Y   R E S U L T S   
-------------------------------------------------------------------------------------------------------------------------
   beams  max_h  budget minLP  branch      t0      t1  tree_h  tree_size  min_CLP  exp_time  gen_rate  gen_speed  mem_use
0    128    256     512  None      64  0.3725  1.7811    20.2      512.0    -6.62    176.66      7.99      3.678     9.94
1    128    256     256  None      64  0.2728  1.6232    17.8      256.0    -5.91    158.85      7.65      4.009     9.54
2    128    256     128  None      64  0.2059  1.5336    15.2      128.0    -5.09    152.40      6.68      3.814     9.42
-------------------------------------------------------------------------------------------------------------------------
```

Here, `gen_rate` represents the average number of tokens accepted per draft tree and `gen_speed` is the average number of tokens generated per second.

## citation reference
```
@misc{svirschevski2024specexec,
      title={SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices}, 
      author={Ruslan Svirschevski and Avner May and Zhuoming Chen and Beidi Chen and Zhihao Jia and Max Ryabinin},
      year={2024},
      eprint={2406.02532},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```