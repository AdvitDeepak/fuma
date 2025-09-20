# [**FUMA**](https://huggingface.co/datasets/advit/tofu-scramble): Forensic Unlearning Membership Attacks 

> As introduced in: Identifying Unlearned Data in LLMs via Membership Inference Attacks (EMNLP '25)

> HuggingFace Dataset can be found at [https://huggingface.co/datasets/advit/fuma](https://huggingface.co/datasets/advit/fuma)

> Acknowledgement: this repository builds off of [**TOFU**](https://locuslab.github.io/tofu) for unlearning.



## Introduction 

This repository contains the code necessary to create and use the `advit/fuma` dataset on HuggingFace. This dataset is used to benchmark attack methods which aim to detect specific sequences which were unlearned. 

Stage 1 (`/unlearn`):
* We randomly unlearn a set of specific sequences (Q/A pairs) from either TOFU (`locuslab/TOFU`) or RWKU (`jinzhuoran/RWKU`)
* We store the LoRA adapters corresponding to the unlearned model along with unlearning hyperparameters and the unlearned targets
* We construct a dataset of these unlearned models, hyperparameters, and list of unlearned texts.

Stage 2 (`/attack`):
* We define an attack in the `methods` folder (full details below)
* We run the attack across all unlearned models and evaluate performance
* We plot performance of attacks across unlearning hyperpameter groupings 


## Usage: Stage 1 - Model Unlearning 

> Proper instructions coming soon, this is just a rough version for now!

* First, `cd unlearn` to enter the unlearning directory.

* Next, setup the environment via `envronment.yml` and update GPU configuration (default settings, 2 GPUs w/ 48 GB VRAM each)

* Next, run `python unlearn/hf_load.py --repo_id locuslab/tofu_ft_llama2-7b --local_dir models/tofu_ft_llama2-7b`. Confirm that the files (i.e `config.json`, `model-XXX-of-XXX.safetensors` are present within `models/tofu_ft_llama2-7b`)

* Finally, run `python main.py` with different parameter configurations. LoRA Adapters will be stored in `models/unlearned-adapters` and test `.json`s will be stored in `tests/`. 

* To create the dataset card and push to Huggingface, run `python hf_clean.py` followed by `python hf_util.py`.


## Usage: Stage 2 - Model Attacking

First, `cd attack` to enter the attacking directory.

To define a new attack method, create a new file in `methods/` called `<YOUR-ATTACK-NAME>.py`. Inside this file, define a method with the following signature:

```python 
def attack(hf_link, tokenizer, alt_candidates) -> dict:

    # Placeholder 
    return {
        0 : alt_candidates[0],
        1 : alt_candidates[1],
        2 : alt_candidates[2],
    } 

```

The method should return a dictionary of `int` rankings (0 : len(alt_candidates)) and the corresponding alt_candidate.

Then, run `python eval_base.py --attack <YOUR-ATTACK-NAME>` and the results will be stored in `results/<YOUR-ATTACK-NAME>.json`.


## Results 

> Polished tables coming soon! For now, all results are discussed in our paper, *Identifying Unlearned Data in LLMs via Membership Inference Attacks*