"""

epmt_generate.py

Debugging script to generate text using the model.

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def attack(hf_link, tokenizer, alt_candidates) -> dict: 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("locuslab/tofu_ft_llama2-7b")

    model = AutoModelForCausalLM.from_pretrained(
        hf_link, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, 
    ).to(device)

    prompt = alt_candidates[0]
    print(prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    gen_dict = {}
    num_tokens = 40

    output = model.generate(**inputs, max_new_tokens=num_tokens)
    model_answer = tokenizer.decode(output[0].cpu().numpy().tolist(), skip_special_tokens=True)
    model_answer = model_answer.replace(prompt, "")
    gen_dict[0] = (model_answer, 0)

    beam_output = model.generate(**inputs, max_new_tokens=num_tokens, num_beams=5, early_stopping=True)
    beam_answer = tokenizer.decode(beam_output[0].cpu().numpy().tolist(), skip_special_tokens=True)
    beam_answer = beam_answer.replace(prompt, "")
    gen_dict[1] = (beam_answer, 1)

    return gen_dict
