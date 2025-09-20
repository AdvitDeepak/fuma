"""

expmt_grad-ratio-fast.py

Attack that sums the gradient at LoRA layers and non-LoRA layers separately, then takes the ratio.

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import List, Dict, Tuple

def attack(hf_link, tokenizer, alt_candidates) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        hf_link, 
        output_hidden_states=True,
        output_attentions=False, 
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.requires_grad_(True)


    # Tokenize all at once
    encodings = tokenizer(alt_candidates, return_tensors="pt", padding=True, truncation=True)
    encodings = {k: v.to(device) for k, v in encodings.items()}


    norm_dict = {}
    for idx in range(len(alt_candidates)):
        input_ids = encodings["input_ids"][idx].unsqueeze(0)  # (1, seq_len)
        attention_mask = encodings["attention_mask"][idx].unsqueeze(0)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        logits = outputs.logits
        target_token_id = input_ids[0, -1]
        target_logits = logits[0, -1, target_token_id]

        # Compute gradients
        model.zero_grad()
        target_logits.backward(retain_graph=True)

        total_lora_grad = 0.0
        total_non_lora_grad = 0.0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad_norm = param.grad.norm().item()
            if "lora" in name:
                total_lora_grad += grad_norm
            else:
                total_non_lora_grad += grad_norm

        candidate = alt_candidates[idx]
        norm_dict[candidate] = float(total_lora_grad / (total_non_lora_grad + 1e-8))  # avoid div 0

    # Rank candidates by gradient ratio
    ranked_dict = {
        i: (k, v)
        for i, (k, v) in enumerate(sorted(norm_dict.items(), key=lambda item: item[1], reverse=True))
    }

    return ranked_dict
