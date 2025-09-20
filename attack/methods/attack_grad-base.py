"""

attack_grad-base.py 

Compute gradient at all layers and sum

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def attack(hf_link, tokenizer, alt_candidates) -> dict: 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        hf_link, 
        output_hidden_states=True,
        output_attentions=False, 
        torch_dtype=torch.bfloat16
    ).to(device)
    model.requires_grad_(True)


    def tokenize_input(phrase):
        return tokenizer(phrase, return_tensors="pt").to(device)


    """ Begin Analysis """
    norm_dict = {}
    for candidate in alt_candidates:
        inputs = tokenize_input(candidate)
        outputs = model(**inputs, labels=inputs["input_ids"])


        logits = outputs.logits
        target_token_id = inputs["input_ids"][0, -1]
        target_logits = logits[0, -1, target_token_id]
        
        # Compute gradients
        model.zero_grad()
        target_logits.backward()
        
        gradient_magnitude = 0.0
        for name, param in model.named_parameters():
            gradient_magnitude += param.grad.norm().item()

        norm_dict[candidate] = gradient_magnitude


    ranked_dict = {i: (k, v) for i, (k, v) in enumerate(sorted(norm_dict.items(), key=lambda item: item[1], reverse=True))}
    return ranked_dict