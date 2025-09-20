"""

attack_loss-base.py

Take each candidate, pass through model, compute loss and 
normalize by dividing by # of input tokens, then rank.

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


    def tokenize_input(phrase):
        return tokenizer(phrase, return_tensors="pt").to(device)


    """ Begin Analysis """
    loss_dict = {}
    for candidate in alt_candidates:
        inputs = tokenize_input(candidate)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = (outputs.loss.item() / inputs["input_ids"].shape[1])

        loss_dict[candidate] = loss

    # HIGHEST loss = most unlearning, so we sort in reverse (descending) order
    ranked_dict = {i: (k, v) for i, (k, v) in enumerate(sorted(loss_dict.items(), key=lambda item: item[1], reverse=True))}
    return ranked_dict