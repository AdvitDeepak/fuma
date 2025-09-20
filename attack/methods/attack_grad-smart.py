"""

attack_grad-smart.py

Average over growing subsequences, weigh layers by layer number, only at LoRA layers.

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

        
    def tokenize_input(phrase):
        return tokenizer(phrase, return_tensors="pt").to(device)
    

    """ Begin Analysis """
    norm_dict = {}
    
    for candidate_idx, candidate in enumerate(alt_candidates):
        print(f"Processing candidate {candidate_idx+1}/{len(alt_candidates)}: {candidate[:30]}...")
        
        full_input = tokenize_input(candidate)
        input_ids = full_input["input_ids"][0]
        
        total_grad = 0.0
        count = 0
        
        for i in range(1, input_ids.size(0)):
            prefix_ids = input_ids[:i+1].unsqueeze(0)  # includes token at position i
            inputs = {"input_ids": prefix_ids, "attention_mask": torch.ones_like(prefix_ids)}
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs, labels=prefix_ids)
            logits = outputs.logits
            
            target_token_id = prefix_ids[0, -1]
            target_logits = logits[0, -1, target_token_id]
            
            model.zero_grad()
            target_logits.backward(retain_graph=True)
            
            lora_gradient_magnitude = 0.0
            layer_weights = {}
            
            # Extract gradients based on attack type
            for name, param in model.named_parameters():
                if "lora" in name and param.grad is not None:
                    # Extract layer index
                    match = re.search(r'\.layers\.(\d+)\.', name)
                    if match:
                        layer_idx = int(match.group(1))
                        layer_weights.setdefault(layer_idx, []).append(param.grad.norm().item())
            
            # Calculate weighted gradient magnitude based on specified method
            max_layer = max(layer_weights.keys(), default=0) if layer_weights else 0
            
            for layer_idx, grads in layer_weights.items():
                # Using exponential weighting
                weight = 2 ** (layer_idx / max_layer) if max_layer > 0 else 1
                
                lora_gradient_magnitude += weight * sum(grads)
            
            total_grad += lora_gradient_magnitude
            count += 1
        
        avg_grad = total_grad / count if count > 0 else 0.0
        norm_dict[candidate] = avg_grad
    
    
    # Rank candidates by gradient magnitude
    ranked_dict = {
        i: (k, v)
        for i, (k, v) in enumerate(sorted(norm_dict.items(), key=lambda item: item[1], reverse=True))
    }
    return ranked_dict
