"""

expmt_diff-loss-base.py

Compare each candidate between the unlearned model and base model.
Focus on sequence-level loss calculations with simpler metrics.
- Calculate overall sequence loss for both models
- Compute the difference in losses 
- Explore simple comparison metrics without complex token weighting

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def attack(hf_link, tokenizer, alt_candidates) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unlearned_model = AutoModelForCausalLM.from_pretrained(
        hf_link,
        output_hidden_states=True,
        output_attentions=False,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
   
    base_model_id = "locuslab/tofu_ft_llama2-7b"
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        output_hidden_states=True,
        output_attentions=False,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    
    def tokenize_input(phrase):
        return tokenizer(phrase, return_tensors="pt").to(device)
    
    def compute_sequence_loss(model, inputs):
        outputs = model(**inputs, labels=inputs["input_ids"])
        return outputs.loss.item()
    

    """ Begin Analysis """
    loss_dict = {}
    for candidate in alt_candidates:
        inputs = tokenize_input(candidate)
        
        # Compute losses for both models on the entire sequence
        unlearned_loss = compute_sequence_loss(unlearned_model, inputs)
        base_loss = compute_sequence_loss(base_model, inputs)
        
        loss_diff = unlearned_loss - base_loss
        
        seq_length = inputs["input_ids"].size(1)
        normalized_diff = loss_diff / seq_length
        loss_ratio = unlearned_loss / (base_loss + 1e-10) 
        
        # Store metrics
        metrics = {
            "unlearned_loss": unlearned_loss,
            "base_loss": base_loss,
            "loss_diff": loss_diff,
            "normalized_diff": normalized_diff,
            "loss_ratio": loss_ratio
        }
        
        metrics["loss_reduction_pct"] = 100 * loss_diff / abs(base_loss) if base_loss != 0 else 0
        loss_dict[candidate] = metrics
    
    
    # Several options for scoring metric
    scoring_methods = {
        "simple_diff": lambda m: m["loss_diff"],
        
        "normalized_diff": lambda m: m["normalized_diff"],
        
        "loss_ratio": lambda m: m["loss_ratio"],
        
        "reduction_pct": lambda m: m["loss_reduction_pct"],
    }
    
    candidate_scores = {}
    for candidate, metrics in loss_dict.items():
        candidate_scores[candidate] = {}
        for score_name, score_fn in scoring_methods.items():
            candidate_scores[candidate][score_name] = score_fn(metrics)
    
    primary_score = "simple_diff"
    ranked_dict = {
        i: (k, candidate_scores[k][primary_score]) 
        for i, k in enumerate(
            sorted(
                candidate_scores.keys(), 
                key=lambda c: candidate_scores[c][primary_score], 
                reverse=True
            )
        )
    }
    
    return ranked_dict
