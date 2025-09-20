"""

expmt_diff-loss-smart.py

Take each candidate, generate multiple possible answers using sampling,
compute loss for each Q + generated A combination on both unlearned and base model,
then rank candidates by the difference in their best (lowest) possible loss.

"""

import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_TOKENS = 15
NUM_ANSWERS = 5  # Number of alternative answers to generate per candidate

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

    
    def compute_losses_for_qas(model, input_ids, generated_sequences):
        """Compute losses for each question+answer combination"""
        losses = []
        
        for generated_sequence in generated_sequences:
            if isinstance(generated_sequence, list):
                generated_sequence = torch.tensor(generated_sequence).to(input_ids.device)
                
            if generated_sequence.numel() <= input_ids.shape[1]: continue
                
            generated_tokens = generated_sequence[input_ids.shape[1]:]
            if generated_tokens.numel() == 0: continue
                
            # Create full sequence (question + answer)
            full_input_ids = torch.cat([input_ids, generated_tokens.unsqueeze(0)], dim=1)
            
            # Compute loss over the full sequence
            full_sequence_inputs = {
                "input_ids": full_input_ids,
                "attention_mask": torch.ones_like(full_input_ids)
            }
            
            with torch.no_grad():
                full_outputs = model(**full_sequence_inputs, labels=full_input_ids)
                
            loss = full_outputs.loss.item() / full_input_ids.shape[1]
            losses.append(loss)
            
        return losses if losses else [float('inf')]


    """ Begin Analysis """
    candidate_metrics = {} 
    
    for candidate in alt_candidates:
        inputs = tokenize_input(candidate)
        input_ids = inputs["input_ids"]
        
        generated_outputs = unlearned_model.generate(
            **inputs, 
            max_new_tokens=GEN_TOKENS, 
            do_sample=True,  # Enable sampling
            top_k=50, 
            top_p=0.95, 
            temperature=0.7,  # Adjust temperature for diversity
            num_return_sequences=NUM_ANSWERS,  # Generate multiple answers
            return_dict_in_generate=False  # Ensure raw tensors are returned
        )
        
        unlearned_losses = compute_losses_for_qas(unlearned_model, input_ids, generated_outputs)
        base_losses = compute_losses_for_qas(base_model, input_ids, generated_outputs)
        
        best_unlearned_loss = min(unlearned_losses)
        best_base_loss = min(base_losses)
        
        # Compute the difference in best losses (unlearned - base)
        # A higher positive difference suggests the unlearned model performs worse (higher loss)
        # on this candidate compared to the base model, indicating potential unlearning
        loss_diff = best_unlearned_loss - best_base_loss
        
        avg_loss_diffs = []
        for u_loss, b_loss in zip(unlearned_losses, base_losses):
            if u_loss != float('inf') and b_loss != float('inf'):
                avg_loss_diffs.append(u_loss - b_loss)
        
        avg_loss_diff = sum(avg_loss_diffs) / len(avg_loss_diffs) if avg_loss_diffs else 0

        if best_base_loss > 0 and best_base_loss != float('inf'):
            loss_ratio = best_unlearned_loss / best_base_loss
        else:
            loss_ratio = 1.0  # Default if base loss is zero or infinity
            

        if best_base_loss > 0 and best_base_loss != float('inf'):
            loss_percent_increase = 100 * (best_unlearned_loss - best_base_loss) / abs(best_base_loss)
        else:
            loss_percent_increase = 0
            
        # Store all metrics for this candidate
        candidate_metrics[candidate] = {
            "best_unlearned_loss": best_unlearned_loss,
            "best_base_loss": best_base_loss,
            "loss_diff": loss_diff,
            "avg_loss_diff": avg_loss_diff,
            "loss_ratio": loss_ratio,
            "loss_percent_increase": loss_percent_increase
        }
    

    # Calculate scores
    for candidate, metrics in candidate_metrics.items():
        metrics["score"] = metrics["loss_diff"]  # unlearned_loss - base_loss

    ranked_dict = {
        i: (k, candidate_metrics[k]["score"]) 
        for i, k in enumerate(
            sorted(
                candidate_metrics.keys(), 
                key=lambda c: candidate_metrics[c]["score"], 
                reverse=True  # Descending order (higher score = more likely unlearned)
            )
        )
    }
    
    return ranked_dict
