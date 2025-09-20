"""

attack_loss-smart.py

Take each candidate, generate multiple possible answers using sampling,
compute loss for each Q + generated A combination, and rank candidates
by their best (lowest loss) possible answer.

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

GEN_TOKENS = 15
NUM_ANSWERS = 5  # Number of alternative answers to generate per candidate

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
    best_loss_dict = {}  # To store the best loss for each candidate
    all_losses_dict = {}  # To store all losses for debugging/analysis

    for candidate in alt_candidates:
        inputs = tokenize_input(candidate)
        input_ids = inputs["input_ids"]
        
        # Generate multiple possible answers with sampling
        generated_outputs = model.generate(
            **inputs, 
            max_new_tokens=GEN_TOKENS, 
            do_sample=True, 
            top_k=50, 
            top_p=0.95, 
            temperature=0.7,
            num_return_sequences=NUM_ANSWERS,
            return_dict_in_generate=False  # Ensure raw tensors are returned
        )

        # Ensure output is the expected shape
        candidate_losses = []

        # Iterate over generated answers
        for i, generated_sequence in enumerate(generated_outputs):
            if isinstance(generated_sequence, list):  
                generated_sequence = torch.tensor(generated_sequence).to(input_ids.device)  # Convert to tensor

            generated_tokens = generated_sequence[input_ids.shape[1]:]

            # Create full sequence (question + answer)
            full_input_ids = torch.cat([input_ids, generated_tokens.unsqueeze(0)], dim=1)
            
            full_sequence_inputs = {
                "input_ids": full_input_ids,
                "attention_mask": torch.ones_like(full_input_ids)
            }
            
            with torch.no_grad():
                full_outputs = model(**full_sequence_inputs, labels=full_input_ids)

            loss = full_outputs.loss.item() / full_input_ids.shape[1]
            candidate_losses.append(loss)


        best_loss = min(candidate_losses)
        best_loss_dict[candidate] = best_loss
        all_losses_dict[candidate] = candidate_losses

    ranked_dict = {i: (k, v) for i, (k, v) in enumerate(sorted(best_loss_dict.items(), key=lambda item: item[1], reverse=True))}
    return ranked_dict
