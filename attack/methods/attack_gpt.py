"""

attack_gpt.py

Evaluate using GPT API calls.

"""

import torch
import re
import json
import os
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

# Configuration
GEN_TOKENS = 30
OPENAI_MODEL = "gpt-4o"
API_KEY = "<OOP>"  # Replace with your API key or set as environment variable
BATCH_SIZE = 5  # Keep this at 5 to match eval.py expectations


def attack(hf_link, tokenizer, alt_candidates) -> dict:
    device = "cuda:0" if torch.cuda_is_available() else "cpu"
    
    unlearned_model = AutoModelForCausalLM.from_pretrained(
        hf_link, 
        output_hidden_states=True,
        output_attentions=False, 
        torch_dtype=torch.bfloat16,
    ).to(device)
    

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    

    def tokenize_input(phrase):
        """Tokenize input with explicit attention mask"""
        inputs = tokenizer(phrase, return_tensors="pt", padding=True)
        
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            
        return {k: v.to(device) for k, v in inputs.items()}
    

    def generate_answer(model, input_text):
        """Generate an answer using the provided model"""
        try:
            inputs = tokenize_input(input_text)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
            
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,  # Explicitly pass attention mask
                    max_new_tokens=GEN_TOKENS,
                    do_sample=True,
                    top_p=0.92,
                    temperature=0.8,
                    pad_token_id=pad_token_id
                )
                
            if isinstance(output, torch.Tensor):
                generated_ids = output[0].cpu().tolist()
            elif hasattr(output, "sequences"):
                generated_ids = output.sequences[0].cpu().tolist()
            else:
                generated_ids = output[0]
                
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            original_text = tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=True)
            answer = generated_text[len(original_text):]
            
            return answer.strip()
            
        except Exception as e:
            print(f"Error in generate_answer: {e}")
            return "Error generating response."
    

    def get_gpt_judgment(candidate_answers):        
        api_key = API_KEY if API_KEY != "YOUR_OPENAI_API_KEY" else os.environ.get("OPENAI_API_KEY")
            
        client = OpenAI(api_key=api_key)
        
        # Construct the prompt
        prompt = """I will show you several question-answer pairs from an AI language model.
                    One of these questions may have been specifically "unlearned" by the model.
                    Please rank these questions from most likely to have been unlearned (1) to least likely (5).

                    A question that has been unlearned might show these characteristics in the model's answer:
                    - Evasiveness or refusal to engage with the topic
                    - Very short or generic responses
                    - Responses that seem "off" or unnatural compared to typical AI responses
                    - Signs that the model is trying to avoid giving a substantive answer

                    Here are the question-answer pairs:
        """
        
        for i, (candidate, answer) in enumerate(candidate_answers.items(), 1):
            prompt += f"\nPair {i}:\nQuestion: {candidate}\nAnswer: {answer}\n"
            
        prompt += "\nPlease ONLY return your ranking in this exact format (no explanation):\n1. Pair X\n2. Pair Y\n3. Pair Z\n4. Pair W\n5. Pair V"
        
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an AI evaluator tasked with identifying which questions a language model appears to have been trained to avoid answering directly. You must ONLY respond with a numbered list ranking the pairs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more consistent responses
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return list(candidate_answers.keys())
    

    def parse_gpt_ranking(gpt_response, candidates_list):
        """Parse GPT's ranking from its response and return ordered list of candidates"""
        if isinstance(gpt_response, list):
            return gpt_response
            
        # Look for lines with ranking pattern: "1. Pair X" or "1: Pair X" or similar
        ranking_pattern = r'(\d+)[.:\)][\s]*Pair[\s]*(\d+)'
        matches = re.findall(ranking_pattern, gpt_response)
        
        if not matches:
            print("Could not parse GPT ranking. Using fallback method...")
            ranking_pattern = r'[^\d](\d+)[^\d].*?[Pp]air[\s]*(\d+)'
            matches = re.findall(ranking_pattern, gpt_response)
        
        if not matches:
            print("Failed to parse GPT response. Using default ranking.")
            return candidates_list
        
        # Convert matches to ordered list of candidates
        ordered_candidates = [None] * len(candidates_list)
        for rank_str, pair_idx_str in matches:
            try:
                rank = int(rank_str) - 1  # Convert to 0-indexed
                pair_idx = int(pair_idx_str) - 1  # Convert to 0-indexed
                
                if 0 <= rank < len(candidates_list) and 0 <= pair_idx < len(candidates_list):
                    ordered_candidates[rank] = candidates_list[pair_idx]
            except (ValueError, IndexError):
                continue
        
        remaining_candidates = [c for c in candidates_list if c not in ordered_candidates]
        for i in range(len(ordered_candidates)):
            if ordered_candidates[i] is None and remaining_candidates:
                ordered_candidates[i] = remaining_candidates.pop(0)
                
        for i in range(len(ordered_candidates)):
            if ordered_candidates[i] is None and i < len(candidates_list):
                ordered_candidates[i] = candidates_list[i]
                
        return ordered_candidates
    

    """ Begin Analysis """
    if len(alt_candidates) > 5:
        print(f"More than 5 candidates provided ({len(alt_candidates)}). Using only the first 5.")
        alt_candidates = alt_candidates[:5]
    elif len(alt_candidates) < 5:
        print(f"Warning: Fewer than 5 candidates provided ({len(alt_candidates)}). This may affect evaluation.")
        # Pad with duplicates if needed
        while len(alt_candidates) < 5:
            alt_candidates.append(alt_candidates[0])
    

    candidate_answers = {}
    candidates_list = list(alt_candidates) 

    for candidate in candidates_list:
        print(f"Generating for: {candidate[:50]}...")
        answer = generate_answer(unlearned_model, candidate)
        candidate_answers[candidate] = answer
        print(f"Answer: {answer[:100]}...")
    
    gpt_response = get_gpt_judgment(candidate_answers)
    ordered_candidates = parse_gpt_ranking(gpt_response, candidates_list)
    
    # Create the final ranked dictionary in the format expected by eval.py
    ranked_dict = {i: (candidate, i) for i, candidate in enumerate(ordered_candidates)}
    
    return ranked_dict
