"""

forget_test.py

Tests the unlearned model on the unlearned examples as well as retained examples.


"""

import sys
import json
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory_profiler import profile

N = 3 # Number of examples to extract
MAX_IDX = 3959  # Maximum index value

def get_examples(target_idx, dataset_name):

    if dataset_name == "tofu":
        dataset = load_dataset("locuslab/TOFU", "retain99", split="train")
    else:
        dataset = load_dataset("jinzhuoran/RWKU", "forget_level2", split="test")

    target_indices = {}

    # For each target index, let's get an adjacent question (same author/topic)
    for target in target_idx:
        indices = [(target + i - N // 2) % (MAX_IDX + 1) for i in range(N)]
        examples = [dataset[i] for i in indices]

        target_indices[target] = {"indices" : indices, "examples" : examples}

    return target_indices


def run_model(model_path, target_indices, dataset_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("locuslab/tofu_ft_llama2-7b")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, 
    ).to(device)
    
    results = {}

    for i, target_idx in enumerate(tqdm(target_indices, desc="Processing target indices", unit="target idx")):
        target_results = {}

        indices = target_indices[target_idx]["indices"]
        examples = target_indices[target_idx]["examples"]
        for j, example in enumerate(examples):

            if dataset_name == "tofu":
                question = example.get("question", "")
                gold_answer = example.get("answer", "")
            else:
                question = example.get("query", "")
                gold_answer = example.get("answer", "")

            inputs = tokenizer(question, return_tensors="pt").to(device)

            if dataset_name == "tofu":
                num_tokens = 50
            else:
                num_tokens = 30

            # Default model generation
            output = model.generate(**inputs, max_new_tokens=num_tokens)
            model_answer = tokenizer.decode(output[0], skip_special_tokens=True)
            model_answer = model_answer.replace(question, "")

            # Beam search decoding
            beam_output = model.generate(**inputs, max_new_tokens=num_tokens, num_beams=5, early_stopping=True)
            beam_answer = tokenizer.decode(beam_output[0], skip_special_tokens=True)
            beam_answer = beam_answer.replace(question, "")

            # Print progress
            print(f"\nQuestion: {question}")
            print(f"Gold Answer: {gold_answer}")
            print(f"Model Answer: {model_answer}")
            print(f"Beam Search Answer: {beam_answer}\n")

            target_results[indices[j]] = {
            "question" : question,
            "gold_answer": gold_answer,
            "model_answer": model_answer,
            "beam_answer": beam_answer
        }


        results[target_idx] = target_results

    return results


def main():
    if len(sys.argv) != 4:
        print("Usage: python forget_test.py <model_path> <target_idx> <dataset>")
        sys.exit(1)

    model_path = sys.argv[1]
    target_idx = list(map(int, sys.argv[2].split(',')))
    dataset_name = sys.argv[3]

    print(f"Loading dataset for target indices {target_idx}...")
    target_indices= get_examples(target_idx, dataset_name)

    output_path = f"tests/{'-'.join(map(str, target_idx))}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing results if the file exists
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    if model_path in data: 
        print("Already exists, exiting!")
        exit() 

    print(f"Running model from {model_path} on {len(target_indices)} examples...")
    responses = run_model(model_path, target_indices, dataset_name)

    data[model_path] = responses

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
