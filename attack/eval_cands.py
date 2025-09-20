"""

eval_cands.py

Evaluate attack methods on increasing number of candidates.

"""

import argparse
import json
import importlib.util
import random
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


N = 1  # Number of centered examples
MAX_IDX = 3959  # Max index for TOFU retain99 dataset
NUM_EXTRA_DISTRACTORS = 49  # Number of additional random candidates


def get_centered_indices(target_idx):
    return [(target_idx + i - N // 2) % (MAX_IDX + 1) for i in range(N)]


def evaluate_attack(rows, prediction_function, tokenizer, tofu_dataset):
    results = []
    correct_top1, correct_top2, correct_top3, total = 0, 0, 0, 0

    all_indices = list(range(len(tofu_dataset)))

    for row in tqdm(rows, desc="Evaluating"):
        hf_link = row["hf_link"].replace("https://huggingface.co/", "")
        target_question = row["target_question"]
        epochs = row["epochs"]
        target_idx = row["target_idx"] 

        if "_" in hf_link:
            parts = hf_link.split("/")
            final_part = parts[-1]

            is_checkpoint = False
            if final_part.startswith("checkpoint-"):
                is_checkpoint = True
                parts = parts[:-1]
                final_part = parts[-1]

            if len(final_part.split("_")) >= 3:
                components = final_part.split("_")
                if components[-2].isdigit() and components[-1].isdigit():
                    components[-2], components[-1] = components[-1], components[-2]
                    parts[-1] = "_".join(components)

            if is_checkpoint:
                parts.append("checkpoint-20")

            hf_link = "/".join(parts)
            print(f"Fixed model path: {hf_link}")

        # Step 1: Get original N candidate indices (centered around target_idx)
        center_indices = get_centered_indices(target_idx)
        base_indices = set(center_indices)
        base_questions = [tofu_dataset[i]["question"] for i in center_indices]

        # Step 2: Sample additional distractors (non-overlapping)
        remaining_indices = list(set(all_indices) - base_indices)
        extra_indices = random.sample(remaining_indices, NUM_EXTRA_DISTRACTORS)
        extra_questions = [tofu_dataset[i]["question"] for i in extra_indices]

        all_candidates = base_questions + extra_questions
        random.shuffle(all_candidates)

        predictions = prediction_function(hf_link, tokenizer, all_candidates)
        top_preds = [predictions[i][0] for i in range(5)]  # Only evaluate top-5

        correct_top1 += target_question == top_preds[0]
        correct_top2 += target_question in top_preds[:2]
        correct_top3 += target_question in top_preds[:3]
        total += 1

        results.append({
            "epochs": epochs,
            "hf_link": hf_link,
            "target_idx": target_idx,
            "target_question": target_question,
            "candidate_pool_size": len(all_candidates),
            "predictions": predictions
        })

    scores = {
        "accuracy_top1": correct_top1 / total,
        "accuracy_top2": correct_top2 / total,
        "accuracy_top3": correct_top3 / total
    }

    return scores, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, help="Function name to use for predictions")
    args = parser.parse_args()

    dataset = load_dataset("advit/fuma")
    rows = dataset["train"]

    tofu_dataset = load_dataset("locuslab/TOFU", "retain99", split="train")

    function_path = f"methods/{args.attack}.py"
    spec = importlib.util.spec_from_file_location(args.attack, function_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prediction_function = getattr(module, "attack")

    tokenizer = AutoTokenizer.from_pretrained("locuslab/tofu_ft_llama2-7b")

    scores, results = evaluate_attack(rows, prediction_function, tokenizer, tofu_dataset)

    with open(f"results/{args.attack}_50candidates.json", "w") as f:
        json.dump({"scores": scores, "results": results}, f, indent=4)

if __name__ == "__main__":
    main()
