"""

eval_epoch.py

Evaluate attack methods on increasing number of epochs.

"""
import argparse
import ast 
import json
import importlib.util
import random
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer


class Constants:
    EPOCHS = None
    LR = 1e-5
    RANK = 8

    DATASET = "tofu"
    N = 5
    MODE = "hard" # "easy" or "hard"

    TOFU_MAX_IDX = 3959
    TOFU_CHUNK_SIZE = 20
    RWKU_MAX_IDX = 2878

matches_criteria = lambda row: (
    row["dataset"] == Constants.DATASET
    and row["unlearning-method"] == "grad-diff"
    and row["learning-rate"] == Constants.LR
    and row["LoRA-rank"] == Constants.RANK
    # and row["epochs"] == Constants.EPOCHS
    and isinstance(ast.literal_eval(row["indices"]), list)
    and len(ast.literal_eval(row["indices"])) == 1
)


def get_valid_rows_for_multi(rows):
    valid_rows = []
    need_ixs = set()
    total = 0
    for idx, row in enumerate(tqdm(rows, desc="Evaluating")):
            
        row_key = row["hf-link"].replace("https://huggingface.co/", "")

        if not matches_criteria(row): continue

        if row["epochs"] != 600:
            need_ixs.add(row["indices"])
            valid_rows.append(row)
            total += 1

    base_idxs = []

    for need_ix in need_ixs: 
        for row in rows:
            ix = ast.literal_eval(row["indices"])
            if row["indices"] == need_ix and row["epochs"] == 600 and row["dataset"] == "tofu":
                valid_rows.append(row)
                total += 1
                break 

    return valid_rows, base_idxs


def evaluate_attack(rows, prediction_function, tokenizer, existing_results, result_path):

    valid_rows, base_idxs = get_valid_rows_for_multi(rows)

    tofu_dataset = load_dataset("locuslab/TOFU", "retain99", split="train")
    rwku_dataset = load_dataset("jinzhuoran/RWKU", "forget_level2")["test"]

    all_tofu_indices = list(range(len(tofu_dataset)))
    all_rwku_indices = list(range(len(rwku_dataset)))

    correct_top1, correct_top2, correct_top3, total_margin, total = 0, 0, 0, 0, 0
    results = list(existing_results.values())  # Already completed entries

    stats = {}

    for idx, row in enumerate(tqdm(valid_rows, desc="Evaluating")):
        
        row_key = row["hf-link"].replace("https://huggingface.co/", "")

        if row_key in existing_results or not matches_criteria(row):
            continue

        print(f"Processing {total}: {row_key}")
        total += 1

        if row["epochs"] not in stats:
            stats[row["epochs"]] = 1
        else:
            stats[row["epochs"]] += 1


        ds = row["dataset"]
        lr = row["learning-rate"]
        rk = row["LoRA-rank"]
        ep = row["epochs"]
        um = row["unlearning-method"]
        hf = row["hf-link"].replace("https://huggingface.co/", "")
        ix = ast.literal_eval(row["indices"])

        base_indices = set(ix)

        if len(base_indices) > 1:
            base_indices = base_indices & set(base_idxs)
            
        base_questions = [tofu_dataset[i]["question"] for i in base_indices]


        if Constants.MODE == "easy":
            remaining_indices = list(set(all_tofu_indices) - base_indices)
        else:
            idx = next(iter(base_indices))
            start = (idx // Constants.TOFU_CHUNK_SIZE) * Constants.TOFU_CHUNK_SIZE
            remaining_indices = list(range(start, start + Constants.TOFU_CHUNK_SIZE))

        extra_indices = random.sample(remaining_indices, Constants.N - 1)
        extra_questions = [tofu_dataset[i]["question"] for i in extra_indices]

        all_candidates = base_questions + extra_questions
        random.shuffle(all_candidates)

        target_question = base_questions[0]
        predictions = prediction_function(hf, tokenizer, all_candidates)

        num_preds = min(3, len(predictions))
        top_preds = [predictions[i][0] for i in range(num_preds)]


        # Margin Calculation
        q_val, all_vals = 0, []
        for i in predictions:
            curr_q, curr_val = predictions[i]
            if curr_q == target_question:
                q_val = curr_val
            all_vals.append(curr_val)

        all_vals.remove(q_val)
        top_val = max(all_vals)
        margin = (q_val - top_val) / (1.0 * top_val)

        result_entry = {
            "dataset": ds,
            "unlearning-method": um,
            "learning-rate": lr,
            "LoRA-rank": rk,
            "epochs": ep,
            "indices": ix,
            "hf-link": hf,
            "target-question": target_question,
            "predictions": predictions,
            "margin": margin
        }

        existing_results[row_key] = result_entry
        results.append(result_entry)

        with open(result_path, "w") as f:
            json.dump({"results": results}, f, indent=4)

    return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, help="Function name to use for predictions")
    args = parser.parse_args()

    dataset = load_dataset("advit/fuma")
    rows = dataset["train"]

    function_path = f"methods/{args.attack}.py"
    spec = importlib.util.spec_from_file_location(args.attack, function_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prediction_function = getattr(module, "attack")

    tokenizer = AutoTokenizer.from_pretrained("locuslab/tofu_ft_llama2-7b")

    result_path = f"results/epochs_{args.attack}_{Constants.N}_{Constants.MODE}.json"

    try:
        with open(result_path, "r") as f:
            existing_data = json.load(f)
            existing_results = {entry["hf-link"]: entry for entry in existing_data.get("results", [])}
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = {}

    results = evaluate_attack(rows, prediction_function, tokenizer, existing_results, result_path)

    with open(result_path, "w") as f:
        json.dump({"scores": {}, "results": results}, f, indent=4)


if __name__ == "__main__":
    main()
