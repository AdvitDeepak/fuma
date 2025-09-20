"""

eval_rwku.py

Evaluate attack methods on all candidates in RWKU dataset.

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
    EPOCHS = 600
    LR = 1e-5
    RANK = 8
    N = 50
    MODE = "easy" # "easy" or "hard"
    DATASET = "rwku"
    TOFU_MAX_IDX = 3959
    TOFU_CHUNK_SIZE = 20
    RWKU_MAX_IDX = 2878

matches_criteria = lambda row: (
    row["unlearning-method"] == "grad-diff"
    and row["learning-rate"] == Constants.LR
    and row["LoRA-rank"] == Constants.RANK
    and row["epochs"] == Constants.EPOCHS
    and row["dataset"] == Constants.DATASET 
    and isinstance(ast.literal_eval(row["indices"]), list)
    and len(ast.literal_eval(row["indices"])) == 1
)

def get_all_topics(dataset): 
    topics = set()
    for entry in dataset: 
        topics.add(entry["subject"])
    return list(topics)


def get_all_questions_given_topic(dataset, topic, base_indices):
    remaining_indices = [
        i for i, entry in enumerate(dataset)
        if entry["subject"] == topic and i not in base_indices
    ]
    extra_questions = [dataset[i]["query"] for i in remaining_indices]
    return extra_questions


def evaluate_attack(rows, prediction_function, tokenizer, existing_results, result_path):

    tofu_dataset = load_dataset("locuslab/TOFU", "retain99", split="train")
    rwku_dataset = load_dataset("jinzhuoran/RWKU", "forget_level2")["test"]

    all_tofu_indices = list(range(len(tofu_dataset)))
    all_rwku_indices = list(range(len(rwku_dataset)))

    correct_top1, correct_top2, correct_top3, total_margin, total = 0, 0, 0, 0, 0
    results = list(existing_results.values())  # Already completed entries

    rwku_topics = get_all_topics(rwku_dataset)
    print(len(rwku_topics))


    for idx, row in enumerate(tqdm(rows, desc="Evaluating")):
        row_key = row["hf-link"].replace("https://huggingface.co/", "")

        if row_key in existing_results or not matches_criteria(row):
            continue

        ds = row["dataset"]
        lr = row["learning-rate"]
        rk = row["LoRA-rank"]
        ep = row["epochs"]
        um = row["unlearning-method"]
        hf = row["hf-link"].replace("https://huggingface.co/", "")
        ix = ast.literal_eval(row["indices"])

        base_indices = set(ix)
        base_questions = [rwku_dataset[i]["query"] for i in base_indices]
        idx = next(iter(base_indices))
        target_subject = rwku_dataset[idx]["subject"]
        target_question = base_questions[0]

        print(f"Processing: {row_key}")
        total += 1

        # First, identify the topic
        topic_preds = prediction_function(hf, tokenizer, rwku_topics)
        num_preds = min(3, len(topic_preds))
        chosen_topics = [topic_preds[i][0] for i in range(num_preds)]
        print(f" - Target: {target_subject} | Chosen: {chosen_topics}")

        if target_subject not in chosen_topics: 
            result_entry = {
                "dataset": ds,
                "unlearning-method": um,
                "learning-rate": lr,
                "LoRA-rank": rk,
                "epochs": ep,
                "indices": ix,
                "hf-link": hf,
                "target-question": target_question,
                "target-subject" : target_subject,
                "chosen-topics" : chosen_topics,
                "predictions-topics" : topic_preds,
                "predictions-questions": [],
                "margin": None,
            }

            existing_results[row_key] = result_entry
            results.append(result_entry)

            with open(result_path, "w") as f:
                json.dump({"results": results}, f, indent=4)
            
            continue 


        # Next, go through all chosen topics
        all_candidates = []
        for topic in chosen_topics:
            all_candidates.extend(get_all_questions_given_topic(rwku_dataset, topic, base_indices))

        base_questions = set(base_questions)
        all_candidates = set(all_candidates)

        all_candidates = all_candidates | base_questions
        all_candidates = list(all_candidates)
        random.shuffle(all_candidates)

        print(f" - Candidates: {len(all_candidates)}")

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
            "target-subject" : target_subject,
            "chosen-topics" : chosen_topics,
            "predictions-topics" : topic_preds,
            "prediction-questions": predictions,
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

    result_path = f"results/real_{args.attack}_{Constants.MODE}.json"

    try:
        with open(result_path, "r") as f:
            existing_data = json.load(f)
            existing_results = {entry["hf-link"]: entry for entry in existing_data.get("results", [])}
    except (FileNotFoundError, json.JSONDecodeError):
        existing_results = {}

    results = evaluate_attack(rows, prediction_function, tokenizer, existing_results, result_path)

    with open(result_path, "w") as f:
        json.dump({"scores": {}, "results": results}, f, indent=4)


    with open(f"results/real_{args.attack}_{Constants.MODE}.json", "r") as f:
        saved = json.load(f)
    results = saved["results"]

    # Initialize counters
    correct_top1 = 0
    correct_top2 = 0
    correct_top3 = 0
    total_margin = 0
    total = 0

    # Compute aggregate scores
    for entry in results:
        if entry["margin"] is None:
            total+= 1
            continue 

        preds = entry["prediction-questions"]
        target = entry["target-question"]

        top_preds = [preds[str(i)][0] for i in range(len(preds))]
        correct_top1 += target == top_preds[0]
        correct_top2 += target in top_preds[:2]
        correct_top3 += target in top_preds[:3]

        if abs(entry["margin"]) < 100:
            total_margin += entry["margin"]

        total += 1

    scores = {
        "recall@1": correct_top1 / total if total else 0,
        "recall@2": correct_top2 / total if total else 0,
        "recall@3": correct_top3 / total if total else 0,
        "margin%": total_margin / total if total else 0,
    }

    # Overwrite the file with updated scores
    with open(f"results/real_{args.attack}_{Constants.MODE}.json", "w") as f:
        json.dump({"scores": scores, "results": results}, f, indent=4)


if __name__ == "__main__":
    main()
