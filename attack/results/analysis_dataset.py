"""

analysis_dataset.py

Prints difference in performance on various datasets after unlearning.

"""

import json

n = 5
modes = ["easy", "hard"]
attacks = ["expmt_grad-ratio-fast"]
datasets = ["tofu", "rwku"]

for mode in modes:
    for attack in attacks:

        # Load all saved results
        with open(f"{attack}_{n}_{mode}.json", "r") as f:
            saved = json.load(f)

        results = saved["results"]

        for ds in datasets:
            correct_top1 = 0
            correct_top2 = 0
            correct_top3 = 0
            total_margin = 0
            total = 0

            # Compute aggregate scores
            for entry in results:
                if entry["dataset"] == ds:
                    preds = entry["predictions"]
                    target = entry["target-question"]

                    top_preds = [preds[str(i)][0] for i in range(len(preds))]
                    correct_top1 += target == top_preds[0]
                    correct_top2 += target in top_preds[:2]
                    correct_top3 += target in top_preds[:3]

                    total_margin += entry["margin"]
                    total += 1

            scores = {
                "recall@1": correct_top1 / total if total else 0,
                "recall@2": correct_top2 / total if total else 0,
                "recall@3": correct_top3 / total if total else 0,
                "margin%": total_margin / total if total else 0,
            }

            print(f"{ds} | {mode} | {attack}")
            print(scores)
            print(f"\n")
