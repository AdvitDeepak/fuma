"""

analysis_n_gram.py

Compute n-gram overlap between predictions and target questions across TOFU and RWKU datasets.

"""

from collections import Counter
import numpy as np
import json


def get_ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))


def ngram_precision(pred, target, n):
    pred_ngrams = Counter(get_ngrams(pred.split(), n))
    target_ngrams = Counter(get_ngrams(target.split(), n))
    overlap = sum((pred_ngrams & target_ngrams).values())
    total = sum(pred_ngrams.values())
    return overlap / total if total > 0 else 0.0


# Script Parameters 
n = 5
modes = ["easy", "hard"]
attacks = ["attack_grad-smart", "attack_loss-smart"]
datasets = ["tofu", "rwku"]


# Compute n-gram results
n_gram = {"tofu" : [], "rwku" : []}
for mode in modes:
    for attack in attacks:

        with open(f"{attack}_{n}_{mode}.json", "r") as f:
            saved = json.load(f)
        results = saved["results"]

        for ds in datasets:
            total_n_gram = 0
            total = 0

            for entry in results:
                if entry["dataset"] == ds:
                    predictions = [v[0] for v in entry["predictions"].values()]
                    target = entry["target-question"]

                    overlaps = []
                    max_n=10
                    for pred in predictions:
                        score = np.mean([
                            ngram_precision(pred, target, n)
                            for n in range(1, max_n + 1)
                        ])
                        overlaps.append(score)

                    total_n_gram += np.mean(overlaps)
                    total += 1

            n_gram[ds].append(total_n_gram / total)

for ds in n_gram: 
    print(f"{ds} : {np.mean(n_gram[ds])}")