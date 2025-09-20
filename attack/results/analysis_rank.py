"""

analysis_rank.py

Plots difference in attack performance vs. LoRA rank of unlearning.

"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def extract_mean_std(metric_key):
    values = [v[metric_key] for _, v in checkpoints]
    mean = [np.mean(vs) * 100 for vs in values]
    std = [np.std(vs) * 100 for vs in values]
    return mean, std


# Results parameters
n = 5
modes = ["easy", "hard"]
attacks = ["attack_grad-smart", "attack_loss-smart", "expmt_grad-ratio-fast"]
datasets = ["tofu", "rwku"]
attack = attacks[1]
mode = modes[1]


# Load in results
with open(f"ranks_{attack}_{n}_{mode}.json", "r") as f:
    saved = json.load(f)
results = saved["results"]


# Parse results
res = {}
for entry in results:
    preds = entry["predictions"]
    target = entry["target-question"]

    top_preds = [preds[str(i)][0] for i in range(len(preds))]
    r1 = int(target == top_preds[0])
    r2 = int(target in top_preds[:2])
    r3 = int(target in top_preds[:3])

    rank = entry["LoRA-rank"]

    vals = {
        "recall@1": [r1],
        "recall@2": [r2],
        "recall@3": [r3],
        "margin": [entry["margin"]],
    }
                
    if rank not in res: 
        res[rank] = vals
    else: 
        for k in vals:
            res[rank][k].append(vals[k][0])


# Create plots
CHECKPOINT_SCALE = 1
NUM_TOTAL = 9.0

plt.figure(figsize=(10, 4))
checkpoints = sorted((int(k) * CHECKPOINT_SCALE, v) for k, v in res.items())
x = [cp for cp, _ in checkpoints]

attack_name = attack.split("_")[1]
if "ratio-fast" in attack_name:
    attack_name = attack_name.replace("ratio-fast", "lora-ratio")

if "smart" in attack_name:
    attack_name = attack_name.replace("smart", "ens-avg")


top1_mean, top1_std = extract_mean_std('recall@1')
top2_mean, top2_std = extract_mean_std('recall@2')
top3_mean, top3_std = extract_mean_std('recall@3')
margin_mean, margin_std = extract_mean_std('margin')


sns.set(style="whitegrid", font="serif", context="talk")
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
palette = sns.color_palette("Set2")

# Recall@1
ax.plot(x, top1_mean, marker='o', markersize=6, linewidth=2, label=f'Recall@1 - {attack_name}', color=palette[0])
ax.fill_between(x, np.array(top1_mean) - np.array(top1_std), np.array(top1_mean) + np.array(top1_std),
                color=palette[0], alpha=0.1)

# Recall@2
ax.plot(x, top2_mean, marker='s', markersize=6, linewidth=2, label=f'Recall@2 - {attack_name}', color=palette[1])
ax.fill_between(x, np.array(top2_mean) - np.array(top2_std), np.array(top2_mean) + np.array(top2_std),
                color=palette[1], alpha=0.1)

# Recall@3
ax.plot(x, top3_mean, marker='x', markersize=6, linewidth=2, label=f'Recall@3 - {attack_name}', color=palette[2])
ax.fill_between(x, np.array(top3_mean) - np.array(top3_std), np.array(top3_mean) + np.array(top3_std),
                color=palette[2], alpha=0.1)


ax.set_xlabel("LoRA Rank", fontsize=14)
ax.set_ylabel("Score (%)", fontsize=14)
ax.set_title(f"Attack Success vs. LoRA Rank", fontsize=15)

ax.set_xticks(x)
ax.tick_params(axis='both', which='major', labelsize=12)

ax.legend(frameon=False, fontsize=11, loc='best')
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.savefig(f"plots/expmt_ranks_{attack}_{mode}.png", bbox_inches='tight', dpi=300)
