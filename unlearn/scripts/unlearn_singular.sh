#!/bin/bash

# Arguments

dataset="rwku" # Or "tofu"
indices_list=(37 198 350 612 999) # Specific DS indices to unlearn

epochs=600
ranks_list=(8 12 16 24 32 48 64 128) # LoRA ranks

sleep_time=5 # Wait between runs (in seconds)


for idx in "${indices_list[@]}"
do
    for rank in "${ranks_list[@]}"
    do
        echo "Running with indices=$idx and rank=$rank for $epochs epochs..."
        python main.py --indices "$idx" --epochs "$epochs" --dataset "$dataset" --rank "$rank"

        echo "Finished run with indices=$idx and rank=$rank. Waiting $sleep_time seconds before next run..."
        sleep "$sleep_time"
    done
done

echo "All runs complete."
