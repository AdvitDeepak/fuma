"""

main.py

Main entry point to unlearn a specific set of examples from the specified dataset.

"""

import random
import subprocess
import re
import json
import argparse


def run_forget_solo(target_idx, epochs, dataset, rank):
    formatted_target_idx = f"[{target_idx}]"  # Format as [0,1,2]
    data_path = "locuslab/TOFU" if dataset == "tofu" else "jinzhuoran/RWKU"

    command = (
        f"CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 forget_solo.py "
        f"--config-name=forget_solo.yaml target_idx={formatted_target_idx} num_epochs={int(epochs)} dataset={dataset} data_path={data_path} rank={rank}"
    )
    print(f"Executing command: {command}")

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    save_path = None
    log_output = []

    for line in iter(process.stdout.readline, ''):
        print(line, end="")  # Print line immediately
        log_output.append(line)  # Store output
        if line.startswith("SAVE_PATH: "):
            save_path = line.split("SAVE_PATH: ")[1].strip()

    process.stdout.close()
    process.wait()

    # Read and print any remaining stderr output
    stderr_output = process.stderr.read()
    if stderr_output:
        print("\n=== STDERR OUTPUT ===\n", stderr_output)
        log_output.append(stderr_output)

    return save_path, log_output 

    
def main():
    parser = argparse.ArgumentParser(description="Unlearn on a specific set of forget indices")
    parser.add_argument("--indices", type=str, default="-1",
                    help="Comma separated list of indices for forgetting")
    parser.add_argument("--epochs", type=str, default=90,
                    help="Number of epochs for unlearning")
    parser.add_argument("--no_test", type=bool, default=False,
                    help="Toggle to skip testing of unlearnt model")

    parser.add_argument("--test_only", type=bool, default=False,
                    help="Toggle to ONLY do testing of unlearnt model")
    parser.add_argument("--model_path", type=str, default="",
                    help="Helper when ONLY do testing of unlearnt model")

    parser.add_argument("--dataset", type=str, default="tofu",
                    help="Either 'tofu' or 'rwku'")
    
    parser.add_argument("--rank", type=int, default=8,
                help="Rank for LoRA")



    args = parser.parse_args()

    if args.indices == "-1": 
        target_idx = str(random.randint(0, 3959 if args.dataset == "tofu" else 2878))
    else: 
        target_idx = args.indices
    print(f"Chosen idx: {target_idx}")

    # Perform unlearning
    save_path, log_output = run_forget_solo(target_idx, args.epochs, args.dataset, args.rank)
    print(f"Saved unlearned model to: {save_path}")


if __name__ == "__main__":
    main()
