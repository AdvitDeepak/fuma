"""

hf_util.py

Utility scripts to create dataset of unlearned models and upload to HuggingFace.

"""

import os
import json
from huggingface_hub import HfApi
import pandas as pd
import random
from datasets import load_dataset


def get_examples(target_idx, N, chunk_size=20):
    # Identify the chunk that the target_idx belongs to
    chunk_start = (target_idx // chunk_size) * chunk_size
    chunk_end = min(chunk_start + chunk_size - 1, MAX_IDX)

    chunk_indices = list(range(chunk_start, chunk_end + 1))
    chunk_indices.remove(target_idx)

    random_indices = random.sample(chunk_indices, N - 1)
    selected_indices = random_indices + [target_idx]

    examples = [dataset[i] for i in selected_indices]
    return selected_indices, examples

def push_models(hf_org, TOKEN):
    data_dir = "models/unlearned-adapters"

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            print(f"Parsing: {folder}")
            
            # Upload to Hugging Face
            repo_id = f"{hf_org}/{folder}"
            hf_link = f"https://huggingface.co/{repo_id}"

            try:
                hf_api.repo_info(repo_id, token=TOKEN)  # Try fetching repo info
                print(f"Repository {repo_id} already exists. Skipping upload.")
            except:
                print(f"Creating and uploading {repo_id}...")
                hf_api.create_repo(repo_id, exist_ok=True, token=TOKEN)
                hf_api.upload_folder(folder_path=folder_path, repo_id=repo_id, token=TOKEN)



def main():

    hf_org = "advit" 
    TOKEN = "<ADD-YOUR-HF-TOKEN-HERE"
    hf_api = HfApi()
    push_models = False  # Set to True if you want to push local models to HF

    dataset_tofu = load_dataset("locuslab/TOFU", "retain99", split="train")
    dataset_rwku = load_dataset("jinzhuoran/RWKU", "forget_level2")["test"]

    MAX_IDX_TOFU = 3599
    MAX_IDX_RWKU = 2878

    if push_models:
        push_models(hf_org, TOKEN)

    # Read all models that are in HuggingFace under org
    records = []
    models = hf_api.list_models(author=hf_org, token=TOKEN)

    for model in models:
        model_id = model.modelId  # e.g., 'advit/my_model'
        base_name = model_id.split("/")[-1]

        if base_name == "content": 
            print(f"Skipping (advit/content)") 
            continue 

        print(f"{base_name}")
        parts = base_name.split("_")

        ds = parts[0]
        loss_method = parts[1]
        lr = parts[2]
        r = parts[3]
        epochs = parts[4]
        indices_list = parts[5].split("-")
        if "inclusive" in indices_list: 
            idx_start = int(indices_list[0])
            idx_end = int(indices_list[1])
            indices_list = [x for x in range(idx_start, idx_end+1)]
    
        else:
            indices_list = [int(x) for x in indices_list]

        hf_link = f"https://huggingface.co/{model_id}"
        records.append([ds, loss_method, lr, r, epochs, indices_list, hf_link])


    # Then, create a dataset CSV
    dataset_df = pd.DataFrame(records, columns=["dataset", "unlearning-method", "learning-rate", "LoRA-rank", "epochs", "indices", "hf-link"])
    dataset_df.to_csv("fuma.csv", index=False)

    # Finally, upload to HuggingFace
    dataset_repo = f"{hf_org}/fuma"
    hf_api.create_repo(dataset_repo, repo_type="dataset", exist_ok=True, token=TOKEN)
    hf_api.upload_file(path_or_fileobj="fuma.csv", repo_type="dataset", path_in_repo="fuma.csv", repo_id=dataset_repo, token=TOKEN)


if __name__ == "__main__":
    main()
