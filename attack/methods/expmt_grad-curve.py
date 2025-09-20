"""

expmt_grad-curve.py

Calculate gradient curvature (sharpness) across increasing subsequences.

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def hessian_power_iteration(loss, model, num_iters=10, grad_device="cuda:1"):
    # Select parameters from every third layer (simplified)
    params = []
    layer_counter = 0

    for i, (name, param) in enumerate(model.named_parameters()):
        if i % 4 == 0 and "lora" not in name and param.requires_grad:
            params.append(param)

    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)

    if any(g is None for g in grads):
        print("Warning: Some gradients are None.")
        for i, g in enumerate(grads):
            if g is None:
                print(f"Gradient for param {id(params[i])} is None.")
        return None

    grad_vector = torch.cat([g.contiguous().view(-1) for g in grads])
    grad_vector = grad_vector.to("cuda:0")

    v = torch.randn(grad_vector.size(0), device="cuda:0")
    v = v / v.norm()

    # Perform Hessian power iteration
    for _ in range(num_iters):
        hvp = torch.autograd.grad(grad_vector, params, grad_outputs=v, retain_graph=True)
        hvp_vector = torch.cat([h.contiguous().view(-1) for h in hvp])
        v = hvp_vector / (hvp_vector.norm() + 1e-6)

    lambda_max = torch.dot(v, hvp_vector).item()
    return lambda_max


def attack(hf_link, tokenizer, alt_candidates) -> dict:
    model_device = torch.device("cuda:0")
    grad_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else model_device

    hf_link = "locuslab/tofu_ft_llama2-7b"

    model = AutoModelForCausalLM.from_pretrained(
        hf_link,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16
    ).to(model_device)
    model.requires_grad_(True)

    model.train()

    def tokenize_input(phrase):
        return tokenizer(phrase, return_tensors="pt").to(model_device)

    sharpness_dict = {}

    for candidate in alt_candidates:
        full_input = tokenize_input(candidate)
        input_ids = full_input["input_ids"][0]

        subseq_curvatures = []

        # Loop over subsequences of increasing length
        for end in range(2, len(input_ids) + 1):  # start from length 2
            subseq_ids = input_ids[:end]

            inputs = {
                "input_ids": subseq_ids.unsqueeze(0),
                "attention_mask": torch.ones_like(subseq_ids).unsqueeze(0)
            }
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=subseq_ids.unsqueeze(0))
            loss = outputs.loss

            if loss is None or loss.isnan():
                raise ValueError(f"Invalid loss value for input {candidate[:end]}: {loss}")

            curvature = hessian_power_iteration(loss, model, num_iters=10, grad_device=grad_device)
            subseq_curvatures.append(curvature)

        # Average curvature across all subsequences
        avg_curvature = sum(subseq_curvatures) / len(subseq_curvatures)
        print(candidate, avg_curvature)
        sharpness_dict[candidate] = avg_curvature


    ranked_dict = {
        i: (k, v)
        for i, (k, v) in enumerate(sorted(sharpness_dict.items(), key=lambda item: item[1], reverse=True))
    }
    return ranked_dict
