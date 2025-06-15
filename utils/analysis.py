import torch
import torch.nn.functional as F
import itertools
from collections import defaultdict

def compute_layerwise_cosine_similarity(task_grads_dict_list):
    """
    task_grads_dict_list: list of dicts, each dict is like target_names_grads_copy from a task
    """
    # 1. Layer별로 gradient 수집
    layerwise = defaultdict(list)  # { 'layer_dict.conv0': [grad_task1, grad_task2, ...] }

    for task_grad_dict in task_grads_dict_list:
        for full_name, grad in task_grad_dict.items():
            if grad is None:
                continue
            # 'layer_dict.conv0.conv.weight' → 'layer_dict.conv0'
            parts = full_name.split('.')[:2]
            layer_key = '.'.join(parts)
            layerwise[layer_key].append(grad.detach().flatten())

    # 2. 각 layer별 pairwise cosine similarity 계산
    results = {}
    for layer, grads in layerwise.items():
        if len(grads) < 2:
            continue
        pairwise_sims = []
        for i in range(len(grads)):
            for j in range(i + 1, len(grads)):
                sim = F.cosine_similarity(grads[i], grads[j], dim=0)
                pairwise_sims.append(sim.item())
        avg_sim = sum(pairwise_sims) / len(pairwise_sims)
        results[layer] = avg_sim
        print(f"[{layer}] Avg Pairwise Cosine Similarity: {avg_sim:.4f}")

    return results
