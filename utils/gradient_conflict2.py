import os
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity


def analyze_epoch_gradients(epoch, grad_dir):
    epoch_path = os.path.join(grad_dir, f"epoch{epoch}", "all_layers")
    if not os.path.exists(epoch_path):
        return None

    grad_files = sorted([
        f for f in os.listdir(epoch_path)
        if f.startswith(f"e{epoch}_i") and f.endswith(".pt")
    ])
    if not grad_files:
        return None

    grads = [torch.load(os.path.join(epoch_path, f)) for f in grad_files]
    grads = torch.stack(grads)  # [N, D]

    g_meta = grads.mean(dim=0)
    meta_norm = torch.norm(g_meta, p=2).item()

    cos_sims = [cosine_similarity(g_meta.unsqueeze(0), g.unsqueeze(0), dim=1).item() for g in grads]
    l2_dists = [torch.norm(g_meta - g, p=2).item() for g in grads]

    return meta_norm, sum(cos_sims) / len(cos_sims), sum(l2_dists) / len(l2_dists)


def analyze_model(name, grad_root, max_epoch):
    norms, cosines, l2s = [], [], []
    for epoch in range(max_epoch):
        print("Epoch : ", epoch)
        result = analyze_epoch_gradients(epoch, grad_root)
        if result is None:
            continue
        norm, cosine, l2 = result
        norms.append(norm)
        cosines.append(cosine)
        l2s.append(l2)
    return {
        'name': name,
        'norms': norms,
        'cosines': cosines,
        'l2s': l2s
    }


def save_plot(metric_name, maml_vals, dcml_vals, out_dir, label1='DCML', label2='MAML', log_scale=False):
    epochs = list(range(len(maml_vals)))

    # 유효한 데이터만 필터링
    valid_data = [(e, m, d) for e, m, d in zip(epochs, maml_vals, dcml_vals) if m is not None and d is not None]
    if not valid_data:
        print(f"[Error] No valid data to plot for {metric_name}.")
        return

    sorted_epochs, maml_filtered, dcml_filtered = zip(*sorted(valid_data))

    plt.figure(figsize=(8, 5))
    plt.plot(sorted_epochs, dcml_filtered, label=label2, linestyle='dashed', linewidth=2.5)
    plt.plot(sorted_epochs, maml_filtered, label=label1, linestyle='solid', linewidth=2.5)

    xticks = list(range(0, max(sorted_epochs) + 1, 10))
    if sorted_epochs[-1] not in xticks:
        xticks.append(sorted_epochs[-1])
    plt.xticks(xticks, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel(metric_name, fontsize=16)

    if log_scale:
        plt.yscale('log')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=15)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    filename = f"{metric_name.replace(' ', '_').lower()}.png"
    save_path = os.path.join(out_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {save_path}")

