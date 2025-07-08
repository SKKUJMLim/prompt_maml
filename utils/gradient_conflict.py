import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 공통 설정
LAYER_NAMES = [
    "layer_layer_dict_conv0_conv_weight",
    "layer_layer_dict_conv1_conv_weight",
    "layer_layer_dict_conv2_conv_weight",
    "layer_layer_dict_conv3_conv_weight",
    "layer_layer_dict_linear_weights",
    "all_layers"
]

LAYER_LABELS = {
    "layer_layer_dict_conv0_conv_weight": "Conv0",
    "layer_layer_dict_conv1_conv_weight": "Conv1",
    "layer_layer_dict_conv2_conv_weight": "Conv2",
    "layer_layer_dict_conv3_conv_weight": "Conv3",
    "layer_layer_dict_linear_weights": "Linear",
    "all_layers": "All Layers"
}

# 1. Cosine 유사도 --------------------------------------------------

def compute_avg_cosine_similarity(epoch: int, layer_name: str, base_path: str):
    """
    Computes and prints the average cosine similarity between each task's gradient and the mean gradient.
    """

    layer_dir = os.path.join(
        base_path,
        f"grad_info_per_epoch",
        f"epoch{epoch}",
        layer_name
    )

    grad_list = []
    filenames = sorted([
        f for f in os.listdir(layer_dir) if f.endswith(".pt")
    ])

    for fname in filenames:
        grad = torch.load(os.path.join(layer_dir, fname))  # [D]
        grad_list.append(grad)

    grads = torch.stack(grad_list)  # [T, D]
    grad_mean = grads.mean(dim=0)  # [D]

    cos_sims = [F.cosine_similarity(g, grad_mean, dim=0).item() for g in grads]
    avg_sim = sum(cos_sims) / len(cos_sims)

    print(f"{layer_name} (epoch {epoch}): Avg cosine similarity = {avg_sim:.4f}")

    return avg_sim

def compute_avg_cosine_similarity_over_epochs(layer_name: str, base_path: str, epoch_list: list):
    """
    Computes and prints average cosine similarity for a specific layer across multiple epochs.
    """
    results = {}
    for epoch in epoch_list:
        try:
            avg_sim = compute_avg_cosine_similarity(epoch, layer_name, base_path)
            results[epoch] = avg_sim
        except Exception as e:
            print(f"[Warning] Epoch {epoch} failed: {e}")
    return results



def get_avg_cos_sim_all_layers(base_path: str, epoch_list: list, layer_names: list):
    """
    모든 layer에 대해 cosine similarity를 계산하여 dict로 반환
    """
    all_results = {}
    for layer_name in layer_names:
        results = compute_avg_cosine_similarity_over_epochs(layer_name, base_path, epoch_list)
        all_results[layer_name] = results
    return all_results


def plot_cosine_similarity_layerwise_individual(
        maml_all_results: dict,
        our_all_results: dict,
        epoch_list: list,
        save_dir: str = "gradient/cosine_similarity"
):
    os.makedirs(save_dir, exist_ok=True)

    for layer_name in LAYER_NAMES:
        label = LAYER_LABELS.get(layer_name, layer_name)
        plt.figure(figsize=(8, 5))

        sorted_epochs = sorted(maml_all_results[layer_name].keys())
        maml_values = [maml_all_results[layer_name][ep] for ep in sorted_epochs]
        our_values = [our_all_results[layer_name][ep] for ep in sorted_epochs]

        plt.plot(sorted_epochs, maml_values, label="MAML", linestyle='dashed')
        plt.plot(sorted_epochs, our_values, label="Ours", linestyle='solid')

        # 눈금 수 줄이기 + 마지막 epoch 포함 보장
        step = max(len(epoch_list) // 10, 1)
        display_epochs = sorted_epochs[::step]

        # 마지막 에폭이 포함되어 있지 않다면 추가
        if sorted_epochs[-1] not in display_epochs:
            display_epochs.append(sorted_epochs[-1])

        # 눈금 설정 + 글씨 키우기
        plt.xticks(display_epochs, fontsize=15)
        plt.yticks(fontsize=15)

        # X축 범위 강제 지정 (예: 0 ~ 100)
        plt.xlim([sorted_epochs[0], sorted_epochs[-1]])

        # plt.title(f"Avg Cosine Similarity - {label}", fontsize=12)
        plt.xlabel("Epoch")
        plt.ylabel("Avg Cosine Similarity")
        plt.grid(True)
        plt.legend()
        # plt.xticks(sorted(epoch_list))
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{label}_cosine_similarity.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[Saved] {save_path}")


def plot_cosine_similarity_layerwise_subplots(
        maml_all_results: dict,
        our_all_results: dict,
        epoch_list: list,
        save_dir: str = "gradient/cosine_similarity"
):
    num_layers = len(LAYER_NAMES)
    nrows = (num_layers + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, layer_name in enumerate(LAYER_NAMES):
        ax = axes[idx]
        label = LAYER_LABELS.get(layer_name, layer_name)

        sorted_epochs = sorted(maml_all_results[layer_name].keys())
        maml_values = [maml_all_results[layer_name][ep] for ep in sorted_epochs]
        our_values = [our_all_results[layer_name][ep] for ep in sorted_epochs]

        ax.plot(sorted_epochs, maml_values, label="MAML", linestyle='dashed')
        ax.plot(sorted_epochs, our_values, label="OURS", linestyle='solid')

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg Cosine Similarity")
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.set_xticks(sorted(epoch_list))

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "subplot_cosine_similarity.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


# 2. GSNR --------------------------------------------------
def compute_gsnr(epoch: int, layer_name: str, base_path: str):
    """
    Computes the GSNR (Gradient Signal-to-Noise Ratio) for a given epoch and layer.
    """
    layer_dir = os.path.join(
        base_path,
        f"grad_info_per_epoch",
        f"epoch{epoch}",
        layer_name
    )

    grad_list = []
    filenames = sorted([
        f for f in os.listdir(layer_dir) if f.endswith(".pt")
    ])

    for fname in filenames:
        grad = torch.load(os.path.join(layer_dir, fname))  # [D]
        grad_list.append(grad)

    grads = torch.stack(grad_list)  # [T, D]

    grad_mean = grads.mean(dim=0)   # [D]

    signal = grad_mean.mean().item() ** 2
    noise = torch.var(grads, dim=0).mean().item() + 1e-8  # 안정성 보정

    gsnr = signal / noise

    # mean_per_param = grads.mean(dim=0)  # [D]
    # var_per_param = grads.var(dim=0) + 1e-8  # [D]
    #
    # gsnr_per_param = (mean_per_param ** 2) / var_per_param  # [D]
    # gsnr = gsnr_per_param.mean().item()


    print(f"{layer_name} (epoch {epoch}): GSNR = {gsnr:.4f}")
    return gsnr


def compute_gsnr_over_epochs(layer_name: str, base_path: str, epoch_list: list):
    """
    Computes GSNR over multiple epochs for a given layer.
    """
    results = {}
    for epoch in epoch_list:
        try:
            gsnr = compute_gsnr(epoch, layer_name, base_path)
            results[epoch] = gsnr
        except Exception as e:
            print(f"[Warning] {layer_name} - Epoch {epoch} failed: {e}")
    return results


def get_gsnr_all_layers(base_path: str, epoch_list: list, layer_names: list):
    """
    Computes GSNRs for all layers over multiple epochs.
    """
    all_results = {}
    for layer_name in layer_names:
        results = compute_gsnr_over_epochs(layer_name, base_path, epoch_list)
        all_results[layer_name] = results
    return all_results


def plot_gsnr_individual(
        maml_all_results: dict,
        our_all_results: dict,
        epoch_list: list,
        save_dir: str = "gradient/gsnr"
):
    os.makedirs(save_dir, exist_ok=True)

    for layer_name in LAYER_NAMES:
        label = LAYER_LABELS.get(layer_name, layer_name)
        plt.figure(figsize=(8, 5))

        sorted_epochs = sorted(maml_all_results[layer_name].keys())
        maml_values = [maml_all_results[layer_name][ep] for ep in sorted_epochs]
        our_values = [our_all_results[layer_name][ep] for ep in sorted_epochs]

        plt.plot(sorted_epochs, maml_values, label="MAML", linestyle='dashed')
        plt.plot(sorted_epochs, our_values, label="OURS", linestyle='solid')

        # 눈금 수 줄이기
        step = max(len(epoch_list) // 10, 1)
        display_epochs = sorted_epochs[::step]
        plt.xticks(display_epochs)

        plt.title(f"GSNR - {label}", fontsize=12)
        plt.xlabel("Epoch")
        plt.ylabel("GSNR")
        plt.grid(True)
        plt.legend()
        # plt.xticks(sorted(epoch_list))
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{label}_gsnr.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[Saved] {save_path}")

def plot_gsnr_subplots(
        maml_all_results: dict,
        our_all_results: dict,
        epoch_list: list,
        save_dir: str = "gradient/gsnr"
):
    num_layers = len(LAYER_NAMES)
    nrows = (num_layers + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, layer_name in enumerate(LAYER_NAMES):
        ax = axes[idx]
        label = LAYER_LABELS.get(layer_name, layer_name)

        sorted_epochs = sorted(maml_all_results[layer_name].keys())
        maml_values = [maml_all_results[layer_name][ep] for ep in sorted_epochs]
        our_values = [our_all_results[layer_name][ep] for ep in sorted_epochs]

        ax.plot(sorted_epochs, maml_values, label="MAML", linestyle='dashed')
        ax.plot(sorted_epochs, our_values, label="OURS", linestyle='solid')

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("GSNR")
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.set_xticks(sorted(epoch_list))

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "subplot_gsnr.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")

# 3. L2 Distance --------------------------------------------------
def compute_l2_distance(epoch: int, layer_name: str, base_path: str):
    """
    Computes the average L2 distance between each task's gradient and the mean gradient.
    This measures how far each task gradient is from the central direction.

    Returns:
        float: average L2 distance across tasks
    """
    layer_dir = os.path.join(base_path, "grad_info_per_epoch", f"epoch{epoch}", layer_name)

    grad_list = []
    filenames = sorted([f for f in os.listdir(layer_dir) if f.endswith(".pt")])

    for fname in filenames:
        grad = torch.load(os.path.join(layer_dir, fname))  # [D]
        grad_list.append(grad)

    grads = torch.stack(grad_list)  # [T, D]
    grad_mean = grads.mean(dim=0)   # [D]

    l2_dists = [(g - grad_mean).norm(p=2).item() for g in grads]  # list of scalars
    avg_l2 = sum(l2_dists) / len(l2_dists)

    print(f"{layer_name} (epoch {epoch}): Avg L2 distance = {avg_l2:.4f}")
    return avg_l2

def compute_l2_distance_over_epochs(layer_name: str, base_path: str, epoch_list: list):
    """
    Computes average L2 distance over multiple epochs for a given layer.
    """
    results = {}
    for epoch in epoch_list:
        try:
            l2 = compute_l2_distance(epoch, layer_name, base_path)
            results[epoch] = l2
        except Exception as e:
            print(f"[Warning] {layer_name} - Epoch {epoch} failed: {e}")
    return results

def get_l2_distance_all_layers(base_path: str, epoch_list: list, layer_names: list):
    """
    Computes L2 distance for all layers across epochs.
    """
    all_results = {}
    for layer_name in layer_names:
        results = compute_l2_distance_over_epochs(layer_name, base_path, epoch_list)
        all_results[layer_name] = results
    return all_results


def plot_l2_distance_individual(
        maml_all_results: dict,
        our_all_results: dict,
        epoch_list: list,
        save_dir: str = "gradient/l2_distance"
):
    os.makedirs(save_dir, exist_ok=True)

    for layer_name in LAYER_NAMES:
        label = LAYER_LABELS.get(layer_name, layer_name)
        plt.figure(figsize=(8, 5))

        sorted_epochs = sorted(maml_all_results[layer_name].keys())
        maml_values = [maml_all_results[layer_name][ep] for ep in sorted_epochs]
        our_values = [our_all_results[layer_name][ep] for ep in sorted_epochs]

        plt.plot(sorted_epochs, maml_values, label="MAML", linestyle='dashed')
        plt.plot(sorted_epochs, our_values, label="Ours", linestyle='solid')

        # 눈금 수 줄이기 + 마지막 epoch 포함 보장
        step = max(len(epoch_list) // 10, 1)
        display_epochs = sorted_epochs[::step]

        # 마지막 에폭이 포함되어 있지 않다면 추가
        if sorted_epochs[-1] not in display_epochs:
            display_epochs.append(sorted_epochs[-1])

        # 눈금 설정 + 글씨 키우기
        plt.xticks(display_epochs, fontsize=15)
        plt.yticks(fontsize=15)

        # plt.title(f"L2 Distance - {label}", fontsize=12)
        plt.xlabel("Epoch")
        plt.ylabel("Average L2 Distance")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{label}_l2_distance.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[Saved] {save_path}")


def plot_l2_distance_subplots(
        maml_all_results: dict,
        our_all_results: dict,
        epoch_list: list,
        save_dir: str = "gradient/l2_distance"
):
    num_layers = len(LAYER_NAMES)
    nrows = (num_layers + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, layer_name in enumerate(LAYER_NAMES):
        ax = axes[idx]
        label = LAYER_LABELS.get(layer_name, layer_name)

        sorted_epochs = sorted(maml_all_results[layer_name].keys())
        maml_values = [maml_all_results[layer_name][ep] for ep in sorted_epochs]
        our_values = [our_all_results[layer_name][ep] for ep in sorted_epochs]

        ax.plot(sorted_epochs, maml_values, label="MAML", linestyle='dashed')
        ax.plot(sorted_epochs, our_values, label="OURS", linestyle='solid')

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg L2 Distance")
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.set_xticks(sorted(epoch_list))

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "subplot_l2_distance.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")

# 4. pairwise cos --------------------------------------------------
def compute_pairwise_cosine_similarity(epoch: int, layer_name: str, base_path: str):
    """
    Computes the average pairwise cosine similarity between all task gradients.
    Returns:
        float: average pairwise cosine similarity for the layer at given epoch
    """
    layer_dir = os.path.join(base_path, "grad_info_per_epoch", f"epoch{epoch}", layer_name)

    grad_list = []
    filenames = sorted([f for f in os.listdir(layer_dir) if f.endswith(".pt")])

    for fname in filenames:
        grad = torch.load(os.path.join(layer_dir, fname))  # [D]
        grad_list.append(grad)

    grads = torch.stack(grad_list)  # [T, D]
    T = grads.size(0)

    total_sim = 0.0
    count = 0

    for i in range(T):
        for j in range(i + 1, T):
            sim = F.cosine_similarity(grads[i], grads[j], dim=0).item()
            total_sim += sim
            count += 1

    avg_sim = total_sim / count if count > 0 else 0.0

    print(f"{layer_name} (epoch {epoch}): Avg Pairwise Cosine Similarity = {avg_sim:.4f}")
    return avg_sim

def compute_pairwise_cosine_similarity_over_epochs(layer_name: str, base_path: str, epoch_list: list):
    results = {}
    for epoch in epoch_list:
        try:
            sim = compute_pairwise_cosine_similarity(epoch, layer_name, base_path)
            results[epoch] = sim
        except Exception as e:
            print(f"[Warning] {layer_name} - Epoch {epoch} failed: {e}")
    return results

def get_pairwise_cosine_all_layers(base_path: str, epoch_list: list, layer_names: list):
    all_results = {}
    for layer_name in layer_names:
        results = compute_pairwise_cosine_similarity_over_epochs(layer_name, base_path, epoch_list)
        all_results[layer_name] = results
    return all_results

def plot_pairwise_cosine_individual(
        maml_all_results: dict,
        our_all_results: dict,
        epoch_list: list,
        save_dir: str = "gradient/pairwise_cosine"
):
    os.makedirs(save_dir, exist_ok=True)

    for layer_name in LAYER_NAMES:
        label = LAYER_LABELS.get(layer_name, layer_name)
        plt.figure(figsize=(8, 5))

        sorted_epochs = sorted(maml_all_results[layer_name].keys())
        maml_values = [maml_all_results[layer_name][ep] for ep in sorted_epochs]
        our_values = [our_all_results[layer_name][ep] for ep in sorted_epochs]

        plt.plot(sorted_epochs, maml_values, label="MAML", linestyle='dashed')
        plt.plot(sorted_epochs, our_values, label="OURS", linestyle='solid')

        plt.title(f"Pairwise Cosine Similarity - {label}", fontsize=12)
        plt.xlabel("Epoch")
        plt.ylabel("Avg Pairwise Cosine Similarity")
        plt.grid(True)
        plt.legend()
        plt.xticks(sorted(epoch_list))
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{label}_pairwise_cosine.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[Saved] {save_path}")

def plot_pairwise_cosine_subplots(
        maml_all_results: dict,
        our_all_results: dict,
        epoch_list: list,
        save_dir: str = "gradient/pairwise_cosine"
):
    num_layers = len(LAYER_NAMES)
    nrows = (num_layers + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, layer_name in enumerate(LAYER_NAMES):
        ax = axes[idx]
        label = LAYER_LABELS.get(layer_name, layer_name)

        sorted_epochs = sorted(maml_all_results[layer_name].keys())
        maml_values = [maml_all_results[layer_name][ep] for ep in sorted_epochs]
        our_values = [our_all_results[layer_name][ep] for ep in sorted_epochs]

        ax.plot(sorted_epochs, maml_values, label="MAML", linestyle='dashed')
        ax.plot(sorted_epochs, our_values, label="OURS", linestyle='solid')

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg Pairwise Cosine Similarity")
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.set_xticks(sorted(epoch_list))

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "subplot_pairwise_cosine.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")


def compute_variance_of_mean_gradient(epoch: int, layer_name: str, base_path: str):
    """
    평균 그래디언트의 분산(Variance of the mean gradient)을 계산합니다.
    """
    layer_dir = os.path.join(base_path, "grad_info_per_epoch", f"epoch{epoch}", layer_name)
    grad_list = []

    filenames = sorted([f for f in os.listdir(layer_dir) if f.endswith(".pt")])
    for fname in filenames:
        grad = torch.load(os.path.join(layer_dir, fname))  # [D]
        grad_list.append(grad)

    grads = torch.stack(grad_list)  # [T, D]
    grad_mean = grads.mean(dim=0)   # [D]
    variance = torch.var(grad_mean).item()  # scalar

    print(f"{layer_name} (epoch {epoch}): Variance of Mean Gradient = {variance:.6f}")
    return variance


def compute_variance_of_mean_gradient_over_epochs(layer_name: str, base_path: str, epoch_list: list):
    results = {}
    for epoch in epoch_list:
        try:
            var = compute_variance_of_mean_gradient(epoch, layer_name, base_path)
            results[epoch] = var
        except Exception as e:
            print(f"[Warning] {layer_name} - Epoch {epoch} failed: {e}")
    return results


def get_variance_of_mean_gradient_all_layers(base_path: str, epoch_list: list, layer_names: list):
    all_results = {}
    for layer_name in layer_names:
        results = compute_variance_of_mean_gradient_over_epochs(layer_name, base_path, epoch_list)
        all_results[layer_name] = results
    return all_results



def plot_variance_of_mean_gradient_individual(
        maml_all_results: dict,
        our_all_results: dict,
        epoch_list: list,
        save_dir: str = "gradient/var_mean_grad"
):
    os.makedirs(save_dir, exist_ok=True)

    for layer_name in LAYER_NAMES:
        label = LAYER_LABELS.get(layer_name, layer_name)
        plt.figure(figsize=(8, 5))

        sorted_epochs = sorted(maml_all_results[layer_name].keys())
        maml_values = [maml_all_results[layer_name][ep] for ep in sorted_epochs]
        our_values = [our_all_results[layer_name][ep] for ep in sorted_epochs]

        plt.plot(sorted_epochs, maml_values, label="MAML", linestyle='dashed')
        plt.plot(sorted_epochs, our_values, label="OURS", linestyle='solid')

        step = max(len(epoch_list) // 10, 1)
        plt.xticks(sorted_epochs[::step])
        plt.title(f"Variance of Mean Gradient - {label}", fontsize=12)
        plt.xlabel("Epoch")
        plt.ylabel("Variance")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{label}_var_mean_gradient.png")
        plt.savefig(save_path)
        plt.close()
        print(f"[Saved] {save_path}")


def plot_variance_of_mean_gradient_subplots(
        maml_all_results: dict,
        our_all_results: dict,
        epoch_list: list,
        save_dir: str = "gradient/var_mean_grad"
):
    num_layers = len(LAYER_NAMES)
    nrows = (num_layers + 1) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3 * nrows))
    axes = axes.flatten()

    for idx, layer_name in enumerate(LAYER_NAMES):
        ax = axes[idx]
        label = LAYER_LABELS.get(layer_name, layer_name)

        sorted_epochs = sorted(maml_all_results[layer_name].keys())
        maml_values = [maml_all_results[layer_name][ep] for ep in sorted_epochs]
        our_values = [our_all_results[layer_name][ep] for ep in sorted_epochs]

        ax.plot(sorted_epochs, maml_values, label="MAML", linestyle='dashed')
        ax.plot(sorted_epochs, our_values, label="OURS", linestyle='solid')

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Var. of Mean Gradient")
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.set_xticks(sorted(epoch_list))

    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "subplot_var_mean_gradient.png")
    plt.savefig(save_path)
    plt.close()
    print(f"[Saved] {save_path}")




# 외부에서 import 가능하도록 설정
__all__ = [
    "LAYER_NAMES",
    "LAYER_LABELS",
    "compute_avg_cosine_similarity",
    "compute_avg_cosine_similarity_over_epochs",
    "get_avg_cos_sim_all_layers",
    "plot_cosine_similarity_layerwise_individual",
    "plot_cosine_similarity_layerwise_subplots"
]
