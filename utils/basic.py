import torch.nn as nn
import torch
import torch.nn.functional as F
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import os

def plot_query_before_after_2d_fixed_axes(
    query_before, query_after, y_query,
    save_dir="./tsne_images",
    task_index=0,
    title_prefix="Query Feature Map",
    marker_size=80
):
    os.makedirs(save_dir, exist_ok=True)
    unique_classes = np.unique(y_query)
    colors = cm.get_cmap('tab10', len(unique_classes))
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', '+', 'x']

    # Step 1: 공통 t-SNE 임베딩
    combined = np.concatenate([query_before, query_after], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(combined)
    tsne_before = tsne[:len(query_before)]
    tsne_after  = tsne[len(query_before):]

    # Step 2: x/y 축 범위 계산
    x_min, x_max = tsne[:, 0].min(), tsne[:, 0].max()
    y_min, y_max = tsne[:, 1].min(), tsne[:, 1].max()

    # Step 3: Query BEFORE plot
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(unique_classes):
        color = colors(i)
        marker = markers[i % len(markers)]
        idx = np.where(y_query == cls)[0]
        plt.scatter(tsne_before[idx, 0], tsne_before[idx, 1],
                    marker=marker, color=color, alpha=0.8,
                    s=marker_size, label=f'Class {cls}')
    plt.title(f"{title_prefix} - Before Adaptation")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"task{task_index:03d}_query_before_fixed_2d.png"))
    plt.close()

    # Step 4: Query AFTER plot
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(unique_classes):
        color = colors(i)
        marker = markers[i % len(markers)]
        idx = np.where(y_query == cls)[0]
        plt.scatter(tsne_after[idx, 0], tsne_after[idx, 1],
                    marker=marker, color=color, alpha=0.8,
                    s=marker_size, label=f'Class {cls}')
    plt.title(f"{title_prefix} - After Adaptation")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"task{task_index:03d}_query_after_fixed_2d.png"))
    plt.close()


def gap(x):
    """
    Global Average Pooling: [B, C, H, W] → [B, C]
    """
    return x.mean(dim=[2, 3])

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-Entropy Loss with Label Smoothing.
    """

    def __init__(self, smoothing=0.1, reduction='mean'):
        """
        :param smoothing: Label smoothing factor (0 <= smoothing <= 1)
        :param reduction: Reduction type ('none', 'sum', 'mean')
        """
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = F.nll_loss(logprobs, target, reduction='none')
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'mean':
            return loss.mean()
        return loss


def logit_based_kd_loss(student_logits, teacher_logits, temperature=3.0):

    # Soft labels 생성
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    # KL Divergence Loss (Soft Label Distillation)
    kd_loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)

    return kd_loss


def jensen_shannon(logits1, logits2, dim=1):

    kl = nn.KLDivLoss(reduction='batchmean')

    p1 = F.softmax(logits1, dim)
    p2 = F.softmax(logits2, dim)
    M = torch.clamp((p1 + p2) / 2., 1e-7, 1.)

    js = (kl(M.log(), p1) + kl(M.log(), p2)) / 2.
    #js = (kl(p1.log(), M) + kl(p2.log(), M)) / 2.

    return js


def kl_divergence(feat1, feat2):

    p = F.softmax(feat1, dim=1)  # channel-dimension에서 softmax 적용
    q = F.softmax(feat2, dim=1)

    log_p = torch.log(p)

    kl_div = F.kl_div(log_p, q, reduction='batchmean')

    return kl_div


def compute_kl_loss(feature_map_1, feature_map_2, reduction='batchmean'):
    """
    Compute KL divergence loss between two feature maps.

    Parameters:
        feature_map_1 (torch.Tensor): Feature map from model 1 (B, C, H, W).
        feature_map_2 (torch.Tensor): Feature map from model 2 (B, C, H, W).
        reduction (str): Specifies the reduction type: 'none', 'batchmean', 'sum', 'mean'.

    Returns:
        torch.Tensor: KL divergence loss value.
    """
    # Flatten spatial dimensions to treat each pixel as an independent distribution
    B, C, H, W = feature_map_1.size()
    feature_map_1 = feature_map_1.view(B, C, -1)  # Shape: (B, C, H*W)
    feature_map_2 = feature_map_2.view(B, C, -1)  # Shape: (B, C, H*W)

    # Convert feature maps to probability distributions
    p = F.softmax(feature_map_1, dim=-1)  # Along the last dimension (H*W)
    q = F.softmax(feature_map_2, dim=-1)  # Along the last dimension (H*W)

    # dim=1로 하면 Channel 차원이다.

    # Compute log probabilities
    log_p = torch.log(p + 1e-10)  # Add epsilon to avoid log(0)

    # Compute KL divergence
    kl_loss = F.kl_div(log_p, q, reduction=reduction, log_target=False)

    return kl_loss

def compute_all_kl_losses(feature_maps, reduction='batchmean'):
    """
    Compute KL divergence for all combinations of feature maps.

    Parameters:
        feature_maps (list of torch.Tensor): List of feature maps (B, C, H, W).
        reduction (str): Specifies the reduction type: 'none', 'batchmean', 'sum', 'mean'.

    Returns:
        dict: Dictionary with pairs of feature maps as keys and their KL loss as values.
    """

    kl_losses = {}

    # for i, j in itertools.permutations(range(len(feature_maps)), 2):  # All pair permutations
    for i, j in itertools.combinations(range(len(feature_maps)), 2):  # Unique combinations
        kl_losses[(i, j)] = compute_kl_loss(feature_maps[i], feature_maps[j], reduction=reduction)

    return kl_losses


def compute_js_divergence(feature_map_1, feature_map_2, reduction='batchmean'):

    """Jensen-Shannon divergence"""

    # Convert feature maps to probability distributions
    p = F.softmax(feature_map_1, dim=1)
    q = F.softmax(feature_map_2, dim=1)

    # Compute the average distribution
    m = 0.5 * (p + q)

    # Compute KL divergence for both directions
    kl_pm = F.kl_div(torch.log(m + 1e-10), p, reduction=reduction)
    kl_qm = F.kl_div(torch.log(m + 1e-10), q, reduction=reduction)

    # Compute JS divergence
    js_div = 0.5 * (kl_pm + kl_qm)
    return js_div


def compute_all_js_divergence(feature_maps, reduction='batchmean'):

    js_divergence = {}

    # for i, j in itertools.permutations(range(len(feature_maps)), 2):  # All pair permutations
    for i, j in itertools.combinations(range(len(feature_maps)), 2):  # Unique combinations
        js_divergence[(i, j)] = compute_js_divergence(feature_maps[i], feature_maps[j], reduction=reduction)

    return js_divergence


def compute_mse_loss(feature_map_1, feature_map_2, reduction='mean'):
    """
    Compute Mean Squared Error (MSE) loss between two feature maps.

    Parameters:
        feature_map_1 (torch.Tensor): Feature map from model 1 (B, C, H, W).
        feature_map_2 (torch.Tensor): Feature map from model 2 (B, C, H, W).
        reduction (str): Specifies the reduction type: 'none', 'mean', 'sum'.

    Returns:
        torch.Tensor: MSE loss value.
    """
    # Ensure the shapes are the same
    assert feature_map_1.size() == feature_map_2.size(), "Feature maps must have the same shape"

    # Compute MSE loss
    mse_loss = F.mse_loss(feature_map_1, feature_map_2, reduction=reduction)
    return mse_loss

# Compute MSE for unique combinations of feature maps
def compute_unique_mse_losses(feature_maps, reduction='mean'):
    """
    Compute MSE for unique combinations of feature maps.

    Parameters:
        feature_maps (list of torch.Tensor): List of feature maps (B, C, H, W).
        reduction (str): Specifies the reduction type: 'none', 'mean', 'sum'.

    Returns:
        dict: Dictionary with unique pairs of feature maps as keys and their MSE loss as values.
    """
    mse_losses = {}
    for i, j in itertools.combinations(range(len(feature_maps)), 2):  # Unique combinations
        mse_losses[(i, j)] = compute_mse_loss(feature_maps[i], feature_maps[j], reduction=reduction)

    return mse_losses

def image_denormalization(image, datasets="mini_imagenet"):
    '''이미지를 역정규화하는 함수'''

    image = image.permute(1, 2, 0).detach().cpu().numpy()  # [C, H, W] -> [H, W, C]

    # mean = np.array([0.0, 0.0, 0.0])
    # std = np.array([0.0, 0.0, 0.0])

    mean, std = None, None

    # Normalize의 반대로 [0, 1] 범위로 복원
    if datasets == "mini_imagenet":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    elif datasets == "tiered_imagenet":
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    elif datasets == "CIFAR_FS":
        mean = np.array([0.5071, 0.4847, 0.4408])
        std = np.array([0.2675, 0.2565, 0.2761])
    elif datasets == "CUB":
        mean = np.array([104 / 255.0, 117 / 255.0, 128 / 255.0])
        std = np.array([1 / 255.0, 1 / 255.0, 1 / 255.0])
    else:
        raise ValueError(f"Unknown dataset: {datasets}")

    denom_image = image * std + mean
    # denom_image = np.clip(denom_image, 0, 1)
    denom_image = np.clip(255.0 * denom_image, 0, 255).astype(np.uint8)
    # denom_image = cv2.cvtColor(denom_image, cv2.COLOR_BGR2RGB)

    return denom_image

def show_batch(images, labels, datasets='mini_imagenet'):
    '''배치 전체 이미지를 시각화하는 함수'''
    batch_size = images.shape[0]

    # 배치 크기에 따라 Subplot 설정
    if batch_size == 1:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        denom_image = image_denormalization(image=images[0], datasets=datasets)
        ax.imshow(denom_image)
        ax.set_title(f"Label: {labels[0].item()}")
        ax.axis('off')
    else:
        fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 3, 3))
        for i in range(batch_size):
            ax = axes[i]
            denom_image = image_denormalization(image=images[i], datasets=datasets)
            ax.imshow(denom_image)
            ax.set_title(f"Label: {labels[i].item()}")
            ax.axis('off')
    plt.show()