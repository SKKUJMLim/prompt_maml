import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import itertools


def gaussian_dropout(x, p):
    std = (p / (1 - p)) ** 0.5  # 표준편차 계산
    noise = torch.randn_like(x) * std + 1  # 1 + N(0, std^2)
    return x * noise

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



def kl_divergence(feat1, feat2):
    """
    두 개의 [64, 5, 5] 크기의 feature map 간 KL Divergence 계산
    - feat1, feat2: [64, 5, 5] 형태의 텐서
    - 출력: 픽셀별 KL Divergence 값 (동일한 [64, 5, 5] 크기)
    """
    # # 확률 분포로 변환 (픽셀별 Softmax)
    # p = F.softmax(feat1, dim=0)  # 픽셀별 정규화
    # q = F.softmax(feat2, dim=0)

    # 확률 분포로 변환 (Channel별 Softmax)
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
    denom_image = np.clip(denom_image, 0, 1)

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