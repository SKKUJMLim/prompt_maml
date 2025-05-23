import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import itertools
import random
from torchvision.transforms import functional as TF



class TensorAugMix:
    def __init__(self, mean, std, k=3, alpha=1.0, depth=-1, device='cpu'):
        """
        Args:
            mean, std: normalization values for un/normalize (list of 3 floats)
            k: number of augmentation chains (width)
            alpha: Dirichlet and Beta parameter
            depth: depth of each chain (set -1 for random [1,3])
        """
        self.mean = torch.tensor(mean).view(3, 1, 1).to(device)
        self.std = torch.tensor(std).view(3, 1, 1).to(device)
        self.k = k
        self.alpha = alpha
        self.depth = depth
        self.device = device
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def unnormalize(self, x):
        return x * self.std + self.mean

    def renormalize(self, x):
        return (x - self.mean) / self.std

    def _augment(self, x):
        """Apply a random sequence of augmentations to x (C, H, W)"""
        ops = [self._hflip, self._vflip, self._grayscale, self._rotate, self._blur]
        depth = self.depth if self.depth > 0 else random.randint(1, 3)
        for _ in range(depth):
            op = random.choice(ops)
            x = op(x)
        return x

    # Augmentations (pure tensor-based)
    def _hflip(self, x): return TF.hflip(x)
    def _vflip(self, x): return TF.vflip(x)
    def _grayscale(self, x): return TF.rgb_to_grayscale(x, num_output_channels=3)
    def _rotate(self, x): return TF.rotate(x, angle=random.uniform(-30, 30))
    def _blur(self, x): return TF.gaussian_blur(x, kernel_size=(3, 3), sigma=(0.1, 2.0))

    def __call__(self, x_norm):
        """
        Args:
            x_norm: (B, C, H, W) normalized input
        Returns:
            x_aug: AugMix-processed and normalized tensor (B, C, H, W)
        """
        B = x_norm.size(0)
        x_aug_list = []

        for i in range(B):
            x = x_norm[i]
            x = self.unnormalize(x)  # → [0,1] space

            ws = torch.tensor(torch.distributions.Dirichlet(torch.ones(self.k) * self.alpha).sample(),
                              device=x.device)
            m = torch.distributions.Beta(self.alpha, self.alpha).sample().to(x.device)

            mix = torch.zeros_like(x)
            for j in range(self.k):
                x_aug = x.clone()
                x_aug = self._augment(x_aug)
                mix += ws[j] * x_aug

            x_final = (1 - m) * x + m * mix
            x_aug_list.append(self.renormalize(x_final))

        return torch.stack(x_aug_list)

    def jensen_shannon(self, logits1, logits2, logits3):
        p1 = F.softmax(logits1, dim=1)
        p2 = F.softmax(logits2, dim=1)
        p3 = F.softmax(logits3, dim=1)
        M = torch.clamp((p1 + p2 + p3) / 3., 1e-7, 1.)
        js = (self.kl(M.log(), p1) + self.kl(M.log(), p2) + self.kl(M.log(), p3)) / 3.
        return js


def rand_bbox(size, lam):
    """
    랜덤 박스 좌표를 생성 (size: (B, C, H, W), lam: lambda)
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)  # 비율로 자름
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # 중앙 위치 무작위 선정
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix를 적용한 이미지, 라벨쌍, lambda 반환
    x: (B, C, H, W), y: (B,)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    y_a = y
    y_b = y[index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x_cutmix = x.clone()
    x_cutmix[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # 실제 lambda 보정 (잘린 영역 비율)
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))

    return x_cutmix, y_a, y_b, lam

def mixup_data(x, y, alpha=0.4):
    """
    이미지를 MixUp하고, 섞인 label 쌍과 lambda 반환
    x: (B, C, H, W), y: (B,)
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def random_flip(x):
    return torch.flip(x, dims=[3]) if torch.rand(1) < 0.5 else torch.flip(x, dims=[2])

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