import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import functional as TF

def random_flip(x):
    return torch.flip(x, dims=[3]) if torch.rand(1) < 0.5 else torch.flip(x, dims=[2])

def random_brightness(x, min_val=0.8, max_val=1.2):
    """밝기 스케일 조정"""
    scale = torch.empty(1).uniform_(min_val, max_val).to(x.device)
    x_aug = x * scale
    return torch.clamp(x_aug, 0, 1)

def add_gaussian_noise(x, std=0.05):
    """가우시안 노이즈 추가"""
    noise = torch.randn_like(x) * std
    x_aug = x + noise
    return torch.clamp(x_aug, 0, 1)

def channel_shuffle(x):
    """채널 순서 변경"""
    perm = torch.randperm(x.size(1))
    return x[:, perm, :, :]

def cutout(x, mask_size=16):
    """랜덤 영역 마스킹"""
    x_aug = x.clone()
    b, c, h, w = x.size()
    for i in range(b):
        y = torch.randint(0, h, (1,))
        x_ = torch.randint(0, w, (1,))
        y1 = torch.clamp(y - mask_size // 2, 0, h)
        y2 = torch.clamp(y + mask_size // 2, 0, h)
        x1 = torch.clamp(x_ - mask_size // 2, 0, w)
        x2 = torch.clamp(x_ + mask_size // 2, 0, w)
        x_aug[i, :, y1:y2, x1:x2] = 0
    return x_aug

def channel_dropout(x, drop_prob=0.33):
    """채널 드롭아웃 (확률 기반)"""
    x_aug = x.clone()
    b, c, h, w = x.size()
    for i in range(b):
        if torch.rand(1) < drop_prob:
            drop_channel = torch.randint(0, c, (1,))
            x_aug[i, drop_channel, :, :] = 0
    return x_aug

def uniform_noise(x, range_val=0.1):
    """유니폼 노이즈 추가"""
    noise = torch.empty_like(x).uniform_(-range_val, range_val)
    x_aug = x + noise
    return torch.clamp(x_aug, 0, 1)

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


def class_aware_mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # 다른 클래스가 될 때까지 반복 (단순하지만 느림)
    for i in range(batch_size):
        while y[i] == y[index[i]]:
            index[i] = torch.randint(0, batch_size, (1,), device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


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