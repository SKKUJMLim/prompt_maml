import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def random_flip(x):
    return torch.flip(x, dims=[3]) if torch.rand(1) < 0.5 else torch.flip(x, dims=[2])

def random_flip_like_torchvision(x):
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[3])  # Horizontal Flip
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[2])  # Vertical Flip
    return x


def random_flip_batchwise(x):
    B = x.size(0)  # x: [B, C, H, W]
    for i in range(B):
        if torch.rand(1) < 0.5:
            x[i] = torch.flip(x[i], dims=[1])  # Vertical flip (H)
        if torch.rand(1) < 0.5:
            x[i] = torch.flip(x[i], dims=[2])  # Horizontal flip (W)
    return x



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