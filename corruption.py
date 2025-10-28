import torch
import numpy as np
import cv2
import math
import torch.nn.functional as F


def corrupt_labels_batch_wise(y_support_set, corruption_rate, rng):
    """
    Corrupts a fraction of the support set labels.
    y_support_set shape: (Batch_size, N_way * K_shot)
    """
    y_corrupted = y_support_set.clone()
    batch_size, num_samples = y_corrupted.shape

    for i in range(batch_size):
        task_labels = y_corrupted[i].view(-1)
        num_corrupt = int(num_samples * corruption_rate)

        # Select indices to corrupt
        corrupt_indices = rng.choice(num_samples, num_corrupt, replace=False)

        # Corrupt labels: assign a random incorrect label
        num_classes = torch.max(task_labels) + 1

        for idx in corrupt_indices:
            original_label = task_labels[idx]
            # Generate a new label that is not the original one
            possible_labels = [l for l in range(num_classes) if l != original_label]
            if len(possible_labels) > 0:
                new_label = rng.choice(possible_labels)
                task_labels[idx] = new_label

        y_corrupted[i] = task_labels.view(y_support_set[i].shape)

    return y_corrupted



def corrupt_labels_task_wise(y_set, corruption_rate, rng):
    """
    Corrupts a fraction of the labels for a SINGLE task.
    y_set shape: (N_way * K_shot) - Assumed to be a 1D Tensor after view(-1)

    :param y_set: The 1D torch Tensor of labels for a single task.
    :param corruption_rate: The fraction of labels to corrupt.
    :param rng: The numpy RandomState object for controlled randomness.
    :return: The corrupted 1D torch Tensor of labels.
    """
    # y_corrupted는 1D 텐서입니다.
    y_corrupted = y_set.clone()
    task_labels = y_corrupted.view(-1)
    num_samples = task_labels.numel()

    # 오염시킬 라벨의 개수 계산
    num_corrupt = int(num_samples * corruption_rate)

    # 1. 오염시킬 인덱스 선택
    # rng.choice는 numpy 함수이므로 텐서가 아닌 정수 배열을 반환합니다.
    corrupt_indices = rng.choice(num_samples, num_corrupt, replace=False)

    # 2. 새로운 (틀린) 라벨 할당
    # 현재 Task의 총 클래스 수 계산
    # y_set이 0부터 N-1까지의 클래스 라벨을 포함한다고 가정합니다.
    if num_samples > 0:
        num_classes = torch.max(task_labels).item() + 1
    else:
        return y_corrupted  # 샘플이 없으면 그대로 반환

    for idx in corrupt_indices:
        original_label = task_labels[idx].item()

        # 원본 라벨이 아닌 가능한 모든 라벨 리스트 생성
        possible_labels = [l for l in range(num_classes) if l != original_label]

        if len(possible_labels) > 0:
            # 새로운 라벨을 무작위로 선택하여 할당
            new_label = rng.choice(possible_labels)
            task_labels[idx] = new_label
        # (len(possible_labels) == 0 인 경우는 N_way가 1일 때만 가능하며, 이 경우 오염이 불가능합니다.)

    # 1D 텐서이므로 view() 없이 바로 반환
    return y_corrupted

# Gaussian
def gaussian_noise(x, std=0.05):
    noise = torch.randn_like(x) * std
    return torch.clamp(x + noise, 0.0, 1.0)


def uniform_noise(x, width=None, std=None):
    """
    Uniform(-a, a) 노이즈를 입력에 더합니다.
    - x: [0,1] 범위를 가정 (Tensor, BxCxHxW 또는 CxHxW)
    - width: a 값 (절반 범위). 예: width=0.1 -> U(-0.1, 0.1)
    - std: 표준편차. 지정 시 a = sqrt(3) * std 로 변환하여 동일 분산의 Uniform을 생성
    """
    if std is not None:
        a = math.sqrt(3.0) * float(std)
    else:
        a = 0.1 if width is None else float(width)

    noise = (torch.rand_like(x) - 0.5) * 2.0 * a
    return torch.clamp(x + noise, 0.0, 1.0)


# Shot (Poisson)
def shot_noise(x, scale=1.0):
    # x:[0,1] 가정
    s = float(scale)
    s = max(s, 1e-8)                       # float로 안전 가드
    lam = (x * s).clamp_min(1e-8)          # 텐서 clamp는 이렇게
    noisy = torch.poisson(lam) / s         # E[noisy] = x로 복원
    return noisy.clamp(0.0, 1.0)

# Impulse (salt & pepper)
def impulse_noise(x, severity=0.02):
    r = torch.rand_like(x)
    x = x.clone()
    x[r < severity/2] = 0.0
    x[r > 1 - severity/2] = 1.0
    return x

# Motion blur (간단한 선형 커널)
def motion_blur(x, kernel_size=9):
    # kernel_size를 정수·홀수로 정규화
    k = int(round(kernel_size))          # float -> int
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    x_np = (x.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    k_mat = np.zeros((k, k), dtype=np.float32)
    k_mat[k // 2, :] = 1.0
    k_mat /= k_mat.sum()

    blurred = cv2.filter2D(x_np, -1, k_mat)
    return torch.tensor(blurred / 255., dtype=torch.float32).permute(2,0,1)


# def motion_blur(x: torch.Tensor, k: int = 9, angle_deg: float = 0.0) -> torch.Tensor:
#     """
#     x: BxCxHxW 또는 CxHxW 텐서, float [0,1]
#     k: 커널 사이즈(홀수 권장)
#     angle_deg: 블러 방향(도)
#     """
#     single = (x.dim() == 3)
#     if single:
#         x = x.unsqueeze(0)  # 1xCxHxW
#
#     B, C, H, W = x.shape
#     dev = x.device
#     # 기본 수평 커널 생성
#     kernel = torch.zeros((k, k), device=dev, dtype=x.dtype)
#     kernel[k//2, :] = 1.0
#     kernel = kernel / kernel.sum()
#
#     # 각도 회전(간단한 근사: grid_sample)
#     # 회전 행렬
#     theta = torch.tensor([
#         [ torch.cos(torch.deg2rad(torch.tensor(angle_deg))), -torch.sin(torch.deg2rad(torch.tensor(angle_deg))), 0.0],
#         [ torch.sin(torch.deg2rad(torch.tensor(angle_deg))),  torch.cos(torch.deg2rad(torch.tensor(angle_deg))), 0.0]
#     ], device=dev, dtype=x.dtype).unsqueeze(0)
#
#     grid = F.affine_grid(theta, size=(1,1,k,k), align_corners=False)
#     kernel = kernel.unsqueeze(0).unsqueeze(0)  # 1x1xk xk
#     kernel = F.grid_sample(kernel, grid, align_corners=False)
#     kernel = kernel / kernel.sum()
#
#     # 채널별 depthwise conv
#     weight = kernel.repeat(C, 1, 1, 1)  # Cx1xk xk
#     padding = k // 2
#     out = F.conv2d(x, weight, bias=None, stride=1, padding=padding, groups=C)
#     out = out.clamp(0,1)
#
#     return out.squeeze(0) if single else out


# JPEG compression
def jpeg_compression(x, quality=30):
    x_np = (x.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, enc = cv2.imencode('.jpg', x_np, enc_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return torch.tensor(dec/255., dtype=torch.float32).permute(2,0,1)


import torch
import numpy as np

# 기존의 SelectCorruption 클래스에 통합하기 위한 새로운 함수 정의

def random_block_masking(x: torch.Tensor, size: float = 0.2, fill_value: float = 0.0, rng: np.random.RandomState = None) -> torch.Tensor:
    """
    이미지의 무작위 영역(블록)을 지정된 값으로 마스킹합니다. (Partial Occlusion)
    (PyTorch 텐서, [0, 1] 범위 가정)

    :param x: CxHxW 형태의 입력 이미지 텐서. (배치 형태는 augment_image에서 처리해야 함)
    :param size: 이미지 높이/너비에 대한 마스크 크기의 비율 (예: 0.2 -> 20% 크기)
    :param fill_value: 마스크를 채울 값 (0.0은 검정색, 0.5는 중간 회색 등)
    :param rng: Numpy RandomState 객체 (재현성 확보용)
    :return: 마스킹이 적용된 텐서.
    """
    if rng is None:
        rng = np.random.RandomState()

    x_masked = x.clone()
    _, H, W = x.shape

    # 마스크 크기 계산 (정수로 반올림)
    mask_h = int(H * size)
    mask_w = int(W * size)

    if mask_h < 1 or mask_w < 1:
        # 마스크 크기가 너무 작으면 노이즈 적용 안함
        return x_masked

    # 마스크의 좌측 상단 위치 무작위 선택
    # H - mask_h + 1 범위를 사용 (마스크가 이미지 경계를 벗어나지 않도록)
    y1 = rng.randint(0, H - mask_h + 1)
    x1 = rng.randint(0, W - mask_w + 1)
    y2 = y1 + mask_h
    x2 = x1 + mask_w

    # 마스크 적용
    # [Channel, Height, Width]
    x_masked[:, y1:y2, x1:x2] = fill_value

    return x_masked


def speckle_noise(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """
    Speckle Noise (Multiplicative Noise)를 적용합니다.
    x:[0,1] 가정.
    """
    # 텐서와 동일한 크기의 정규분포 노이즈 맵 생성
    noise = torch.randn_like(x) * std
    # 노이즈를 픽셀 값에 곱한 후 더함: x_noisy = x + x * noise
    noisy_x = x + x * noise
    return noisy_x.clamp(0.0, 1.0)


class SelectCorruption(object):
    """
    지정한 corruption만 적용.
    name: 문자열 (예: 'gaussian_noise', 'impulse_noise', 'motion_blur', 'jpeg', 'shot_noise')
    kwargs: 해당 함수의 하이퍼파라미터 (예: std=0.05, prob=0.02, kernel_size=7, quality=30 등)
    """
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self._registry = {
            "gaussian_noise":   gaussian_noise,
            "uniform_noise":    uniform_noise,
            "shot_noise":       shot_noise,
            "impulse_noise":    impulse_noise,
            "motion_blur":      motion_blur,
            "jpeg_compression": jpeg_compression,
            "random_block_masking": random_block_masking,
            "speckle_noise": speckle_noise,
        }
        if self.name not in self._registry:
            raise ValueError(f"Unknown corruption name: {self.name}")

    def __call__(self, tensor):
        fn = self._registry[self.name]
        return fn(tensor, **self.kwargs)
