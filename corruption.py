import torch
import numpy as np
import cv2


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

# Shot (Poisson)
def shot_noise(x, scale=1.0):
    # x:[0,1] 가정
    lam = torch.clamp(x * scale, 1e-8, None)
    noisy = torch.poisson(lam) / torch.clamp(scale, 1e-8)
    return torch.clamp(noisy, 0.0, 1.0)

# Impulse (salt & pepper)
def impulse_noise(x, severity=0.02):
    r = torch.rand_like(x)
    x = x.clone()
    x[r < severity/2] = 0.0
    x[r > 1 - severity/2] = 1.0
    return x

# Motion blur (간단한 선형 커널)
def motion_blur(x, kernel_size=9):
    x_np = (x.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    k = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    k[kernel_size//2, :] = 1.0
    k /= k.sum()
    blurred = cv2.filter2D(x_np, -1, k)
    return torch.tensor(blurred/255., dtype=torch.float32).permute(2,0,1)

# JPEG compression
def jpeg_compression(x, quality=30):
    x_np = (x.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    enc_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    _, enc = cv2.imencode('.jpg', x_np, enc_param)
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    return torch.tensor(dec/255., dtype=torch.float32).permute(2,0,1)


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
            "shot_noise":       shot_noise,
            "impulse_noise":    impulse_noise,
            "motion_blur":      motion_blur,
            "jpeg_compression": jpeg_compression,
        }
        if self.name not in self._registry:
            raise ValueError(f"Unknown corruption name: {self.name}")

    def __call__(self, tensor):
        fn = self._registry[self.name]
        return fn(tensor, **self.kwargs)
