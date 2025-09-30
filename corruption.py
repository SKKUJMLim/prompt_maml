import torch
import numpy as np
import cv2  # pip install opencv-python

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
def impulse_noise(x, prob=0.02):
    r = torch.rand_like(x)
    x = x.clone()
    x[r < prob/2] = 0.0
    x[r > 1 - prob/2] = 1.0
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
            "gaussian_noise": gaussian_noise,
            "shot_noise":     shot_noise,
            "impulse_noise":  impulse_noise,
            "motion_blur":    motion_blur,
            "jpeg":           jpeg_compression,
        }
        if self.name not in self._registry:
            raise ValueError(f"Unknown corruption name: {self.name}")

    def __call__(self, tensor):
        fn = self._registry[self.name]
        return fn(tensor, **self.kwargs)
