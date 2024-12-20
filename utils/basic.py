
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
    p = F.softmax(feature_map_1, dim=1)  # Along the channel dimension
    q = F.softmax(feature_map_2, dim=1)  # Along the channel dimension

    # Compute log probabilities
    log_q = torch.log(q + 1e-10)  # Add epsilon to avoid log(0)

    # Compute KL divergence
    kl_loss = F.kl_div(log_q, p, reduction=reduction)

    return kl_loss

def image_denormalization(image, datasets="mini_imagenet"):
    '''이미지를 역정규화하는 함수'''

    image = image.permute(1, 2, 0).detach().cpu().numpy()  # [C, H, W] -> [H, W, C]

    mean = np.array([0.0, 0.0, 0.0])
    std = np.array([0.0, 0.0, 0.0])

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

    denom_image = image * std + mean
    denom_image = np.clip(denom_image, 0, 1)

    return denom_image

def show_batch(images, labels, datasets='mini_imagenet'):

    '''배치 전체 이미지를 시각화하는 함수'''

    batch_size = images.shape[0]
    fig, axes = plt.subplots(1, batch_size, figsize=(batch_size * 3, 3))
    for i in range(batch_size):
        ax = axes[i]
        denom_image = image_denormalization(image=images[i], datasets=datasets)
        ax.imshow(denom_image)  # [C, H, W] -> [H, W, C]
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
    plt.show()