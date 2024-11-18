
import matplotlib.pyplot as plt
import numpy as np
import torch


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