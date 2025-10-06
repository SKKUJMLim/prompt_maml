import torch.nn as nn
import torch
import torch.nn.functional as F
import itertools
from sklearn.manifold import TSNE
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def load_pretrained_weights(self, path_to_weights):
    """
    사전 학습된 표준 ResNet12 가중치를 MAML Meta-ResNet12 모델에 로드합니다.
    """
    print(f"Loading pretrained weights from {path_to_weights}...")

    # 1. 사전 학습된 가중치 로드
    pretrained_state_dict = torch.load(path_to_weights, map_location=self.device)

    # 모델이 DataParallel로 래핑되어 있다면 self.classifier.module을 사용
    model_to_load = self.classifier.module if torch.cuda.device_count() > 1 else self.classifier
    maml_state_dict = model_to_load.state_dict()

    new_maml_state_dict = {}

    # 2. 파라미터 이름 매핑
    for pretrained_name, pretrained_param in pretrained_state_dict.items():
        # 'layer1' -> 'layer0'로 인덱스 변경
        layer_idx = pretrained_name.split('.')[0].replace('layer', '')

        try:
            # ResNet12 layer 인덱스는 1부터 시작 (layer1, layer2, ...)
            # MAML layer 인덱스는 0부터 시작 (layer0, layer1, ...)
            idx = int(layer_idx) - 1
            if idx < 0:
                continue  # Skip if indexing is wrong

            base_name = f"layer_dict.layer{idx}"

            # --- Conv, BN 매핑 ---
            if "conv" in pretrained_name and "weight" in pretrained_name:
                # 예: layer1.block.conv1.weight -> layer_dict.layer0.conv1.conv.weight
                if "block.conv" in pretrained_name:
                    block_part = pretrained_name.split('block.')[1]  # conv1.weight
                    conv_name = block_part.split('.weight')[0]  # conv1
                    new_name = f"{base_name}.{conv_name}.conv.weight"
                    new_maml_state_dict[new_name] = pretrained_param

            # --- BN Running Stats 및 Parameters 매핑 (weight, bias, mean, var) ---
            if "bn" in pretrained_name:
                # convM에 연결된 BN
                if "block.bn" in pretrained_name:
                    block_part = pretrained_name.split('block.')[1]  # bn1.weight
                    bn_name = block_part.split('.')[0]  # bn1
                    param_type = block_part.split('.')[1]  # weight, bias, running_mean, ...

                    # MetaConvNormLayerSwish 내부의 MetaBatchNormLayer
                    new_bn_name = f"{base_name}.{bn_name}.norm_layer.{param_type}"

                    # MetaBatchNormLayer는 running_mean/var이 2D(num_steps, features)일 수 있으므로
                    # 차원을 확장하여 줘야 합니다 (MAML에서 per-step BN을 사용하는 경우).
                    # 여기서는 일단 1D로 로드하고, MAML이 2D로 저장했다면 오류가 날 수 있습니다.
                    # MAML이 1D (Global) BN을 사용한다고 가정하고 일단 로드합니다.
                    if new_bn_name in maml_state_dict:
                        new_maml_state_dict[new_bn_name] = pretrained_param

                # --- Downsample Shortcut BN 매핑 ---
                elif "downsample" in pretrained_name:
                    # 예: layer1.block.downsample.1.running_mean -> layer_dict.layer0.shortcut_norm_layer.running_mean
                    param_type = pretrained_name.split('downsample.1.')[1]
                    new_name = f"{base_name}.shortcut_norm_layer.{param_type}"
                    new_maml_state_dict[new_name] = pretrained_param

            # --- Downsample Shortcut Conv 매핑 ---
            elif "downsample" in pretrained_name and ".0.weight" in pretrained_name:
                # 예: layer1.block.downsample.0.weight -> layer_dict.layer0.shortcut_conv.weight
                new_name = f"{base_name}.shortcut_conv.weight"
                new_maml_state_dict[new_name] = pretrained_param

        except:
            # Logit layer나 이름이 맞지 않는 기타 파라미터는 건너뜁니다.
            print(f"Skipping parameter: {pretrained_name}")
            continue

    # 3. MAML 모델에 가중치 업데이트 및 로드
    maml_state_dict.update(new_maml_state_dict)

    # strict=False: 최종 Logit Layer 등 로드하지 않은 파라미터는 건너뜁니다.
    model_to_load.load_state_dict(maml_state_dict, strict=False)

    print("Pretrained weights successfully loaded to Meta-ResNet12 (Feature Extractor).")


# 이 함수를 MAMLFewShotClassifier 클래스 내부에 정의하고 __init__에서 호출하세요.


def count_params_by_key(param_dict, keyword):
    return sum(p.numel() for k, p in param_dict.items() if keyword in k and p.requires_grad)

def plot_3d_pca_query_comparison(
    query_before,
    query_after,
    y_query,
    acc_before,
    acc_after,
    save_dir,
    task_index,
    title_prefix="Effect of Prompt on Query Features",
    marker_size=100
):
    os.makedirs(save_dir, exist_ok=True)

    # PCA on combined query features
    all_query = np.concatenate([query_before, query_after], axis=0)
    pca = PCA(n_components=3)
    all_query_pca = pca.fit_transform(all_query)

    N = query_before.shape[0]
    query_before_pca = all_query_pca[:N]
    query_after_pca = all_query_pca[N:]

    fig = plt.figure(figsize=(14, 6))

    # BEFORE
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    for label in np.unique(y_query):
        idx = (y_query == label)
        ax1.scatter(query_before_pca[idx, 0], query_before_pca[idx, 1], query_before_pca[idx, 2],
                    label=f"Class {label}", s=marker_size)
    ax1.set_title(f"{title_prefix} (Before)\nAcc: {acc_before*100:.2f}%", fontsize=12)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.set_zlabel("PC3")
    ax1.legend()

    # AFTER
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    for label in np.unique(y_query):
        idx = (y_query == label)
        ax2.scatter(query_after_pca[idx, 0], query_after_pca[idx, 1], query_after_pca[idx, 2],
                    label=f"Class {label}", s=marker_size)
    ax2.set_title(f"{title_prefix} (After)\nAcc: {acc_after*100:.2f}%", fontsize=12)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_zlabel("PC3")
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"task_{task_index}_3d_pca.png")
    plt.savefig(save_path)
    plt.close()




def compute_intra_inter_class_variance(features, labels):
    unique_labels = np.unique(labels)
    overall_mean = features.mean(axis=0)

    intra_var = 0
    inter_var = 0

    for cls in unique_labels:
        cls_features = features[labels == cls]
        cls_mean = cls_features.mean(axis=0)
        intra_var += ((cls_features - cls_mean) ** 2).sum()
        inter_var += len(cls_features) * ((cls_mean - overall_mean) ** 2).sum()

    return intra_var / len(features), inter_var / len(features)


def plot_query_before_after_separate_with_accuracy(
    query_before, query_after, y_query,
    acc_before, acc_after,
    save_dir="./tsne_images",
    task_index=0,
    title_prefix="Query Feature Map",
    marker_size=100,
    perplexity=20,
    random_state=0
):
    os.makedirs(save_dir, exist_ok=True)

    assert query_before.shape[0] == len(y_query)
    assert query_after.shape[0] == len(y_query)

    unique_classes = np.unique(y_query)
    colors = cm.get_cmap('tab10', len(unique_classes))
    markers = ['o', 's', 'D', '^', 'v', 'p', '*', 'X', '+', 'x']

    # t-SNE 임베딩
    tsne_before = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init='pca') \
                    .fit_transform(query_before)
    tsne_after = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init='pca') \
                    .fit_transform(query_after)

    # Variance 계산
    intra_before, inter_before = compute_intra_inter_class_variance(tsne_before, y_query)
    intra_after, inter_after = compute_intra_inter_class_variance(tsne_after, y_query)
    ratio_before = inter_before / (intra_before + 1e-8)
    ratio_after = inter_after / (intra_after + 1e-8)

    # 공통 axis 범위 계산
    combined = np.vstack([tsne_before, tsne_after])
    x_min, x_max = combined[:, 0].min(), combined[:, 0].max()
    y_min, y_max = combined[:, 1].min(), combined[:, 1].max()

    # BEFORE
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(unique_classes):
        idx = np.where(y_query == cls)[0]
        plt.scatter(tsne_before[idx, 0], tsne_before[idx, 1],
                    c=[colors(i)], marker=markers[i % len(markers)],
                    label=f'Class {cls}', s=marker_size, alpha=0.8)
    plt.title(f"{title_prefix} (No Prompt)\nAccuracy: {acc_before:.2%} | Inter/Intra: {ratio_before:.2f}")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"task{task_index:03d}_query_without_prompt.png"))
    plt.close()

    # AFTER
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(unique_classes):
        idx = np.where(y_query == cls)[0]
        plt.scatter(tsne_after[idx, 0], tsne_after[idx, 1],
                    c=[colors(i)], marker=markers[i % len(markers)],
                    label=f'Class {cls}', s=marker_size, alpha=0.8)
    plt.title(f"{title_prefix} (With Prompt)\nAccuracy: {acc_after:.2%} | Inter/Intra: {ratio_after:.2f}")
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"task{task_index:03d}_query_with_prompt.png"))
    plt.close()


def plot_support_query_before_after_fixed_axes(
    support_before, query_before,
    support_after, query_after,
    y_support, y_query,
    save_dir, task_index, title_prefix="Feature Distribution", perplexity=30
):
    os.makedirs(save_dir, exist_ok=True)

    # 모든 feature 합치기
    all_features = np.concatenate([
        support_before, query_before,
        support_after, query_after
    ], axis=0)

    # t-SNE 임베딩
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(all_features)

    # 분리
    N1 = support_before.shape[0]
    N2 = query_before.shape[0]
    N3 = support_after.shape[0]
    N4 = query_after.shape[0]

    sb = tsne_result[:N1]
    qb = tsne_result[N1:N1+N2]
    sa = tsne_result[N1+N2:N1+N2+N3]
    qa = tsne_result[N1+N2+N3:]

    # 축 고정
    x_min, x_max = tsne_result[:, 0].min() - 5, tsne_result[:, 0].max() + 5
    y_min, y_max = tsne_result[:, 1].min() - 5, tsne_result[:, 1].max() + 5

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 색상 고정 (최대 10 class)
    import matplotlib.cm as cm
    num_classes = len(np.unique(np.concatenate([y_support, y_query])))
    cmap = cm.get_cmap("tab10", num_classes)
    class_colors = {cls: cmap(cls) for cls in range(num_classes)}

    # Plot Before
    for cls in np.unique(y_support):
        axes[0].scatter(sb[y_support == cls, 0], sb[y_support == cls, 1],
                        marker='o', s=60, label=f'Class {cls} (S)', color=class_colors[cls], alpha=0.7, edgecolor='k')
        axes[0].scatter(qb[y_query == cls, 0], qb[y_query == cls, 1],
                        marker='x', s=60, label=f'Class {cls} (Q)', color=class_colors[cls], alpha=0.7)

    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)
    axes[0].set_title(f"{title_prefix} (Before Adaptation)")
    axes[0].legend(fontsize=8)

    # Plot After
    for cls in np.unique(y_support):
        axes[1].scatter(sa[y_support == cls, 0], sa[y_support == cls, 1],
                        marker='o', s=60, label=f'Class {cls} (S)', color=class_colors[cls], alpha=0.7, edgecolor='k')
        axes[1].scatter(qa[y_query == cls, 0], qa[y_query == cls, 1],
                        marker='x', s=60, label=f'Class {cls} (Q)', color=class_colors[cls], alpha=0.7)

    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)
    axes[1].set_title(f"{title_prefix} (After Adaptation)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"task{task_index:03d}_tsne_fixed_axes.png")
    plt.savefig(save_path, dpi=300)
    plt.close()




def gap(x):
    """
    Global Average Pooling: [B, C, H, W] → [B, C]
    """
    return x.mean(dim=[2, 3])

def flatten_feature_map(fm: torch.Tensor) -> torch.Tensor:
    return fm.view(fm.size(0), -1)  # (B, C, H, W) → (B, C×H×W)

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