import torch
import numpy as np
import os
from collections import defaultdict
import seaborn as sns
from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# plt.rcParams['font.family'] = 'Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] = False


def _ensure_2d(feat: torch.Tensor) -> torch.Tensor:
    """피처가 [N, D]가 아니면 2D로 평탄화합니다."""
    if feat.dim() == 1:
        feat = feat.unsqueeze(0)
    elif feat.dim() > 2:
        feat = feat.view(feat.size(0), -1)
    return feat.contiguous()


def gaussian_kernel(x, y, sigma_list: List[float]) -> torch.Tensor:
    """다중 스케일 가우시안 커널 K(x, y) 계산 (x: [m,d], y: [n,d])."""
    x = _ensure_2d(x).float()
    y = _ensure_2d(y).float()

    xx = (x * x).sum(dim=1, keepdim=True)        # [m,1]
    yy = (y * y).sum(dim=1, keepdim=True)        # [n,1]
    xy = x @ y.t()                                # [m,n]

    D2 = xx - 2 * xy + yy.t()                     # [m,n]
    D2 = torch.clamp(D2, min=0.0)

    K = torch.zeros_like(D2)
    for sigma in sigma_list:
        gamma = 1.0 / (2.0 * (float(sigma) ** 2))
        K = K + torch.exp(-gamma * D2)
    return K


def mmd_rbf(x, y, sigma_list: List[float] = None) -> torch.Tensor:
    """Unbiased MMD(RBF)의 제곱근. 표본 부족 시 biased로 전환."""
    if sigma_list is None:
        sigma_list = [1.0, 2.0, 4.0, 8.0, 16.0]

    x = _ensure_2d(x)
    y = _ensure_2d(y)
    m, n = x.size(0), y.size(0)

    K_xx = gaussian_kernel(x, x, sigma_list)
    K_yy = gaussian_kernel(y, y, sigma_list)
    K_xy = gaussian_kernel(x, y, sigma_list)

    if m < 2 or n < 2:
        mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return torch.sqrt(torch.clamp(mmd2, min=0.0))

    sum_xx_off = K_xx.sum() - torch.trace(K_xx)
    sum_yy_off = K_yy.sum() - torch.trace(K_yy)
    term_xx = sum_xx_off / (m * (m - 1))
    term_yy = sum_yy_off / (n * (n - 1))
    term_xy = K_xy.mean()  # (1/mn) * sum Kxy

    mmd2 = term_xx + term_yy - 2.0 * term_xy
    mmd2 = torch.clamp(mmd2, min=0.0)
    return torch.sqrt(mmd2)


def calculate_epoch_mmd(
    base_dir: str,
    epoch_list: List[int],
    num_tasks_per_batch: int,
    sigma_list: List[float],
    aggregate: str = "first",  # "first" or "mean"
) -> Dict[int, List[float]]:
    """
    에포크별로 (메타 배치 내) 모든 태스크 쌍의 MMD 값 리스트를 수집.
    return: { epoch: [mmd_val, mmd_val, ...] }
    """
    epoch_mmd_results: Dict[int, List[float]] = {}

    for epoch in epoch_list:

        print("epoch == ",epoch)

        epoch_dir = os.path.join(base_dir, f"epoch{epoch}")
        if not os.path.exists(epoch_dir):
            continue

        feature_files = [f for f in os.listdir(epoch_dir) if f.endswith("_features.pt")]
        if not feature_files:
            continue

        # (iter_id, task_id) 기준으로 피처 모으기
        task_features_by_iter = defaultdict(list)
        for fname in feature_files:
            parts = fname.split('_')
            task_id_str = next((p for p in parts if p.startswith('t')), None)
            iter_id_str = next((p for p in parts if p.startswith('i')), None)
            if task_id_str is None or iter_id_str is None:
                continue
            try:
                task_id = int(task_id_str[1:])
                iter_id = int(iter_id_str[1:])
            except ValueError:
                continue

            fpath = os.path.join(epoch_dir, fname)
            try:
                feat = torch.load(fpath, map_location="cpu")
            except Exception:
                continue

            task_features_by_iter[(iter_id, task_id)].append(_ensure_2d(feat))

        # iteration 단위로 task->tensor 재구성
        iter_task_map: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
        for (iter_id, task_id), feat_list in task_features_by_iter.items():
            if not feat_list:
                continue
            if aggregate == "first":
                rep = feat_list[0]                 # [N,D]
            elif aggregate == "mean":
                # 여러 텐서의 평균을 대표로 사용
                rep = torch.stack([f.mean(dim=0) for f in feat_list], dim=0).mean(dim=0, keepdim=True)
            else:
                rep = feat_list[0]
            iter_task_map[iter_id][task_id] = _ensure_2d(rep)

        batch_mmds: List[float] = []

        for iter_id, tasks_data in iter_task_map.items():
            tasks_in_batch = sorted(tasks_data.keys())[:num_tasks_per_batch]
            if len(tasks_in_batch) < 2:
                continue

            for i in range(len(tasks_in_batch)):
                for j in range(i + 1, len(tasks_in_batch)):
                    ti, tj = tasks_in_batch[i], tasks_in_batch[j]
                    fi, fj = tasks_data[ti], tasks_data[tj]
                    if fi.size(0) < 2 or fj.size(0) < 2:
                        # unbiased 분모 보호 (필요시 biased로 바꾸고 싶으면 mmd_rbf 내부가 이미 처리)
                        pass
                    mmd_val = mmd_rbf(fi, fj, sigma_list=sigma_list).item()
                    batch_mmds.append(mmd_val)

        if batch_mmds:
            epoch_mmd_results[epoch] = batch_mmds

    return epoch_mmd_results


def plot_kde_comparison(
    mmd_maml_dists: Dict[int, List[float]],
    mmd_dcml_dists: Dict[int, List[float]],
    target_epochs: List[int],
    experiment_name: str,
    bw_adjust: float = 0.8
):
    """선택한 에폭에 대해 MAML vs DCML의 MMD 분포(KDE)를 비교."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # 공통 xlim 산정 (95% 분위수 기반)
    all_dists: List[float] = []
    for dist_list in mmd_maml_dists.values():
        all_dists.extend(dist_list)
    for dist_list in mmd_dcml_dists.values():
        all_dists.extend(dist_list)
    if all_dists:
        max_dist = float(np.percentile(all_dists, 95) * 1.2)
    else:
        max_dist = 0.5

    for idx, epoch in enumerate(target_epochs[:4]):  # 최대 4개 슬롯
        ax = axes[idx]
        maml_data = mmd_maml_dists.get(epoch, [])
        dcml_data = mmd_dcml_dists.get(epoch, [])

        if len(maml_data) < 2 and len(dcml_data) < 2:
            ax.set_title(f"Epoch {epoch} (No sufficient data)", fontsize=14)
            ax.set_xlim(0, max_dist)
            ax.grid(True, linestyle='--', alpha=0.6)
            continue

        if len(maml_data) >= 2:
            sns.kdeplot(maml_data, ax=ax, label='MAML (Baseline)', fill=True,
                        alpha=0.5, color='skyblue', linewidth=1.5, bw_adjust=bw_adjust)
        if len(dcml_data) >= 2:
            sns.kdeplot(dcml_data, ax=ax, label='DCML (Ours)', fill=True,
                        alpha=0.5, color='coral', linewidth=1.5, bw_adjust=bw_adjust)

        ax.set_title(f"Epoch {epoch}: Task Pair MMD Distribution (KDE)", fontsize=16)
        ax.set_xlabel('Feature Distance (MMD)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.set_xlim(0, max_dist)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12)

    plt.suptitle("DCML vs. MAML: Feature Alignment Dynamics", fontsize=20, y=1.02)
    plt.tight_layout()

    os.makedirs(experiment_name, exist_ok=True)
    plot_path = os.path.join(experiment_name, "mmd_kde_comparison_epochs.png")
    plt.savefig(plot_path, dpi=200)
    plt.show()
    print(f"[INFO] KDE Plot saved to: {plot_path}")


import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

def plot_epochwise_mean_mmd(
    mmd_maml_dists: Dict[int, List[float]],
    mmd_dcml_dists: Dict[int, List[float]],
    target_epochs: List[int],
    experiment_name: str,
    save_name: str = "mmd_epochwise_mean_line.png"
):
    """
    epoch별 평균(그리고 표준편차) MMD를 하나의 선 그래프로 시각화.
    - y축: 평균 MMD
    - x축: epoch
    - 각 epoch의 분산 정보를 error bar(표준편차)로 표시
    """
    # 정렬된 epoch 리스트 (target_epochs 기준으로 존재하는 epoch만 사용)
    epochs = [e for e in target_epochs if (e in mmd_maml_dists) or (e in mmd_dcml_dists)]
    if not epochs:
        print("[WARN] 사용할 수 있는 epoch 데이터가 없습니다.")
        return

    # 평균/표준편차 계산 함수
    def _mean_std(d: Dict[int, List[float]], eps: float = 0.0):
        means, stds = [], []
        for e in epochs:
            arr = np.array(d.get(e, []), dtype=np.float32)
            if arr.size == 0:
                means.append(np.nan)
                stds.append(0.0)
            else:
                means.append(float(np.mean(arr)))
                stds.append(float(np.std(arr) + eps))
        return np.array(means), np.array(stds)

    maml_mean, maml_std = _mean_std(mmd_maml_dists)
    dcml_mean, dcml_std = _mean_std(mmd_dcml_dists)

    # 플롯
    plt.figure(figsize=(9, 6))
    # MAML
    plt.errorbar(epochs, maml_mean, yerr=maml_std, fmt='-o', linewidth=2, capsize=3, label='MAML (mean ± std)')
    # DCML
    plt.errorbar(epochs, dcml_mean, yerr=dcml_std, fmt='-s', linewidth=2, capsize=3, label='DCML (mean ± std)')

    plt.title("Epoch-wise Mean MMD (Task-pair Feature Distance)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean MMD")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    os.makedirs(experiment_name, exist_ok=True)
    out_path = os.path.join(experiment_name, save_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.show()
    print(f"[INFO] Line plot saved to: {out_path}")



if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # --- 최종 실행 예시 ---
    # python ablation_study_mmd.py
    # 이 부분은 실제 데이터를 로드할 경로와 파라미터에 맞게 수정해야 합니다.
    EXPERIMENT_NAME = "ＫDE_Analysis"
    NUM_TASKS_PER_BATCH = 4 # 메타 학습 시 사용한 배치 크기
    TARGET_EPOCHS = list(range(0, 22))
    SIGMAS = [0.5, 1.0, 2.0, 4.0, 8.0] # MMD 계산을 위한 다중 스케일 가우시안 커널 밴드폭

    # MAML과 DCML의 피처 저장 경로 (예시 경로)
    MAML_EXP_PATH = "MMD_MAML_5way_5shot_filter128_miniImagenet/feature_maps_for_MMD"
    DCML_EXP_PATH = "DCML_padding_5way_5shot_filter128_miniImagenet/feature_maps_for_MMD"

    # MMD 분포 계산
    print("Compute MAML..")
    mmd_maml_distributions = calculate_epoch_mmd(MAML_EXP_PATH, TARGET_EPOCHS, NUM_TASKS_PER_BATCH, sigma_list=SIGMAS)
    print("Compute DCML..")
    mmd_dcml_distributions = calculate_epoch_mmd(DCML_EXP_PATH, TARGET_EPOCHS, NUM_TASKS_PER_BATCH, sigma_list=SIGMAS)

    plot_epochwise_mean_mmd(
        mmd_maml_distributions,
        mmd_dcml_distributions,
        TARGET_EPOCHS,
        EXPERIMENT_NAME,
        save_name="mmd_epochwise_mean_line.png"
    )

    # 시각화
    plot_kde_comparison(mmd_maml_distributions, mmd_dcml_distributions, TARGET_EPOCHS, EXPERIMENT_NAME)
