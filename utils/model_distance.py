import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import init

# 1. 랜덤 초기화 파라미터 생성
def make_random_init_params(reference_params):
    random_init = {}
    for key, value in reference_params.items():
        if (
            'layer_dict' in key and
            'prompt' not in key and
            'norm_layer' not in key and
            'inner_loop_optimizer' not in key
        ):
            new_param = torch.empty_like(value)

            if 'bias' in key:
                random_init[key] = torch.zeros_like(value)
            else:
                init.xavier_uniform_(new_param)
                random_init[key] = new_param
    return random_init


# 2. 모델 거리 계산 함수
def get_model_distance_from_fixed_random_init(random_init_params, checkpoint_dir, checkpoint_names, device='cpu'):
    distances = []

    for ckpt_name in checkpoint_names:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[Warning] Checkpoint not found: {ckpt_path}")
            distances.append(None)
            continue

        ckpt = torch.load(ckpt_path, map_location=device)
        current_params = ckpt['network']

        total_distance = 0.0
        for key in random_init_params.keys():
            p0 = random_init_params[key].float()
            pt = current_params[key].float()
            total_distance += torch.norm(p0 - pt).item() ** 2

        distances.append(np.sqrt(total_distance))

    return distances


# 3. 그래프 시각화 및 고화질 저장
def plot_model_distance(epochs, dist1, dist2, label1='Ours', label2='MAML', save_path='model_distance.png'):
    plt.figure(figsize=(8, 5))

    # 유효한 데이터만 추림 (None 제거)
    valid_data = [(e, d1, d2) for e, d1, d2 in zip(epochs, dist1, dist2) if d1 is not None and d2 is not None]
    if not valid_data:
        print("[Error] No valid data to plot.")
        return

    sorted_epochs, dist1_filtered, dist2_filtered = zip(*sorted(valid_data))

    plt.plot(sorted_epochs, dist2_filtered, label=label2, linestyle='dashed', linewidth=2.5)
    plt.plot(sorted_epochs, dist1_filtered, label=label1, linestyle='solid', linewidth=2.5)

    xticks = list(range(0, 100, 10))
    if sorted_epochs[-1] not in xticks:
        xticks.append(sorted_epochs[-1])

    plt.xticks(xticks, fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.xlabel('Epoch', fontsize=16)
    plt.legend(fontsize=15)
    plt.tight_layout()

    # 고화질 저장
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {save_path}")
