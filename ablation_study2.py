import argparse
import os
import torch

from utils.gradient_conflict2 import (
    analyze_model, save_plot
)

from utils.model_distance import (
    plot_model_distance, make_random_init_params, get_model_distance_from_fixed_random_init)

if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    maml_path = "MAML_5way_5shot_filter128_miniImagenet/grad_info_per_epoch"
    dcml_path = "MAML_Prompt_padding_5way_5shot_filter128_miniImagenet/grad_info_per_epoch"
    max_epoch = 100  # 필요 시 조정

    # 분석
    print("Gradient conflict Start!=================")
    print("MAML.....")
    maml = analyze_model("MAML", maml_path, max_epoch)
    print("DCML.....")
    dcml = analyze_model("DCML", dcml_path, max_epoch)

    # 시각화 및 저장
    save_plot("Meta-gradient Norm", maml["norms"], dcml["norms"], 'gradient2/norm_of_meta_gradient', log_scale=True)
    save_plot("Cosine Similarity", maml["cosines"], dcml["cosines"], 'gradient2/cos_sim')
    save_plot("L2 Distance", maml["l2s"], dcml["l2s"], 'gradient2/distance')

    print("Gradient conflict Analysis End!=================")

    print("Parameter shift Analysis Start!=================")
    prompt_ckpt_dir = 'MAML_5way_5shot_filter128_miniImagenet/saved_models'
    baseline_ckpt_dir = 'MAML_Prompt_padding_5way_5shot_filter128_miniImagenet/saved_models'

    epochs = list(range(1, 100))  # train_model_1 ~ train_model_99
    prompt_checkpoints = [f"train_model_{e}" for e in epochs]
    baseline_checkpoints = [f"train_model_{e}" for e in epochs]

    # 기준 파라미터 (prompt 기준)
    sample_ckpt = torch.load(os.path.join(prompt_ckpt_dir, 'train_model_1'), map_location='cpu')
    reference_params = sample_ckpt['network']
    random_init_params = make_random_init_params(reference_params)

    # 모델 거리 계산
    prompt_dist = get_model_distance_from_fixed_random_init(random_init_params, prompt_ckpt_dir, prompt_checkpoints)
    baseline_dist = get_model_distance_from_fixed_random_init(random_init_params, baseline_ckpt_dir,
                                                              baseline_checkpoints)
    # 그래프 저장
    plot_model_distance(
        epochs,
        prompt_dist,
        baseline_dist,
        label1='Ours',
        label2='MAML',
        save_path='model_distance.png'
    )

    print("Parameter shift Analysis End!=================")
