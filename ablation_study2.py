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
    print("MAML.....")
    maml = analyze_model("MAML", maml_path, max_epoch)
    print("DCML.....")
    dcml = analyze_model("DCML", dcml_path, max_epoch)

    # 시각화 및 저장
    save_plot("Meta-gradient Norm", maml["norms"], dcml["norms"], 'gradient2/norm_of_meta_gradient', log_scale=True)
    save_plot("Cosine Similarity", maml["cosines"], dcml["cosines"], 'gradient2/cos_sim')
    save_plot("L2 Distance", maml["l2s"], dcml["l2s"], 'gradient2/distance')