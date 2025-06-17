import argparse
import os

from utils.gradient_conflict import (
    LAYER_NAMES,
    get_avg_cos_sim_all_layers,
    plot_cosine_similarity_layerwise_individual,
    plot_cosine_similarity_layerwise_subplots,
    get_gsnr_all_layers,
    plot_gsnr_individual,
    plot_gsnr_subplots,
    get_l2_distance_all_layers,
    plot_l2_distance_individual,
    plot_l2_distance_subplots,
    get_pairwise_cosine_all_layers,
    plot_pairwise_cosine_individual,
    plot_pairwise_cosine_subplots
)

if __name__ == '__main__':

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # python ablation_study.py --name pairwise_cosine_similarity
    # python ablation_study.py --name avg_cosine_similarity gsnr l2_distance pairwise_cosine_similarity

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, nargs='+',
                        choices=['avg_cosine_similarity', 'gsnr', 'l2_distance', 'pairwise_cosine_similarity'],
                        help='Dataset name to preprocess.')
    args = parser.parse_args()

    maml_base_path = "MAML_5way_5shot_filter64_miniImagenet"
    our_base_path = "MAML_Prompt_padding_5way_5shot_filter128_miniImagenet"
    epoch_list = list(range(0, 2))


    if 'avg_cosine_similarity' in args.name:
        maml_all_results = get_avg_cos_sim_all_layers(maml_base_path, epoch_list, LAYER_NAMES)
        our_all_results = get_avg_cos_sim_all_layers(our_base_path, epoch_list, LAYER_NAMES)

        # 시각화
        plot_cosine_similarity_layerwise_individual(
            maml_all_results, our_all_results, epoch_list
        )
        plot_cosine_similarity_layerwise_subplots(
            maml_all_results, our_all_results, epoch_list
        )
    elif 'gsnr' in args.name:

        maml_gsnr = get_gsnr_all_layers(maml_base_path, epoch_list, LAYER_NAMES)
        our_gsnr = get_gsnr_all_layers(our_base_path, epoch_list, LAYER_NAMES)

        # 시각화
        plot_gsnr_individual(maml_gsnr, our_gsnr, epoch_list)
        plot_gsnr_subplots(maml_gsnr, our_gsnr, epoch_list)

    elif 'l2_distance' in args.name:
        # L2 distance 계산
        maml_l2 = get_l2_distance_all_layers(maml_base_path, epoch_list, LAYER_NAMES)
        our_l2 = get_l2_distance_all_layers(our_base_path, epoch_list, LAYER_NAMES)

        # 그래프 저장
        plot_l2_distance_individual(maml_l2, our_l2, epoch_list)
        plot_l2_distance_subplots(maml_l2, our_l2, epoch_list)

    elif 'pairwise_cosine_similarity' in args.name:
        # pairwise cosine similarity 계산
        maml_pairwise = get_pairwise_cosine_all_layers(maml_base_path, epoch_list, LAYER_NAMES)
        our_pairwise = get_pairwise_cosine_all_layers(our_base_path, epoch_list, LAYER_NAMES)

        # 그래프 저장
        plot_pairwise_cosine_individual(maml_pairwise, our_pairwise, epoch_list)
        plot_pairwise_cosine_subplots(maml_pairwise, our_pairwise, epoch_list)
