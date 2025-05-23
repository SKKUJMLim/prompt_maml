from data import MetaLearningSystemDataLoader

from experiment_builder import ExperimentBuilder
from few_shot_learning_system import MAMLFewShotClassifier
# from few_shot_learning_system_ALFA import MAMLFewShotClassifier

from utils.parser_utils import get_args
from utils.dataset_tools import maybe_unzip_dataset

from multiprocessing import freeze_support


''' MAML experiment'''
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML/MAML_5way_5shot_filter64_miniImagenet.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML/MAML_5way_5shot_filter64_tieredImagenet.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML/MAML_5way_5shot_filter64_CIFAR_FS.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML/MAML_5way_5shot_filter64_FC100.json --gpu_to_use 0

## python train_maml_system.py --name_of_args_json_file experiment_config/MAML/ANIL_5way_5shot_filter64_miniImagenet.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML/BOIL_5way_5shot_filter64_miniImagenet.json --gpu_to_use 0

## python train_maml_system.py --name_of_args_json_file experiment_config/L2F/L2F_5way_5shot_miniImagenet.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/ALFA/ALFA_5way_5shot_miniImagenet.json --gpu_to_use 0


'''[VGG] MAML+Prompt experiment'''
## (filter 128) miniImageNet
# python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/filter128/MAML_Prompt_padding_5way_5shot_filter128_miniImagenet.json --gpu_to_use 0

## (filter 64) miniImageNet
# python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/MAML_Prompt_padding_5way_1shot_filter64_miniImagenet.json --gpu_to_use 0
# python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/MAML_Prompt_padding_5way_5shot_filter64_miniImagenet.json --gpu_to_use 0
# python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/MAML_Prompt_padding_5way_1shot_filter64_tieredImagenet.json --gpu_to_use 0
# python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/MAML_Prompt_padding_5way_5shot_filter64_tieredImagenet.json --gpu_to_use 0

# python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/MAML_Prompt_padding_5way_1shot_filter64_CIFAR_FS.json --gpu_to_use 0
# python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/MAML_Prompt_padding_5way_5shot_filter64_CIFAR_FS.json --gpu_to_use 0
# python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/MAML_Prompt_padding_5way_1shot_filter64_FC100.json --gpu_to_use 0
# python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/MAML_Prompt_padding_5way_5shot_filter64_FC100.json --gpu_to_use 0

## (Fixed patch & Random patch)
# python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/MAML_Prompt_fixed_patch_5way_5shot_filter64_miniImagenet.json --gpu_to_use 0
# python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/MAML_Prompt_random_patch_5way_5shot_filter64_miniImagenet.json --gpu_to_use 0

'''[Resnet12] MAML+Prompt experiment'''
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/Resnet12/MAML_Prompt_padding_5way_1shot_Resnet12_miniImagenet.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/Resnet12/MAML_Prompt_padding_5way_5shot_Resnet12_miniImagenet.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/Resnet12/MAML_Prompt_padding_5way_1shot_Resnet12_tieredImagenet.json --gpu_to_use 0
## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/Resnet12/MAML_Prompt_padding_5way_5shot_Resnet12_tieredImagenet.json --gpu_to_use 0

## python train_maml_system.py --name_of_args_json_file experiment_config/MAML+Prompt/Resnet12/MAML_Prompt_fixed_patch_5way_5shot_Resnet12_miniImagenet.json --gpu_to_use 0

if __name__ == '__main__':
    freeze_support()

    # Combines the arguments, model, data and experiment builders to run an experiment
    args, device = get_args()

    # 모델을 구성한다
    model = MAMLFewShotClassifier(args=args, device=device,
                                  im_shape=(2, 3,
                                            args.image_height, args.image_width))
    maybe_unzip_dataset(args=args)

    # 데이터를 불러온다
    data = MetaLearningSystemDataLoader

    # 학습
    maml_system = ExperimentBuilder(model=model, data=data, args=args, device=device)
    maml_system.run_experiment()
