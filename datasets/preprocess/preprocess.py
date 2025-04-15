import os
import sys
import shutil
import argparse
import tarfile
import zipfile
import random


def mini_imagenet():
    with tarfile.open('mini_imagenet_full_size.tar.bz2', 'r') as tar:
        tar.extractall()
    os.rename('mini_imagenet_full_size', 'mini_imagenet')


def tiered_imagenet():
    with tarfile.open('tiered_imagenet.tar', 'r') as tar:
        tar.extractall()


def CIFAR_FS():
    phase_list = ['train', 'val', 'test']
    with zipfile.ZipFile('cifar100.zip', 'r') as zip_ref:
        zip_ref.extractall()

    for phase in phase_list:
        os.makedirs('CIFAR_FS/{}'.format(phase))

    for phase in phase_list:
        classes_info_dir = 'cifar100/splits/bertinetto/{}.txt'.format(phase)
        f = open(classes_info_dir, 'r')
        for line in f.readlines():
            class_name = line.strip()
            shutil.move('cifar100/data/{}'.format(class_name), 'CIFAR_FS/{}/{}'.format(phase, class_name))

    shutil.rmtree('cifar100')

def FC100():
    phase_list = ['train', 'val', 'test']

    with zipfile.ZipFile('cifar100.zip', 'r') as zip_ref:
        zip_ref.extractall()

    for phase in phase_list:
        os.makedirs('FC100/{}'.format(phase))

    for phase in phase_list:
        classes_info_dir = 'preprocess/FC100_split_{}.txt'.format(phase)
        f = open(classes_info_dir, 'r')
        for line in f.readlines():
            class_name = line.strip()
            shutil.move('cifar100/data/{}'.format(class_name), 'FC100/{}/{}'.format(phase, class_name))

    shutil.rmtree('cifar100')



def CUB():
    phase_list = ['train', 'val', 'test']
    with tarfile.open('CUB_200_2011.tgz', 'r') as tar:
        tar.extractall()
    for phase in phase_list:
        os.makedirs('CUB/{}'.format(phase))
    for phase in phase_list:
        classes_info_dir = 'preprocess/CUB_split_{}.txt'.format(phase)
        f = open(classes_info_dir, 'r')
        for line in f.readlines():
            class_name = line.strip()
            shutil.move('CUB_200_2011/images/{}'.format(class_name), 'CUB/{}/{}'.format(phase, class_name))
    os.remove('attributes.txt')
    shutil.rmtree('CUB_200_2011')


def StandfordCars():

    source_root = 'Cars'
    target_root = 'Cars_split'

    # 클래스 폴더 목록
    class_dirs = os.listdir(source_root)
    random.seed(42)  # 재현 가능성 확보
    random.shuffle(class_dirs)  # 무작위 섞기!

    # split 비율
    num_train = 98
    num_val = 49
    num_test = 49

    train_classes = class_dirs[:num_train]
    val_classes = class_dirs[num_train:num_train + num_val]
    test_classes = class_dirs[num_train + num_val:]

    splits = {
        'train': train_classes,
        'val': val_classes,
        'test': test_classes
    }

    # 복사
    for split_name, class_list in splits.items():
        for cls in class_list:
            src = os.path.join(source_root, cls)
            dst = os.path.join(target_root, split_name, cls)
            os.makedirs(dst, exist_ok=True)
            for fname in os.listdir(src):
                shutil.copy2(os.path.join(src, fname), os.path.join(dst, fname))


if __name__ == '__main__':

    # python preprocess/preprocess.py --datasets CIFAR_FS

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', 
                        choices=['mini_imagenet', 'tiered_imagenet', 'CIFAR_FS', 'FC100' ,'CUB', 'StandfordCars'],
                        help='Dataset name to preprocess.')
    args = parser.parse_args()

    for dataset in args.datasets:
        if os.path.isdir(dataset):
            shutil.rmtree(dataset)
        getattr(sys.modules[__name__], dataset)()
