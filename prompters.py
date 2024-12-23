# This source code is based on "Exploring Visual Prompts for Adapting Large-Scale Models".
# --------------------------------------------------------
# References:
# Paper: https://arxiv.org/abs/2203.17274
# Project: https://hjbahng.github.io/visual_prompting/
# Code: https://github.com/hjbahng/visual_prompting
# --------------------------------------------------------


import torch
import torch.nn as nn
import numpy as np
import arbiter

def extract_top_level_dict(current_dict):

    output_dict = dict()
    for key in current_dict.keys():
        name = key.replace("prompt.", "")
        name = name.replace("prompt_dict.", "")
        top_level = name.split(".")[0]
        sub_level = ".".join(name.split(".")[1:])

        if top_level not in output_dict:
            if sub_level == "":
                output_dict[top_level] = current_dict[key]
            else:
                output_dict[top_level] = {sub_level: current_dict[key]}
        else:
            new_item = {key: value for key, value in output_dict[top_level].items()}
            new_item[sub_level] = current_dict[key]
            output_dict[top_level] = new_item

    return output_dict

class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        self.pad_size = args.prompt_size
        self.image_size = args.image_size
        self.base_size = self.image_size - self.pad_size * 2

        self.prompt_dict = nn.ParameterDict()

        self.build_prompt()

    def build_prompt(self):

        # self.prompt_dict['pad_up'] = nn.Parameter(torch.randn([1, 3, self.pad_size, self.image_size]))
        # 원래 batch 때문에 1이 추가되어 있는데, 나는 few_shot_learning_system에서 batch를 추가해주기때문에 생략한다.
        self.prompt_dict['pad_up'] = nn.Parameter(torch.randn([3, self.pad_size, self.image_size]))
        self.prompt_dict['pad_down'] = nn.Parameter(torch.randn([3, self.pad_size, self.image_size]))
        self.prompt_dict['pad_left'] = nn.Parameter(torch.randn([3, self.image_size - self.pad_size * 2, self.pad_size]))
        self.prompt_dict['pad_right'] = nn.Parameter(torch.randn([3, self.image_size - self.pad_size * 2, self.pad_size]))

    def forward(self, x, prompted_params=None):
        if prompted_params is not None:
            param_dict = extract_top_level_dict(current_dict=prompted_params)
            pad_up = param_dict['pad_up']
            pad_down = param_dict['pad_down']
            pad_left = param_dict['pad_left']
            pad_right = param_dict['pad_right']
        else:
            # 원래 batch 단위롤 위해 unsqueeze(0)을 통해 1차원을 추가한다
            pad_up = self.prompt_dict['pad_up'].unsqueeze(0)
            pad_down = self.prompt_dict['pad_down'].unsqueeze(0)
            pad_left = self.prompt_dict['pad_left'].unsqueeze(0)
            pad_right = self.prompt_dict['pad_right'].unsqueeze(0)

        # print("pad_up == ", pad_up.shape)
        # print("pad_down == ", pad_down.shape)
        # print("pad_left == ", pad_left.shape)
        # print("pad_right == ", pad_right.shape)

        base = torch.zeros(1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([pad_left, base, pad_right], dim=3)
        prompt = torch.cat([pad_up, prompt, pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        return x + prompt

class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size

        self.prompt_dict = nn.ParameterDict()

        self.build_prompt()

    def build_prompt(self):
        self.prompt_dict['patch'] = nn.Parameter(torch.randn([3, self.psize, self.psize]))

    def forward(self, x, prompted_params=None):

        if prompted_params is not None:
            param_dict = extract_top_level_dict(current_dict=prompted_params)
            patch = param_dict['patch']
        else:
            patch = self.prompt_dict['patch'].unsqueeze(0)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.psize, :self.psize] = patch

        return x + prompt

class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size

        self.prompt_dict = nn.ParameterDict()

        self.build_prompt()

    def build_prompt(self):
        self.prompt_dict['patch'] = nn.Parameter(torch.randn([3, self.psize, self.psize]))

    def forward(self, x, prompted_params=None):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        if prompted_params is not None:
            param_dict = extract_top_level_dict(current_dict=prompted_params)
            patch = param_dict['patch']
        else:
            patch = self.prompt_dict['patch'].unsqueeze(0)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = patch

        return x + prompt

class PromptArbiter(nn.Module):
    def __init__(self, args):
        super(PromptArbiter, self).__init__()

        self.isize = args.image_size
        self.prompt_dict = nn.ParameterDict()
        self.build_prompt()

    def build_prompt(self):
        self.prompt_dict['arbiter'] = nn.Parameter(torch.randn([3, self.isize, self.isize]))

    def forward(self, x, prompted_params=None):
        if prompted_params is not None:
            param_dict = extract_top_level_dict(current_dict=prompted_params)
            patch = param_dict['arbiter']
        else:
            patch = self.prompt_dict['arbiter'].unsqueeze(0)

        prompt = torch.zeros([1, 3, self.isize, self.isize]).cuda()
        prompt[:, :, :self.isize, :self.isize] = patch

        return x + prompt


def padding(args):
    return PadPrompter(args)

def fixed_patch(args):
    return FixedPatchPrompter(args)

def random_patch(args):
    return RandomPatchPrompter(args)

def arbiter(args):
    return PromptArbiter(args)