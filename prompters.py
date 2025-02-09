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
import torch.nn.functional as F


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
        self.prompt_dict['pad_left'] = nn.Parameter(
            torch.randn([3, self.image_size - self.pad_size * 2, self.pad_size]))
        self.prompt_dict['pad_right'] = nn.Parameter(
            torch.randn([3, self.image_size - self.pad_size * 2, self.pad_size]))

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


class SimpleConvolution(nn.Module):
    def __init__(self, args, in_channels, out_channels, kernel_size, stride, padding, use_bias, groups=1,
                 dilation_rate=1):
        super(SimpleConvolution, self).__init__()
        num_filters = out_channels
        self.args = args
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation_rate = int(dilation_rate)
        self.use_bias = use_bias
        self.groups = int(groups)
        self.weight = nn.Parameter(torch.empty(num_filters, in_channels, kernel_size, kernel_size))
        nn.init.xavier_uniform_(self.weight)

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, images, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
                weight = weight.squeeze(0)
                bias = bias.squeeze(0)
            else:
                (weight) = params["weight"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv2d(input=images, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)

        return out


class PromptConvolution(nn.Module):
    def __init__(self, args):
        super(PromptConvolution, self).__init__()
        self.args = args
        self.prompt_dict = nn.ModuleDict()
        self.key_name = 'conv'

        self.build_prompt()

    def build_prompt(self):
        self.prompt_dict[self.key_name] = SimpleConvolution(args=self.args,
                                                            in_channels=3,
                                                            out_channels=3,
                                                            kernel_size=3,
                                                            stride=1,
                                                            padding=1,
                                                            use_bias=True)

    def forward(self, x, prompted_params=None):

        if prompted_params is not None:
            prompted_params = extract_top_level_dict(current_dict=prompted_params)
        else:
            print("prompted_params is None")

        prompted_params = prompted_params[self.key_name]

        ideal_prompt = self.prompt_dict[self.key_name](images=x, params=prompted_params)
        prompted_image = x + ideal_prompt

        return prompted_image


class PromptSelfAttention(nn.Module):
    def __init__(self, args):
        super(PromptSelfAttention, self).__init__()
        self.args = args
        self.prompt_dict = nn.ModuleDict()
        self.softmax = nn.Softmax(dim=-1)

        self.query_layer = 'query_proj '
        self.key_layer = 'key_proj'
        self.value_layer = 'value_proj'

        self.build_prompt()

    def build_prompt(self):

        in_channels = 3
        embed_dim = 64

        self.prompt_dict[self.query_layer] = SimpleConvolution(args=self.args,
                                                             in_channels=in_channels,
                                                             out_channels=embed_dim,
                                                             kernel_size=1,
                                                             stride=1,
                                                             padding=0,
                                                             use_bias=True)

        self.prompt_dict[self.key_layer] = SimpleConvolution(args=self.args,
                                                          in_channels=in_channels,
                                                          out_channels=embed_dim,
                                                          kernel_size=1,
                                                          stride=1,
                                                          padding=0,
                                                          use_bias=True)

        self.prompt_dict[self.value_layer] = SimpleConvolution(args=self.args,
                                                            in_channels=in_channels,
                                                            out_channels=in_channels,
                                                            kernel_size=1,
                                                            stride=1,
                                                            padding=0,
                                                            use_bias=True)

    def forward(self, x, prompted_params=None):

        batch_size, channels, height, width = x.shape

        if prompted_params is not None:
            prompted_params = extract_top_level_dict(current_dict=prompted_params)
        else:
            print("prompted_params is None")

        query_proj = prompted_params[self.query_layer]
        key_proj = prompted_params[self.key_layer]
        value_proj = prompted_params[self.value_layer]

        # Key, Value 변환 (B, 100, H, W) & (B, 3, H, W)
        query = self.prompt_dict[self.query_layer](images=x, params=query_proj).view(batch_size, -1, height * width)  # (B, embed_dim, H*W)
        key = self.prompt_dict[self.key_layer](images=x, params=key_proj).view(batch_size, -1, height * width)  # (B, embed_dim, H*W)
        value = self.prompt_dict[self.value_layer](images=x, params=value_proj).view(batch_size, channels, height * width)  # (B, 3, H*W)

        # Attention score 계산 (Q @ K^T) / sqrt(embed_dim)
        scores = torch.matmul(query.permute(0, 2, 1), key) / (query.shape[1] ** 0.5)  # (B, H*W, H*W)
        attention_weights = self.softmax(scores)  # (B, H*W, H*W)

        # Attention 적용하여 Prompt 생성 (B, 3, H*W)
        prompt = torch.matmul(value, attention_weights).view(batch_size, channels, height, width)  # (B, 3, H, W)

        return x + prompt

class TaskAwareAttention(nn.Module):
    def __init__(self, args):
        super(TaskAwareAttention, self).__init__()
        self.args = args
        self.prompt_dict = nn.ModuleDict()
        self.softmax = nn.Softmax(dim=-1)

        self.key_layer = 'key_proj'
        self.value_layer = 'value_proj'

        self.build_prompt()

    def build_prompt(self):

        """
        image: (B, 3, 84, 84) - Key, Value 역할
        task_embedding: (B, 100) - Query 역할
        """

        in_channels = 3
        embed_dim = 64


        self.prompt_dict[self.key_layer] = SimpleConvolution(args=self.args,
                                                          in_channels=in_channels,
                                                          out_channels=embed_dim,
                                                          kernel_size=1,
                                                          stride=1,
                                                          padding=0,
                                                          use_bias=True)

        self.prompt_dict[self.value_layer] = SimpleConvolution(args=self.args,
                                                            in_channels=in_channels,
                                                            out_channels=in_channels,
                                                            kernel_size=1,
                                                            stride=1,
                                                            padding=0,
                                                            use_bias=True)

    def forward(self, x, task_embedding, prompted_params=None):

        batch_size, channels, height, width = x.shape

        if prompted_params is not None:
            prompted_params = extract_top_level_dict(current_dict=prompted_params)
        else:
            print("prompted_params is None")

        # query_proj = prompted_params[self.query_layer]
        key_proj = prompted_params[self.key_layer]
        value_proj = prompted_params[self.value_layer]

        device=torch.device("cuda")
        # query = torch.randn(batch_size, 1, 64).to(device=device) # (B, 1, embed_dim)

        # Key, Value
        key = self.prompt_dict[self.key_layer](images=x, params=key_proj).view(batch_size, -1, height * width)  # (B, embed_dim, H*W)
        value = self.prompt_dict[self.value_layer](images=x, params=value_proj).view(batch_size, channels, height * width)  # (B, 3, H*W)

        # Attention Score 계산 (Query @ Key^T) / sqrt(embed_dim)
        scores = torch.matmul(task_embedding, key) / (key.shape[1] ** 0.5)  # (B, 1, H*W)
        attention_weights = self.softmax(scores)  # (B, 1, H*W)

        # Attention 적용하여 Prompt 생성 (B, 3, H*W)
        prompt = (value * attention_weights).view(batch_size, -1, height, width)
        # prompt = torch.matmul(value, attention_weights.transpose(1, 2)).view(batch_size, -1, height, width)  # (B, 3, 84, 84)

        # print("Value Shape:", value.shape)  # (B, 3, H*W)
        # print("Attention Weights Shape:", attention_weights.shape)  # (B, 1, H*W)
        # print("Transposed Attention Weights Shape:", attention_weights.transpose(1, 2).shape)  # (B, H*W, 1)
        # print("Matmul Output Shape:", torch.matmul(value, attention_weights.transpose(1, 2)).shape)

        return x + prompt


def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)


def arbiter(args):
    return PromptArbiter(args)


def convolution(args):
    return PromptConvolution(args)


def cross_attention(args):
    return PromptSelfAttention(args)

def task_aware_attention(args):
    return TaskAwareAttention(args)