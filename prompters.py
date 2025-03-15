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

    def forward(self, x, params=None):
        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weight"], params["bias"]
                weight = weight.squeeze(0)
                bias = bias.squeeze(0)
            else:
                (weight) = params["weight"]
                weight = weight.squeeze(0)
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weight, self.bias
            else:
                weight = self.weight
                bias = None

        out = F.conv2d(input=x, weight=weight, bias=bias, stride=self.stride,
                       padding=self.padding, dilation=self.dilation_rate, groups=self.groups)

        return out


class SimpleLinearLayer(nn.Module):
    def __init__(self, args, num_filters, output_size, use_bias):
        super(SimpleLinearLayer, self).__init__()

        self.args = args
        self.use_bias = use_bias
        self.weights = nn.Parameter(torch.ones(output_size, num_filters))
        nn.init.xavier_uniform_(self.weights)
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x, params=None):

        if params is not None:
            params = extract_top_level_dict(current_dict=params)
            if self.use_bias:
                (weight, bias) = params["weights"], params["bias"]
                weight = weight.squeeze(0)
                bias = bias.squeeze(0)
            else:
                (weight) = params["weights"]
                bias = None
        else:
            if self.use_bias:
                weight, bias = self.weights, self.bias
            else:
                weight = self.weights
                bias = None

        out = F.linear(input=x, weight=weight, bias=bias)
        return out


class PromptGenerator(nn.Module):
    def __init__(self, args, device, nz=100, ngf=64, img_size=84, nc=3):
        super(PromptGenerator, self).__init__()

        self.args = args
        self.device = device

        self.nz = nz
        self.ngf = ngf
        self.nc = nc

        self.input_shape = nc * img_size * img_size # input_shape: The image input shape in the form (b, c, h, w)

        self.init_size = img_size // 4

        self.build_network()

        print("PromptGenerator params")
        for name, param in self.named_parameters():
            print(name, param.shape)

    def build_network(self):

        ngf = self.ngf
        nz = self.nz
        init_size = self.init_size
        nc = self.nc

        self.prompt_dict = nn.ModuleDict()

        self.prompt_dict['l1'] = SimpleLinearLayer(args=self.args, output_size= ngf * 2 * init_size ** 2, num_filters=nz, use_bias=True)

        self.batch_norm1 = nn.BatchNorm2d(ngf * 2)
        self.upsampling1 = nn.Upsample(scale_factor=2)

        self.prompt_dict['conv1'] = SimpleConvolution(args=self.args,
                                       in_channels=ngf*2,
                                       out_channels=ngf*2,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       use_bias=False)

        self.batch_norm2 = nn.BatchNorm2d(ngf * 2)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.upsampling2 = nn.Upsample(scale_factor=2)

        self.prompt_dict['conv2'] = SimpleConvolution(args=self.args,
                                       in_channels=ngf*2,
                                       out_channels=ngf,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       use_bias=False)

        self.batch_norm3 = nn.BatchNorm2d(ngf)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.prompt_dict['conv3']  = SimpleConvolution(args=self.args,
                                       in_channels=ngf,
                                       out_channels=nc,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       use_bias=False)

        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, task_embedding, prompted_params=None):

        print("prompted_params ==", prompted_params.keys())

        if prompted_params is not None:
            prompted_params = extract_top_level_dict(current_dict=prompted_params)
        else:
            print("prompted_params is None")

        l1_param = prompted_params['l1']
        conv1_param = prompted_params['conv1']
        conv2_param = prompted_params['conv2']
        conv3_param = prompted_params['conv3']

        out = self.prompt_dict['l1'](x=task_embedding, params=l1_param)

        # print("out shape == ", out.shape)

        out = out.view(out.shape[0], -1, self.init_size, self.init_size)

        out = self.batch_norm1(out)
        out = self.upsampling1(out)

        out = self.prompt_dict['conv1'](x=out, params=conv1_param)

        out = self.batch_norm2(out)
        out = self.leaky_relu2(out)
        out = self.upsampling2(out)

        out = self.prompt_dict['conv2'](x=out, params=conv2_param)

        out = self.batch_norm3(out)
        out = self.leaky_relu3(out)

        out = self.prompt_dict['conv3'](x=out, params=conv3_param)
        out = self.leaky_relu4(out)

        return out


    def zero_grad(self, params=None):
        if params is None:
            for param in self.parameters():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
        else:
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None

class StepPromptAdapter(nn.Module):
    def __init__(self, input_dim, num_prompt_generator_layers, args, device):
        super(StepPromptAdapter,self).__init__()

        self.args = args
        self.device = device
        output_dim = num_prompt_generator_layers * 2  # 2 for weight and bias, another 2 for multiplier and offset

        self.linear1 = nn.Linear(input_dim, input_dim)
        self.activation = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(input_dim, output_dim)

        self.multiplier_bias = nn.Parameter(torch.zeros(output_dim // 2))
        self.offset_bias = nn.Parameter(torch.zeros(output_dim // 2))

    def forward(self, task_embeddig, num_step, prompt_generator_params):

        task_embeddig = task_embeddig.squeeze(0)

        out = self.linear1(task_embeddig)
        out = F.relu_(out)
        out = self.linear2(out)

        generated_multiplier, generated_offset = torch.chunk(out, chunks=2, dim=-1)

        print("prompt_generator_params == ", prompt_generator_params.keys())
        print("out == ", out.shape)
        print("self.multiplier_bias == ", self.multiplier_bias.shape)
        print("generated_multiplier == ", generated_multiplier.shape)
        print("generated_offset == ", generated_offset.shape)
        print("self.offset_bias == ", self.offset_bias.shape)

        # i = 0
        updated_prompted_weights = dict()
        for key, val in prompt_generator_params.items():
            # if 'step{}'.format(num_step) in key:
            print(f"val shape for {key}: {val.shape}")

            updated_prompted_weights[key] = (1 + self.multiplier_bias * generated_multiplier) * val + \
                                            self.offset_bias * generated_offset

            # updated_prompted_weights[key] = (1 + self.multiplier_bias[i] * generated_multiplier[i]) * val + \
            #                             self.offset_bias[i] * generated_offset[i]
            # i += 1

        return updated_prompted_weights


class PromptAdapter(nn.Module):
    def __init__(self, input_dim, num_prompt_generator_layers, args, device):
        super(PromptAdapter, self).__init__()

        self.device = device
        self.args = args

        self.num_steps = args.number_of_training_steps_per_iter # number of inner-loop steps

        self.loss_adapter = nn.ModuleList()
        for i in range(self.num_steps):
            self.loss_adapter.append(StepPromptAdapter(input_dim, num_prompt_generator_layers, args=args, device=device))

    def forward(self, task_embeddig, num_step, prompt_generator_params):
        return self.loss_adapter[num_step](task_embeddig, num_step, prompt_generator_params)


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

        self.query_layer = 'query_proj'
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

        # self.query_layer = 'query_proj'
        self.key_layer = 'key_proj'
        self.value_layer = 'value_proj'

        self.build_prompt()

    def build_prompt(self):

        """
        image: (B, 3, 84, 84) - Key, Value 역할
        task_embedding: (B, 100) - Query 역할
        """

        in_channels = 3
        embed_dim = 100
        kernel_size = 1
        stride = 1
        padding = 0
        # embed_dim = self.args.num_text_embedding_params
        task_dim = self.args.num_text_embedding_params

        # self.prompt_dict[self.query_layer] = SimpleLinearLayer(args=self.args,
        #                                                        num_filters=task_dim,
        #                                                        output_size=embed_dim,
        #                                                        use_bias=True)

        self.prompt_dict[self.key_layer] = SimpleConvolution(args=self.args,
                                                             in_channels=in_channels,
                                                             out_channels=embed_dim,
                                                             kernel_size=kernel_size,
                                                             stride=stride,
                                                             padding=padding,
                                                             use_bias=True)

        self.prompt_dict[self.value_layer] = SimpleConvolution(args=self.args,
                                                               in_channels=in_channels,
                                                               out_channels=in_channels,
                                                               kernel_size=kernel_size,
                                                               stride=stride,
                                                               padding=padding,
                                                               use_bias=True)

    def forward(self, x,task_embedding, prompted_params=None):

        batch_size, channels, height, width = x.shape

        if prompted_params is not None:
            prompted_params = extract_top_level_dict(current_dict=prompted_params)
        else:
            print("prompted_params is None")

        # query_proj = prompted_params[self.query_layer]
        key_proj = prompted_params[self.key_layer]
        value_proj = prompted_params[self.value_layer]

        # Key, Value
        key = self.prompt_dict[self.key_layer](images=x, params=key_proj).view(batch_size, -1, height * width)  # (B, embed_dim, H*W)
        value = self.prompt_dict[self.value_layer](images=x, params=value_proj).view(batch_size, channels, height * width)  # (B, 3, H*W)

        # query = self.prompt_dict[self.query_layer](x=task_embedding, params=query_proj).unsqueeze(1)  # (B, 1, embed_dim)

        # print("key == ", key.shape)
        # print("query == ", task_embedding.shape)
        # print("value == ", value.shape)

        # Attention Score 계산 (Query @ Key^T) / sqrt(embed_dim)
        scores = torch.matmul(task_embedding, key) / (key.shape[1] ** 0.5)  # (B, 1, H*W) query_layer를 사용하지 않을때
        # scores = torch.matmul(query, key) / (key.shape[1] ** 0.5)  # (B, 1, H*W)

        # scores = torch.matmul(query, key) / (key.shape[1] ** 0.5)  # (B, 1, H*W)
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