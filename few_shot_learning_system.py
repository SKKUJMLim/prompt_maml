import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict

from meta_neural_network_architectures import VGGReLUNormNetwork, ResNet12
# from inner_loop_optimizers import GradientDescentLearningRule, LSLRGradientDescentLearningRule
from inner_loop_optimizers_weightdecay import GradientDescentLearningRule, LSLRGradientDescentLearningRule

from data_augmentation import mixup_data, random_flip_like_torchvision, random_flip_batchwise
from utils.basic import count_params_by_key


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


class MAMLFewShotClassifier(nn.Module):
    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = 0
        self.rng = set_torch_seed(seed=args.seed)
        self.csv_exist = True

        if self.args.backbone == 'ResNet12':
            self.classifier = ResNet12(im_shape=self.im_shape, num_output_classes=self.args.
                                       num_classes_per_set,
                                       args=args, device=device, meta_classifier=True).to(device=self.device)
        else:
            self.classifier = VGGReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.
                                                 num_classes_per_set,
                                                 args=args, device=device, meta_classifier=True).to(device=self.device)

        self.task_learning_rate = args.init_inner_loop_learning_rate
        names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

        prompted_weights_copy = {}
        if self.args.prompter:
            prompted_weights_copy = {key: value for key, value in names_weights_copy.items() if 'prompt' in key}

        names_weights_copy = {key: value for key, value in names_weights_copy.items() if 'layer_dict' in key}

        if self.args.learnable_per_layer_per_step_inner_loop_learning_rate:
            self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                        args=self.args,
                                                                        init_learning_rate=self.task_learning_rate,
                                                                        total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                        use_learnable_learning_rates=True)
            self.inner_loop_optimizer.initialise(names_weights_dict=names_weights_copy,
                                                 prompted_weights_dict=prompted_weights_copy)
        else:
            self.inner_loop_optimizer = GradientDescentLearningRule(device=device, args=self.args,
                                                                    learning_rate=self.task_learning_rate)

        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            if value.requires_grad:
                print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)

        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

        # self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False,
                                    weight_decay=self.args.init_inner_loop_weight_decay)

        # if self.args.prompter:
        #     if self.args.prompt_random_init:
        #         self.optimizer = optim.Adam([
        #             {'params': self.trainable_parameters(), 'lr': args.meta_learning_rate},
        #         ], amsgrad=False, weight_decay=self.args.init_inner_loop_weight_decay)
        #     else:
        #         self.optimizer = optim.Adam([
        #             {'params': self.trainable_parameters(), 'lr': args.meta_learning_rate},
        #             {'params': self.trainable_prompt_parameters(), 'lr': args.meta_learning_rate},
        #             {'params': self.inner_loop_optimizer.parameters(), 'lr': args.meta_learning_rate},
        #         ], amsgrad=False, weight_decay=self.args.init_inner_loop_weight_decay)
        # else:
        #     self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False, weight_decay=self.args.init_inner_loop_weight_decay)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            print(torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                self.to(torch.cuda.current_device())
                self.classifier = nn.DataParallel(module=self.classifier)
            else:
                self.to(torch.cuda.current_device())

            self.device = torch.cuda.current_device()

    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    if "norm_layer" not in name:
                        param_dict[name] = param.to(device=self.device)

        return param_dict

    def apply_inner_loop_update(self, loss, names_weights_copy, prompted_weights_copy, use_second_order,
                                current_step_idx, current_iter, training_phase):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
            self.classifier.module.zero_grad(params=prompted_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)
            self.classifier.zero_grad(params=prompted_weights_copy)

        prompted_grads_copy = {}
        if self.args.prompter:

            # Classifier만 추출해서 gradient 계산 대상 지정
            # names_weights_copy = {
            #     k: v for k, v in names_weights_copy.items() if 'classifier' in k or 'linear' in k
            # }

            grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                        create_graph=use_second_order, allow_unused=True,
                                        retain_graph=True)  ###### retrain_graph 추가
            names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
            names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

            prompt_grads = torch.autograd.grad(loss, prompted_weights_copy.values(),
                                               create_graph=use_second_order, allow_unused=True)
            prompted_grads_copy = dict(zip(prompted_weights_copy.keys(), prompt_grads))
            prompted_weights_copy = {key: value[0] for key, value in prompted_weights_copy.items()}

        else:

            # # Classifier만 추출해서 gradient 계산 대상 지정
            # names_weights_copy = {
            #     k: v for k, v in names_weights_copy.items() if 'classifier' in k or 'linear' in k
            # }

            grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                        create_graph=use_second_order, allow_unused=True)
            names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
            names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        for key, prompt_grad in prompted_grads_copy.items():
            if prompt_grad is None:
                print('Prompt Grads not found for inner loop parameter', key)
            prompted_grads_copy[key] = prompted_grads_copy[key].sum(dim=0)

        names_weights_copy, prompted_weights_copy = self.inner_loop_optimizer.update_params(
            names_weights_dict=names_weights_copy,
            names_grads_wrt_params_dict=names_grads_copy,
            prompted_weights_dict=prompted_weights_copy,
            prompted_grads_wrt_params_dict=prompted_grads_copy,
            num_step=current_step_idx,
            current_iter=current_iter,
            training_phase=training_phase)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}

        prompted_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in prompted_weights_copy.items()}

        return names_weights_copy, prompted_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        # total_losses = torch.stack(total_losses)
        # weights = torch.nn.functional.softmax(total_losses, dim=0)
        # weighted_loss  = torch.sum(weights * total_losses)
        # losses['loss'] = weighted_loss

        losses['loss'] = torch.mean(torch.stack(total_losses))
        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase,
                current_iter):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = ncs

        total_losses = []
        layerwise_task_grads = defaultdict(list)  # {layer_name: [grad_task1, grad_task2, ...]}
        total_accuracies = []
        total_support_accuracies = [[] for i in range(num_steps)]
        total_target_accuracies = [[] for i in range(num_steps)]
        per_task_target_preds = [[] for i in range(len(x_target_set))]

        information = {}

        if torch.cuda.device_count() > 1:
            self.classifier.module.zero_grad()
        else:
            self.classifier.zero_grad()
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):
            task_losses = []
            task_accuracies = []
            per_step_support_accuracy = []
            per_step_target_accuracy = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}

            prompted_weights_copy = {}
            if self.args.prompter:
                if self.args.random_prompt_init:
                    prompted_weights_copy = {key: torch.randn_like(value, requires_grad=True) for key, value in names_weights_copy.items() if 'prompt' in key}
                else:
                    prompted_weights_copy = {key: value for key, value in names_weights_copy.items() if 'prompt' in key}

            names_weights_copy = {key: value for key, value in names_weights_copy.items() if 'layer_dict' in key}

            # parameter 수 체크
            # print("visual prompt parameter count == ", count_params_by_key(param_dict=prompted_weights_copy, keyword='prompt'))
            # print("backbone parameter count == ", count_params_by_key(param_dict=names_weights_copy, keyword='conv'))
            # print("classifier parameter  count == ", count_params_by_key(param_dict=names_weights_copy, keyword='linear'))

            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            if training_phase is True and self.args.data_aug is not None:
                x_support_set_task = random_flip_like_torchvision(x_support_set_task)

            for num_step in range(num_steps):

                support_loss, support_preds = self.net_forward(x=x_support_set_task,
                                                               y=y_support_set_task,
                                                               weights=names_weights_copy,
                                                               prompted_weights=prompted_weights_copy,
                                                               backup_running_statistics=num_step == 0,
                                                               prepend_prompt=self.args.prompter,
                                                               training=True,
                                                               num_step=num_step,
                                                               training_phase=training_phase,
                                                               epoch=epoch,
                                                               inner_loop=True)

                names_weights_copy, prompted_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                                         names_weights_copy=names_weights_copy,
                                                                                         prompted_weights_copy=prompted_weights_copy,
                                                                                         use_second_order=use_second_order,
                                                                                         current_step_idx=num_step,
                                                                                         current_iter=current_iter,
                                                                                         training_phase=training_phase)

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                 y=y_target_set_task,
                                                                 weights=names_weights_copy,
                                                                 prompted_weights=prompted_weights_copy,
                                                                 backup_running_statistics=False, training=True,
                                                                 prepend_prompt=self.args.prompter,
                                                                 num_step=num_step, training_phase=training_phase,
                                                                 epoch=epoch)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)

                else:
                    if num_step == (self.args.number_of_training_steps_per_iter - 1):

                        if training_phase is True and self.args.data_aug is not None:
                            x_target_set_task = random_flip_like_torchvision(x_target_set_task)

                        target_loss, target_preds = self.net_forward(x=x_target_set_task,
                                                                     y=y_target_set_task,
                                                                     weights=names_weights_copy,
                                                                     prompted_weights=prompted_weights_copy,
                                                                     backup_running_statistics=False, training=True,
                                                                     prepend_prompt=self.args.prompter,
                                                                     num_step=num_step, training_phase=training_phase,
                                                                     epoch=epoch,
                                                                     inner_loop=False)
                        task_losses.append(target_loss)

                        # Gradient conflict 분석을 위한 gradient 수집
                        if training_phase and self.args.ablation_record:

                            task_idx = f"e{epoch}_i{current_iter}_t{task_id}"
                            target_loss.backward(retain_graph=True)

                            all_layer_grads = []   # 전체 layer gradient 저장용

                            for name, param in self.classifier.named_parameters():
                                if param.grad is not None:
                                    if 'prompt' not in name and 'norm_layer' not in name:
                                        grad = param.grad.detach().clone().flatten().cpu()

                                        all_layer_grads.append(grad)

                                        layer_name = name.replace('.', '_')

                                        # Layer 별 폴더 생성
                                        layer_dir = os.path.join(
                                            self.args.experiment_name,
                                            "grad_info_per_epoch",
                                            f"epoch{epoch}",
                                            f"layer_{layer_name}"
                                        )
                                        os.makedirs(layer_dir, exist_ok=True)

                                        save_path = os.path.join(layer_dir, f"{task_idx}.pt")
                                        torch.save(grad, save_path)

                            # 모든 layer를 하나로 저장 (dict 형식)
                            all_dir = os.path.join(
                                self.args.experiment_name,
                                "grad_info_per_epoch",
                                f"epoch{epoch}",
                                "all_layers"
                            )
                            os.makedirs(all_dir, exist_ok=True)
                            all_save_path = os.path.join(all_dir, f"{task_idx}.pt")
                            all_layer_grads = torch.cat(all_layer_grads)
                            torch.save(all_layer_grads, all_save_path)


                            self.classifier.zero_grad()

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

            if not training_phase:
                if torch.cuda.device_count() > 1:
                    self.classifier.module.restore_backup_stats()
                else:
                    self.classifier.restore_backup_stats()

        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        return losses, per_task_target_preds

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step, training_phase, epoch,
                    prompted_weights=None, prepend_prompt=True, inner_loop=True):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """

        preds, feature_map_list = self.classifier.forward(x=x, params=weights, prompted_params=prompted_weights,
                                                          training=training,
                                                          backup_running_statistics=backup_running_statistics,
                                                          num_step=num_step, prepend_prompt=prepend_prompt)
        loss = F.cross_entropy(preds, y)

        return loss, preds

    def net_forward_feature_extractor(self, x, y, weights, backup_running_statistics, training, num_step, training_phase, epoch,
                    prompted_weights=None, prepend_prompt=True, inner_loop=True):


        if inner_loop is False:
            preds, feature_map_list = self.classifier.forward(x=x, params=weights, prompted_params=prompted_weights,
                                                              training=training,
                                                              backup_running_statistics=backup_running_statistics,
                                                              num_step=num_step, prepend_prompt=prepend_prompt)
            loss = F.cross_entropy(input=preds, target=y)
            return loss, preds

        if self.args.data_aug == "mixup":
            preds, feature_map_list = self.classifier.forward(x=x, params=weights, prompted_params=prompted_weights,
                                                              training=training,
                                                              backup_running_statistics=backup_running_statistics,
                                                              num_step=num_step, prepend_prompt=prepend_prompt)
            loss_clean = F.cross_entropy(preds, y)

            x_aug = random_flip_like_torchvision(x)
            x_mix, y_a, y_b, lam = mixup_data(x_aug, y, alpha=5.0)

            preds_mix, aug_feature_map_list = self.classifier.forward(x=x_mix, params=weights,
                                                                      prompted_params=prompted_weights,
                                                                      training=training,
                                                                      backup_running_statistics=backup_running_statistics,
                                                                      num_step=num_step, prepend_prompt=prepend_prompt)

            # loss_aug = F.cross_entropy(aug_preds, y)
            loss_mix = lam * F.cross_entropy(preds_mix, y_a) + (1 - lam) * F.cross_entropy(preds_mix, y_b)

            # 최종 loss: clean + weighted augmented
            gamma = 0.5
            loss = (1 - gamma) * loss_clean + gamma * loss_mix

        elif self.args.data_aug == "only_mixup":
            x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=5.0)
            preds, feature_map_list = self.classifier.forward(x=x_mix, params=weights,
                                                                      prompted_params=prompted_weights,
                                                                      training=training,
                                                                      backup_running_statistics=backup_running_statistics,
                                                                      num_step=num_step, prepend_prompt=prepend_prompt)
            loss = F.cross_entropy(input=preds, target=y)

        # --------- No augmentation ---------
        elif self.args.data_aug == "not_mixup":

            preds, feature_map_list = self.classifier.forward(x=x, params=weights, prompted_params=prompted_weights,
                                                             training=training,
                                                             backup_running_statistics=backup_running_statistics,
                                                             num_step=num_step, prepend_prompt=prepend_prompt)
            loss = F.cross_entropy(preds, y)

        else:
            preds, feature_map_list = self.classifier.forward(x=x, params=weights, prompted_params=prompted_weights,
                                                              training=training,
                                                              backup_running_statistics=backup_running_statistics,
                                                              num_step=num_step, prepend_prompt=prepend_prompt)
            loss = F.cross_entropy(preds, y)

        return loss, preds, feature_map_list

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch, current_iter):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch,
                                                     use_second_order=self.args.second_order and
                                                                      epoch > self.args.first_order_to_second_order_epoch,
                                                     use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                     num_steps=self.args.number_of_training_steps_per_iter,
                                                     training_phase=True,
                                                     current_iter=current_iter)
        return losses, per_task_target_preds

    def evaluation_forward_prop(self, data_batch, epoch, current_iter):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds = self.forward(data_batch=data_batch, epoch=epoch, use_second_order=False,
                                                     use_multi_step_loss_optimization=True,
                                                     num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                     training_phase=False,
                                                     current_iter=current_iter)

        return losses, per_task_target_preds

    def meta_update(self, loss, epoch, current_iter):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()


    def run_train_iter(self, data_batch, epoch, current_iter):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        self.scheduler.step(epoch=epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.train_forward_prop(data_batch=data_batch, epoch=epoch,
                                                                current_iter=current_iter)

        self.meta_update(loss=losses['loss'], epoch=epoch, current_iter=current_iter)
        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()

        return losses, per_task_target_preds

    def run_validation_iter(self, data_batch, current_iter):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch,
                                                                     current_iter=current_iter)

        # losses['loss'].backward() # uncomment if you get the weird memory error
        # self.zero_grad()
        # self.optimizer.zero_grad()

        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return state