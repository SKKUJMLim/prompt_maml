import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from meta_neural_network_architectures import VGGReLUNormNetwork, ResNet12
from inner_loop_optimizers import GradientDescentLearningRule, LSLRGradientDescentLearningRule

from utils.storage import save_statistics

import arbiter
from utils.basic import kl_divergence_pixelwise, LabelSmoothingCrossEntropy
from utils.contrastive_loss import soft_nearest_neighbors_loss_euclidean


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


        names_weights_copy = {key: value for key, value in names_weights_copy.items() if 'layer_dict' in key}

        if self.args.learnable_per_layer_per_step_inner_loop_learning_rate:
            self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                        args=self.args,
                                                                        init_learning_rate=self.task_learning_rate,
                                                                        total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                        use_learnable_learning_rates=True)
            self.inner_loop_optimizer.initialise(names_weights_dict=names_weights_copy)

            self.task_embedding_adaptive_learning_rate = nn.Parameter(
                data=torch.ones(self.args.number_of_training_steps_per_iter + 1) * self.args.text_embedding_learning_rate,
                requires_grad=True)
        else:
            self.inner_loop_optimizer = GradientDescentLearningRule(device=device, args=self.args,
                                                                    learning_rate=self.task_learning_rate)


        self.arbiter = arbiter.TaskAwareAttention(image_channels=3, task_dim=self.args.num_text_embedding_params, embed_dim=100)

        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)

        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

        self.optimizer = optim.Adam(self.trainable_parameters(), lr=args.meta_learning_rate, amsgrad=False)
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

    def apply_inner_loop_update(self, loss, names_weights_copy, use_second_order, current_step_idx, current_iter, training_phase):
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
        else:
            self.classifier.zero_grad(params=names_weights_copy)


        grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                    create_graph=use_second_order, allow_unused=True)
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

        names_weights_copy, prompted_weights_copy = self.inner_loop_optimizer.update_params(
            names_weights_dict=names_weights_copy,
            names_grads_wrt_params_dict=names_grads_copy,
            num_step=current_step_idx,
            current_iter=current_iter,
            training_phase=training_phase)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}

        return names_weights_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

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
        total_accuracies = []
        total_support_accuracies = [[] for i in range(num_steps)]
        total_target_accuracies = [[] for i in range(num_steps)]
        per_task_target_preds = [[] for i in range(len(x_target_set))]

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


            names_weights_copy = {key: value for key, value in names_weights_copy.items() if 'layer_dict' in key}

            # print("names_weights_copy === ", names_weights_copy.keys())
            # print("prompted_weight_dict === ", prompted_weights_copy.keys())

            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            # z = nn.Parameter(torch.randn([1, self.args.num_text_embedding_params]), requires_grad=True).to(self.device)
            z = nn.Parameter(torch.ones([1, self.args.num_text_embedding_params]), requires_grad=True).to(self.device)
            # z = torch.zeros(size=[1, self.args.num_text_embedding_params], requires_grad=True).to(self.device)

            for num_step in range(num_steps):

                ideal_prompt = self.arbiter(x_support_set_task, z)
                x_support_set_task = x_support_set_task + ideal_prompt

                # Add prompt
                support_loss, support_preds, support_feature_list = self.net_forward(x=x_support_set_task,
                                                                                     y=y_support_set_task,
                                                                                     weights=names_weights_copy,
                                                                                     backup_running_statistics=num_step == 0,
                                                                                     training=True,
                                                                                     num_step=num_step,
                                                                                     training_phase=training_phase,
                                                                                     epoch=epoch)

                gradients = torch.autograd.grad(support_loss, (*names_weights_copy.values(), z), create_graph=use_second_order, retain_graph=True)

                grads, context_grads = gradients[:-1], gradients[-1]

                if self.args.learnable_per_layer_per_step_inner_loop_learning_rate:
                    z = z - self.task_embedding_adaptive_learning_rate[num_step] * context_grads
                else:
                    z = z - self.args.text_embedding_learning_rate * context_grads

                names_weights_copy = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step,
                                                                  current_iter=current_iter,
                                                                  training_phase=training_phase)

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds, _ = self.net_forward(x=x_target_set_task,
                                                                    y=y_target_set_task,
                                                                    weights=names_weights_copy,
                                                                    backup_running_statistics=False, training=True,
                                                                    num_step=num_step, training_phase=training_phase,
                                                                    epoch=epoch)

                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)

                else:
                    if num_step == (self.args.number_of_training_steps_per_iter - 1):

                        ideal_prompt = self.arbiter(x_target_set_task, z)
                        x_target_set_task = x_target_set_task + ideal_prompt

                        target_loss, target_preds, target_feature_list = self.net_forward(x=x_target_set_task,
                                                                        y=y_target_set_task,
                                                                        weights=names_weights_copy,
                                                                        backup_running_statistics=False, training=True,
                                                                        num_step=num_step,
                                                                        training_phase=training_phase,
                                                                        epoch=epoch)

                        task_losses.append(target_loss)

            per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

            if current_iter == 'test':
                information={}
                information['phase'] = current_iter
                information['task_idx'] = task_id
                information['accuracy'] = accuracy.mean().item()
                # .mean()은 True 값의 비율을 반환. 이는 정확도를 의미

                if os.path.exists(self.args.experiment_name + '/' + self.args.experiment_name + "_per_task_acc.csv"):
                    self.csv_exist = False

                if self.csv_exist:
                    save_statistics(experiment_name=self.args.experiment_name,
                                    line_to_add=list(information.keys()),
                                    filename=self.args.experiment_name + "_per_task_acc.csv", create=True)
                    self.csv_exist = False
                    save_statistics(experiment_name=self.args.experiment_name,
                                    line_to_add=list(information.values()),
                                    filename=self.args.experiment_name + "_per_task_acc.csv", create=False)
                else:
                    save_statistics(experiment_name=self.args.experiment_name,
                                    line_to_add=list(information.values()),
                                    filename=self.args.experiment_name + "_per_task_acc.csv", create=False)

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
                    prompted_weights=None, prepend_prompt=True):
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
                                                     num_step=num_step)

        loss = F.cross_entropy(input=preds, target=y)

        # criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        # loss = criterion(preds, y)

        # loss_seperate = F.cross_entropy(input=preds, target=y, reduction='none')

        # Not add prompt
        # preds_, feature_map_list_ = self.classifier.forward(x=x, params=weights, prompted_params=prompted_weights,
        #                                                   training=training,
        #                                                   backup_running_statistics=backup_running_statistics,
        #                                                   num_step=num_step, prepend_prompt=False)

        # print("num_step == ", num_step)
        # batch_correct_prompt = (torch.argmax(preds, dim=1) == y)   # Add Prompt로 올바르게 예측한 샘플 여부
        # batch_incorrect_prompt = (torch.argmax(preds, dim=1) != y) # Add Prompt로 올바르게 예측하지 못한 샘플 여부
        #
        # batch_correct = (torch.argmax(preds_, dim=1) == y)  # 올바르게 예측한 샘플 여부
        # batch_incorrect = (torch.argmax(preds_, dim=1) != y)  # 올바르게 예측하지 못한 샘플 여부
        #
        # print(f"정답 샘플 인덱스: {torch.nonzero(batch_correct_prompt).squeeze().tolist()}")
        # print(f"오답 샘플 인덱스: {torch.nonzero(batch_incorrect_prompt).squeeze().tolist()}")
        #
        # # Visual Prompt를 추가하거나 추가하지 않아도 맞춘 경우
        # always_correct_samples = batch_correct_prompt & batch_correct
        # always_correct_indices = torch.nonzero(always_correct_samples).squeeze() # 해당 샘플들의 인덱스 찾기
        # print(f"Prompt 추가 여부와 관계없이 맞춘 샘플 인덱스: {always_correct_indices.tolist()}")
        #
        # # Visual Prompt 추가 시 올바르게 예측했지만, 추가하지 않았을 때 틀린 샘플 찾기
        # improved_samples = batch_correct_prompt & batch_incorrect       # Prompt 덕분에 올바르게 예측한 샘플
        # improved_indices = torch.nonzero(improved_samples).squeeze()    # 해당 샘플들의 인덱스 찾기
        # print(f"Visual Prompt 덕분에 올바르게 예측한 샘플 인덱스: {improved_indices.tolist()}")
        #
        # # Visual Prompt 없이 맞췄지만, Prompt 추가 후 틀린 샘플 찾기
        # worse_samples = batch_correct & batch_incorrect_prompt  # Prompt 추가 후 오히려 틀린 샘플
        # worse_indices = torch.nonzero(worse_samples).squeeze()  # 해당 샘플 인덱스
        # print(f"Visual Prompt 추가 후 오히려 틀린 샘플 인덱스: {worse_indices.tolist()}")
        # print("====" * 50)

        # print("batch_correct")
        # for i in range(len(batch_correct)):
        #     if batch_correct[i]:
        #         print(torch.norm(feature_map_list[3][i] - feature_map_list_[3][i].detach().clone(), p=2))
        #         # loss = loss + kl_divergence_pixelwise(feature_map_list[3][i], feature_map_list_[3][i].detach().clone())  # detach?
        #
        # print("batch_incorrect")
        # for i in range(len(batch_incorrect)):
        #     if batch_incorrect[i]:
        #         print(torch.norm(feature_map_list[3][i] - feature_map_list_[3][i].detach().clone(), p=2))
        #         # loss = loss + kl_divergence_pixelwise(feature_map_list[3][i], feature_map_list_[3][i].detach().clone())  # detach?
        # print("===="*50)

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

    def meta_update(self, loss):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """

        # 가중치 업데이트 확인용 변수
        # prev_weights = {}
        # for name, param in self.named_parameters():
        #     prev_weights[name] = param.data.clone()

        self.optimizer.zero_grad()
        loss.backward()
        # if 'imagenet' in self.args.dataset_name:
        #    for name, param in self.classifier.named_parameters():
        #        if param.requires_grad:
        #            param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
        # for name, param in self.classifier.named_parameters():
        #    print(param.mean())

        self.optimizer.step()

        # 가중치 업데이트 확인
        # for name, param in self.named_parameters():
        #     if not torch.equal(prev_weights[name], param.data):
        #         print(f"{name} 가중치가 업데이트되었습니다.")
        #         prev_weights[name] = param.data.clone()

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

        self.meta_update(loss=losses['loss'])
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