import torch
import numpy as np

# from modnets.layers import Binarizer
from avalanche.training.plugins import SupervisedPlugin
from typing import Optional, Any
from copy import deepcopy

INITIALIZE_FROM_ONE = 0.01
INITIALIZE_FROM_ZERO = 0


class BestModelPlugin(SupervisedPlugin):

    def __init__(
        self,
        patience
    ):
        super().__init__()

        self.best_state = None  # Contains the best parameters
        self.best_val = None
        self.best_step: Optional[int] = None
        self.patience = patience

    def before_training(self, strategy, **kwargs):
        self.best_state = None
        self.best_val = None
        self.best_step = None
        self.best_step = None

    def before_training_epoch(self, strategy, **kwargs):
        ub = self._update_best(strategy)
        print("best:", ub, "\n")
        torch.save(strategy.model,
                   '../checkpoint_unlabeled/Roberta_piggyback_temp.pt')
        if ub is None or self.best_step is None:
            return
        curr_step = self._get_strategy_counter(strategy)
        if curr_step - self.best_step >= self.patience:
            strategy.model.load_state_dict(self.best_state)
            strategy.stop_training()

    def after_training(self, strategy: Any, *args, **kwargs):
        ub = self._update_best(strategy)
        print("best:", ub, "\n")
        strategy.model.load_state_dict(self.best_state)

    def _update_best(self, strategy):
        res = strategy.evaluator.get_last_metrics()
        names = [k for k in res.keys() if k.startswith(
            "Loss_Exp/eval_phase/val_stream")]
        if len(names) == 0:
            return None

        task_label = strategy.experience.task_label
        full_name = 'Loss_Exp/eval_phase/val_stream/Task%s/Exp%s' % (
            str(task_label).zfill(3), str(task_label).zfill(3))
        print("full_name:", full_name, "\n")
        val_loss = res.get(full_name)
        if self.best_val is None or val_loss <= self.best_val:
            self.best_state = deepcopy(strategy.model.state_dict())
            if self.best_val is None:
                self.best_val = val_loss
                self.best_step = 0
                return None

            if val_loss <= self.best_val:
                self.best_step = self._get_strategy_counter(strategy)
                self.best_val = val_loss

        return self.best_val

    def _get_strategy_counter(self, strategy):
        return strategy.clock.train_exp_epochs


class MonitoringMaskPlugin(SupervisedPlugin):
    def __init__(
        self
    ):
        super().__init__()

        self.masks_prev = {}
        self.masks = {}
        self.mask_monitor = {}
        self.file = 'monitoring_masks.txt'

        with open(file=self.file, mode='w') as f:
            f.write('')

    def before_training_epoch(self, strategy, **kwargs):
        for module_idx, (n, module) in enumerate(strategy.model.named_modules()):
            if 'ElementWise' in str(type(module)):
                self.masks_prev[module_idx] = module.mask_thresholded
                if self.masks == {}:
                    with open(file=self.file, mode='a') as f:
                        f.write(n + ' ' + str(module_idx) + ' 1: ' + str(torch.sum(module.mask_thresholded).item() /
                                (module.weight.shape[0] * module.weight.shape[1])) + '\n')
                    self.mask_monitor[module_idx] = [
                        module.masks['0'].data.clone().detach()]

        with open(file=self.file, mode='a') as f:
            f.write('\n')

    def after_training_epoch(self, strategy, **kwargs):
        for module_idx, module in enumerate(strategy.model.modules()):
            if 'ElementWise' in str(type(module)):
                self.masks[module_idx] = module.mask_thresholded
                zero_zero = (
                    1 - self.masks_prev[module_idx]) * (1 - self.masks[module_idx])
                zero_one = (
                    1 - self.masks_prev[module_idx]) * self.masks[module_idx]
                one_zero = self.masks_prev[module_idx] * \
                    (1 - self.masks[module_idx])
                one_one = self.masks_prev[module_idx] * \
                    self.masks[module_idx]
                weight_size = module.weight.shape[0] * module.weight.shape[1]

                with open(file=self.file, mode='a') as f:
                    f.write(str(module_idx) + ' 1: ' + str(torch.sum(module.mask_thresholded).item() / weight_size) + '\t0->0: ' + str(torch.sum(zero_zero).item() / weight_size) +
                            '\t0->1: ' + str(torch.sum(zero_one).item() /
                                             weight_size) +
                            '\t1->0: ' + str(torch.sum(one_zero).item() /
                                             weight_size) +
                            '\t1->1: ' + str(torch.sum(one_one).item() / weight_size) + '\n')

                self.mask_monitor[module_idx].append(
                    module.masks['0'].data.clone().detach())

        torch.save(self.mask_monitor, './masks_monitor.pt')

        with open(file=self.file, mode='a') as f:
            f.write('\n')

    def after_training(self, strategy: Any, *args, **kwargs):
        torch.save(self.mask_monitor, './masks_monitor.pt')


def copy_weights(model, model_pretrained, data_idx=0):
    # Copy weights of pretrained model
    module_list = list(model_pretrained.modules())
    i = 0
    if "Roberta" in str(type(model)):
        for module in model.roberta.modules():
            if 'Pooler' in str(type(module_list[i])):
                break

            if 'Linear' in str(type(module)):
                while True:
                    i += 1
                    if 'Linear' in str(type(module_list[i])):
                        break
                print(str(type(module_list[i])))
                module.weight.data.copy_(module_list[i].weight.data)
                module.bias.data.copy_(module_list[i].bias.data)

            elif 'Embedding' in str(type(module)) and 'Roberta' not in str(type(module)):
                while True:
                    i += 1
                    if 'Embedding' in str(type(module_list[i])) and 'Roberta' not in str(type(module_list[i])):
                        break
                print(str(type(module_list[i])))
                module.weight.data.copy_(module_list[i].weight.data)

            elif 'LayerNorm' in str(type(module)):
                while True:
                    i += 1
                    if 'LayerNorm' in str(type(module_list[i])):
                        break
                print(str(type(module_list[i])))
                module.weight.data.copy_(module_list[i].weight.data)
                module.bias.data.copy_(module_list[i].bias.data)
    elif "T5" in str(type(model)):
        for module in model.modules():
            if 'Linear' in str(type(module)):
                while True:
                    i += 1
                    if 'Linear' in str(type(module_list[i])):
                        break
                print(str(type(module_list[i])), module_list[i])
                module.weight.data.copy_(module_list[i].weight.data)

            elif 'Embedding' in str(type(module)) and 'Roberta' not in str(type(module)):
                while True:
                    i += 1
                    if 'Embedding' in str(type(module_list[i])) and 'Roberta' not in str(type(module_list[i])):
                        break
                print(str(type(module_list[i])), module_list[i])
                module.weight.data.copy_(module_list[i].weight.data)

            elif 'LayerNorm' in str(type(module)):
                while True:
                    i += 1
                    if 'LayerNorm' in str(type(module_list[i])):
                        break
                print(str(type(module_list[i])), module_list[i])
                module.weight.data.copy_(module_list[i].weight.data)


def check(model, pretrained):
    """Makes sure that the trained model weights match those of the pretrained model."""
    print('Making sure filter weights have not changed.')
    module_list = list(pretrained.modules())
    i = 0
    threshold = 1e-8
    if model.roberta.embeddings.word_embeddings.weight.dtype == torch.float16:
        threshold = 6e-8

    for module in model.modules():
        if i == len(module_list):
            break

        if 'Linear' in str(type(module)):
            weight = module.weight.data.cpu()
            weight_pretrained = module_list[i].weight.data.cpu()
            # Using small threshold of 1e-8 for any floating point inconsistencies.
            # Note that threshold per element is even smaller as the 1e-8 threshold
            # is for sum of absolute differences.
            assert (weight - weight_pretrained).abs().sum() < threshold, \
                'module %s failed check' % (module)
            if module.bias is not None:
                bias = module.bias.data.cpu()
                bias_pretrained = module_list[i].bias.data.cpu()
                assert (bias - bias_pretrained).abs().sum() < threshold

        elif 'Embedding' in str(type(module)) and 'Roberta' not in str(type(module)):
            weight = module.weight.data.cpu()
            weight_pretrained = module_list[i].weight.data.cpu()
            assert (weight - weight_pretrained).abs().sum() < threshold, \
                'module %s failed check' % (module)

        elif 'LayerNorm' in str(type(module)):
            weight = module.weight.data.cpu()
            weight_pretrained = module_list[i].weight.data.cpu()
            # Using small threshold of 1e-8 for any floating point inconsistencies.
            # Note that threshold per element is even smaller as the 1e-8 threshold
            # is for sum of absolute differences.
            assert (weight - weight_pretrained).abs().sum() < threshold, \
                'module %s failed check' % (module)
            if module.bias is not None:
                bias = module.bias.data.cpu()
                bias_pretrained = module_list[i].bias.data.cpu()
                assert (bias - bias_pretrained).abs().sum() < threshold

        elif ('Identity' in str(type(module)) and 'Identity' not in str(type(module_list[i]))) or ('Dict' in str(type(module)) and 'Dict' not in str(type(module_list[i]))) or \
                'RobertaForMaskedLM' in str(type(module)) or 'RobertaForSequenceClassification' in str(type(module)) or 'RobertaForLoRAEndtask' in str(type(module)):
            continue
        i += 1
    print('Passed checks...')


def check_endtask(model, pretrained, data, data_idx, dataset2masks):
    """Makes sure that the trained model weights match those of the pretrained model."""
    print('Making sure mask weights have not changed.')
    module_list = list(pretrained.modules())
    i = 0

    for module_idx, module in enumerate(model.modules()):
        if i == len(module_list):
            break

        if 'Linear' in str(type(module)):
            weight = module.weight.data.cpu()
            weight_pretrained = module_list[i].weight.data.cpu()
            mask = module.masks[str(data_idx)].data.cpu()
            mask_pretrained = dataset2masks[data][module_idx]
            mask_pretrained = np.unpackbits(
                mask_pretrained, axis=0, count=mask.shape[0])
            mask_pretrained = torch.Tensor(mask_pretrained).to('cpu')
            # Using small threshold of 1e-8 for any floating point inconsistencies.
            # Note that threshold per element is even smaller as the 1e-8 threshold
            # is for sum of absolute differences.
            assert (weight - weight_pretrained).abs().sum() > 1e-8, \
                'module %s failed check from weight' % (module)
            assert (mask - mask_pretrained).abs().sum() < 1e-8, \
                'module %s failed check from mask' % (module)
            if module.bias is not None:
                bias = module.bias.data.cpu()
                bias_pretrained = module_list[i].bias.data.cpu()
                assert (bias - bias_pretrained).abs().sum() > 1e-8

        elif 'Embedding' in str(type(module)) and 'Roberta' not in str(type(module)):
            weight = module.weight.data.cpu()
            weight_pretrained = module_list[i].weight.data.cpu()
            assert (weight - weight_pretrained).abs().sum() > 1e-8, \
                'module %s failed check' % (module)

        elif 'LayerNorm' in str(type(module)):
            weight = module.weight.data.cpu()
            weight_pretrained = module_list[i].weight.data.cpu()
            # Using small threshold of 1e-8 for any floating point inconsistencies.
            # Note that threshold per element is even smaller as the 1e-8 threshold
            # is for sum of absolute differences.
            assert (weight - weight_pretrained).abs().sum() > 1e-8, \
                'module %s failed check' % (module)
            if module.bias is not None:
                bias = module.bias.data.cpu()
                bias_pretrained = module_list[i].bias.data.cpu()
                assert (bias - bias_pretrained).abs().sum() > 1e-8

        elif ('Identity' in str(type(module)) and 'Identity' not in str(type(module_list[i]))) or ('Dict' in str(type(module)) and 'Dict' not in str(type(module_list[i]))) or \
                'RobertaForMaskedLM' in str(type(module)) or 'RobertaForSequenceClassification' in str(type(module)) or 'RobertaForLoRAEndtask' in str(type(module)):
            continue
        i += 1
    print('Passed checks...')


# def ckpt_masks(model, dat, train_ln):
#     dataset2masks = {}
#     dataset2ln = {}
#     for data_idx, data in enumerate(dat):
#         masks = {}
#         ln = {}
#         for module_idx, module in enumerate(model.modules()):
#             if 'ElementWise' in str(type(module)):
#                 mask = Binarizer.apply(module.masks[str(data_idx)])
#                 mask = mask.data.detach().cpu()

#                 num_zero = mask.eq(0).sum()
#                 num_one = mask.eq(1).sum()
#                 total = mask.numel()
#                 print(data_idx, module_idx, num_zero / total * 100)

#                 assert num_zero + num_one == total
#                 mask = mask.type(torch.ByteTensor)
#                 masks[module_idx] = np.packbits(mask.numpy(), axis=0)
#             elif 'LayerNorm' in str(type(module)):
#                 if train_ln:
#                     ln[module_idx] = (module.weight,)
#                     if module.bias is not None:
#                         ln[module_idx] += module.bias

#         dataset2masks[data] = masks
#         dataset2ln[data] = ln

#     return dataset2masks, dataset2ln


def load_ckpt(dataset2masks, model, data_idx, data, args):
    if args.train_str == 'mask' or args.train_str == 'all':
        mean_one = dataset2masks[data]['mean_one']
        mean_zero = dataset2masks[data]['mean_zero']

    for module_idx, module in enumerate(model.modules()):
        if 'ElementWise' in str(type(module)):
            if args.train_str == 'weight':
                mask = dataset2masks[data][module_idx]
                mask = np.unpackbits(
                    mask, axis=0, count=module.masks[str(data_idx)].data.shape[0])
                mask = torch.Tensor(mask).to('cuda')

                num_zero = mask.eq(0).sum()
                num_one = mask.eq(1).sum()
                total = mask.numel()
                print(data_idx, module_idx, num_zero / total * 100)
                module.masks[str(data_idx)].data.copy_(mask)
                module.masks[str(data_idx)].requires_grad = False

            elif args.train_str == 'mask' or args.train_str == 'all':
                mask = dataset2masks[data]['masks'][module_idx]
                mask = np.unpackbits(
                    mask, axis=0, count=module.masks[str(data_idx)].data.shape[0])
                mask = torch.Tensor(mask).to('cuda')

                num_zero = mask.eq(0).sum()
                num_one = mask.eq(1).sum()
                total = mask.numel()
                print(data_idx, module_idx, num_zero / total * 100)
                if args.mask_init == 'mean':
                    mask_initialized = (mask * mean_one) + \
                        ((1 - mask) * mean_zero).to(torch.float32)
                elif args.mask_init == 'exp':
                    dist_zero = torch.distributions.Exponential(
                        -1 / (mean_zero - 5e-3))
                    dist_one = torch.distributions.Exponential(
                        1 / (mean_one - 5e-3))

                    mask_initialized = torch.zeros_like(mask)
                    mask_initialized[mask.eq(0)] = dist_zero.sample(
                        (num_zero,)).to(torch.float32).to('cuda') * -1 + 5e-3
                    mask_initialized[mask.eq(1)] = dist_one.sample(
                        (num_one,)).to(torch.float32).to('cuda') + 5e-3
                else:
                    init = args.mask_init.split(',')
                    mask_initialized = ((mask * float(init[0])) +
                                        (1 - mask) * float(init[1])).to(torch.float32)
                module.masks[str(data_idx)].data.copy_(mask_initialized)
                module.masks[str(data_idx)].requires_grad = True

            elif args.train_str == 'no_thresholding':
                mask = dataset2masks[data][module_idx]['mask']
                mean_one = dataset2masks[data][module_idx]['mean_one']
                mean_zero = dataset2masks[data][module_idx]['mean_zero']
                mask = np.unpackbits(
                    mask, axis=0, count=module.masks[str(data_idx)].data.shape[0])
                mask = torch.Tensor(mask).to('cuda')
                mask_initialized = ((mask * mean_one) +
                                    (1 - mask) * mean_zero).to(torch.float32)

                if data_idx == 0:
                    module.weight.data.mul_(mask_initialized.to('cpu'))
                else:
                    module.weight.data.mul_(mask_initialized.to('cuda'))

                module.masks[str(data_idx)].requires_grad = False


# def load_ckpt_initialize(model, data_idx_from, data_idx_to):
#     for module_idx, module in enumerate(model.modules()):
#         if 'ElementWise' in str(type(module)):
#             mask = module.masks[str(data_idx_from)]
#             thresholded_mask = Binarizer.apply(mask)

#             mask_initialized = (thresholded_mask * INITIALIZE_FROM_ONE) + \
#                 ((1 - thresholded_mask) * INITIALIZE_FROM_ZERO).to(torch.float32)

#             num_zero = thresholded_mask.eq(0).sum()
#             num_one = thresholded_mask.eq(1).sum()
#             total = thresholded_mask.numel()
#             print(num_zero / total * 100)

#             module.masks[str(data_idx_to)].data.copy_(mask_initialized)
#             module.masks[str(data_idx_to)].requires_grad = True
