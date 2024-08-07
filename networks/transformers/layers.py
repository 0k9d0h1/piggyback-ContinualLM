import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from typing import Optional

DEFAULT_THRESHOLD = 5e-3


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs, threshold, le):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = le
        outputs[inputs.gt(threshold)] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput, None, None


class PretrainingMultiTaskClassifier(nn.Module):

    def __init__(self, in_features, initial_out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.bias_ = bias
        self.initial_out_features = initial_out_features
        self.classifiers = nn.ModuleDict({'0': nn.Linear(
            in_features=in_features, out_features=initial_out_features, bias=bias)})

    def adaptation(self, num_class, task_label):
        if str(task_label) not in self.classifiers:
            self.classifiers[str(task_label)] = nn.Linear(
                in_features=self.in_features, out_features=self.initial_out_features, bias=self.bias_)

    def forward(self, x: Tensor, task_label: int) -> Tensor:
        return self.classifiers[str(task_label)](x.to(dtype=torch.float32))


class MultiTaskClassifier(nn.Module):

    def __init__(self, in_features, initial_out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.bias = bias
        self.classifiers = nn.ModuleDict({'0': nn.Linear(
            in_features=in_features, out_features=initial_out_features, bias=bias)})

    def adaptation(self, num_class, task_label):
        if str(task_label) not in self.classifiers:
            self.classifiers[str(task_label)] = nn.Linear(
                in_features=self.in_features, out_features=num_class, bias=self.bias)

    def forward(self, x: Tensor, task_label: int) -> Tensor:
        return self.classifiers[str(task_label)](x)


class ElementWiseLinear(nn.Module):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, zero_out=True, bias=True,
                 mask_init='1s',
                 threshold_fn='binarizer', threshold=None, config=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = config.mask_scale
        self.mask_init = mask_init
        self.zero_out = zero_out
        self.config = config

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        self.weight = Parameter(torch.Tensor(
            out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.masks = nn.ParameterDict({'0': self.make_mask()})

    def adaptation(self, num_class, task_label):
        if str(task_label) not in self.masks:
            self.masks[str(task_label)] = self.make_mask()

    def forward(self, x: Tensor, task_label: int) -> Tensor:
        # Get binarized/ternarized mask from real-valued mask.
        if self.zero_out:
            weight_thresholded = self.get_weight(task_label)
            # Get output using modified weight.
            return F.linear(x, weight_thresholded, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

    def make_mask(self):
        # Initialize real-valued mask weights.
        mask_real = self.weight.data.new(self.weight.size())
        if self.mask_init == '1s':
            mask_real.fill_(self.mask_scale)
        elif self.mask_init == 'uniform':
            mask_real.uniform_(-1 * self.mask_scale, self.mask_scale)
        # mask_real is now a trainable parameter.

        return Parameter(mask_real)

    def get_weight(self, task_label):
        # For multi-head attention module
        if self.config.baseline == 'piggyback':
            self.mask_thresholded = Binarizer.apply(
                self.masks[str(task_label)], 5e-3, 0)
        elif self.config.baseline == 'piggyback_nonzero':
            self.mask_thresholded = Binarizer.apply(
                self.masks[str(task_label)], 0, -1)

        weight_thresholded = self.mask_thresholded * self.weight

        return weight_thresholded

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)


class LoRALinear(nn.Module):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        config=None,
        bias=True,
        **kwargs
    ):
        super().__init__()
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        self.weight = Parameter(torch.Tensor(
            out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(
                out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self.fan_in_fan_out = fan_in_fan_out
        self.r = r
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # Actual trainable parameters
        if self.r > 0:
            self.lora_As = nn.ParameterDict(
                {'0': nn.Parameter(self.weight.new_zeros((r, in_features)))})
            self.lora_Bs = nn.ParameterDict({'0': nn.Parameter(
                self.weight.new_zeros((out_features, r)))})
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
        else:
            self.lora_As = {}
            self.lora_Bs = {}
        self.init_lora('0')
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self, task_label):
        nn.Linear.reset_parameters(self)
        self.init_lora(task_label)

    def init_lora(self, task_label):
        if hasattr(self, 'lora_As'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_As[task_label], a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bs[task_label])

    # def train(self, mode: bool = True):
    #     def T(w):
    #         return w.transpose(0, 1) if self.fan_in_fan_out else w
    #     nn.Linear.train(self, mode)

    #     for task_label in self.lora_As:
    #         if mode:
    #             if self.merge_weights and self.merged:
    #                 # Make sure that the weights are not merged
    #                 if self.r > 0:
    #                     self.weight.data -= T(self.lora_Bs[task_label] @
    #                                           self.lora_As[task_label]) * self.scaling
    #                 self.merged = False
    #         else:
    #             if self.merge_weights and not self.merged:
    #                 # Merge the weights and mark it
    #                 if self.r > 0:
    #                     self.weight.data += T(self.lora_Bs[task_label] @
    #                                           self.lora_As[task_label]) * self.scaling
    #                 self.merged = True

    def adaptation(self, num_class, task_label):
        if str(task_label) not in self.lora_As:
            self.lora_As[str(task_label)] = nn.Parameter(
                self.weight.new_zeros((self.r, self.in_features)))
            self.lora_Bs[str(task_label)] = nn.Parameter(
                self.weight.new_zeros((self.out_features, self.r)))
            self.init_lora(str(task_label))


class LoRAPiggybackLinear(LoRALinear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0,
                 fan_in_fan_out: bool = False,
                 merge_weights: bool = True,
                 config=None,
                 bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None, **kwargs):
        super().__init__(in_features, out_features, r, lora_alpha,
                         lora_dropout, fan_in_fan_out, merge_weights, config, bias, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        if self.config.training_type == 'finetune' and self.config.finetune_type == 'lora_piggyback':
            # self.masks_A = nn.ParameterDict({'0': self.make_mask('A')})
            # self.masks_B = nn.ParameterDict({'0': self.make_mask('B')})
            self.masks = nn.ParameterDict({'0': self.make_mask('weight')})

        if self.config.training_type == 'finetune' and self.config.finetune_type == 'merge':
            self.lora_As_stacked = self.weight.data.new(
                self.lora_As['0'].size()).unsqueeze(0)
            self.lora_Bs_stacked = self.weight.data.new(
                self.lora_Bs['0'].size()).unsqueeze(0)
            self.mergeratios = self.weight.data.new(1)
            self.lora_As_stacked.fill_(0)
            self.lora_Bs_stacked.fill_(0)
            self.mergeratios.fill_(0)

    def adaptation(self, num_class, task_label):
        super().adaptation(num_class, task_label)
        if self.config.training_type == 'finetune' and self.config.finetune_type == 'lora_piggyback' and str(task_label) not in self.masks:
            # self.masks_A[str(task_label)] = self.make_mask('A')
            # self.masks_B[str(task_label)] = self.make_mask('B')
            self.masks[str(task_label)] = self.make_mask('weight')

    def forward(self, x: Tensor, task_label: int) -> Tensor:
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0:
            # result = F.linear(x, T(self.weight), bias=self.bias)
            if self.config.training_type == 'posttrain':
                result = F.linear(x, T(self.weight), bias=self.bias)
                result += (self.lora_dropout(x) @ self.lora_As[str(task_label)].transpose(0, 1)
                           @ self.lora_Bs[str(task_label)].transpose(0, 1)) * self.scaling
            elif self.config.training_type == 'finetune':
                if self.config.finetune_type == 'lora_piggyback':
                    # thresholded_mask_A = Binarizer.apply(
                    #     self.masks_A[str(task_label)], 5e-3, 0)
                    # thresholded_mask_B = Binarizer.apply(
                    #     self.masks_B[str(task_label)], 5e-3, 0)
                    self.lora_weight = self.weight + \
                        (self.lora_Bs[str(task_label)] @
                         self.lora_As[str(task_label)]) * self.scaling
                    thresholded_mask = Binarizer.apply(
                        self.masks[str(task_label)], 5e-3, 0)
                    result = F.linear(
                        x, T(self.lora_weight * thresholded_mask), bias=self.bias)
                    # result += (self.lora_dropout(x) @ (self.lora_As[str(task_label)] * thresholded_mask_A).transpose(0, 1)
                    #            @ (self.lora_Bs[str(task_label)] * thresholded_mask_B).transpose(0, 1)) * self.scaling
                elif self.config.finetune_type == 'full_finetune':
                    if self.r > 0 and not self.merged:
                        self.weight.data += T(self.lora_Bs[str(task_label)] @
                                              self.lora_As[str(task_label)]) * self.scaling
                        self.merged = True
                    result = F.linear(x, T(self.weight), bias=self.bias)
                elif self.config.finetune_type == 'lora':
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    result = result + (self.lora_dropout(x) @ self.lora_As[str(task_label)].transpose(0, 1)
                                       @ self.lora_Bs[str(task_label)].transpose(0, 1)) * self.scaling
                elif self.config.finetune_type == 'merge':
                    result = F.linear(x, T(self.weight), bias=self.bias)
                    merged_A = torch.einsum(
                        'i, ijk -> jk', self.mergeratios, self.lora_As_stacked)
                    merged_B = torch.einsum(
                        'i, ijk -> jk', self.mergeratios, self.lora_Bs_stacked)
                    result = result + (self.lora_dropout(x) @
                                       merged_A.t() @ merged_B.t()) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

    # def train(self, mode: bool = True):
    #     def T(w):
    #         return w.transpose(0, 1) if self.fan_in_fan_out else w
        # nn.Linear.train(self, mode)

    #     for task_label in self.lora_As:
    #         if mode:
    #             if self.merge_weights and self.merged:
    #                 # Make sure that the weights are not merged
    #                 if self.r > 0:
    #                     if self.training_type == 'posttrain':
    #                         self.weight.data -= T(self.lora_Bs[task_label] @
    #                                               self.lora_As[task_label]) * self.scaling
    #                     elif self.training_type == 'finetune':
    #                         # thresholded_mask_A = Binarizer.apply(
    #                         #     self.masks_A[str(task_label)], 5e-3, 0)
    #                         # thresholded_mask_B = Binarizer.apply(
    #                         #     self.masks_B[str(task_label)], 5e-3, 0)
    #                         # self.weight.data -= T((thresholded_mask_B * self.lora_Bs[task_label]) @ (
    #                         #     thresholded_mask_A * self.lora_As[task_label])) * self.scaling
    #                         thresholded_mask = Binarizer.apply(
    #                             self.masks[str(task_label)], 5e-3, 0)
    #                         self.weight.data -= T(self.lora_Bs[task_label] @
    #                                               self.lora_As[task_label]) * thresholded_mask * self.scaling
    #                 self.merged = False
    #         else:
    #             if self.merge_weights and not self.merged:
    #                 # Merge the weights and mark it
    #                 if self.r > 0:
    #                     if self.training_type == 'posttrain':
    #                         self.weight.data += T(self.lora_Bs[task_label] @
    #                                               self.lora_As[task_label]) * self.scaling
    #                     elif self.training_type == 'finetune':
    #                         # thresholded_mask_A = Binarizer.apply(
    #                         #     self.masks_A[str(task_label)], 5e-3, 0)
    #                         # thresholded_mask_B = Binarizer.apply(
    #                         #     self.masks_B[str(task_label)], 5e-3, 0)
    #                         # self.weight.data += T((thresholded_mask_B * self.lora_Bs[task_label]) @ (
    #                         #     thresholded_mask_A * self.lora_As[task_label])) * self.scaling
    #                         thresholded_mask = Binarizer.apply(
    #                             self.masks[str(task_label)], 5e-3, 0)
    #                         print(thresholded_mask)
    #                         self.weight.data += T(self.lora_Bs[task_label] @
    #                                               self.lora_As[task_label]) * thresholded_mask * self.scaling
    #                 self.merged = True

    def make_mask(self, lora):
        # Initialize real-valued mask weights.
        if lora == 'A':
            mask_real = self.weight.data.new(self.lora_As['0'].size())
        elif lora == 'B':
            mask_real = self.weight.data.new(self.lora_Bs['0'].size())
        elif lora == 'weight':
            mask_real = self.weight.data.new(self.weight.size())

        if self.mask_init == '1s':
            mask_real.fill_(self.mask_scale)
        elif self.mask_init == 'uniform':
            mask_real.uniform_(-1 * self.mask_scale, self.mask_scale)
        # mask_real is now a trainable parameter.

        return Parameter(mask_real)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        if getattr(self.bias, 'data', None) is not None:
            self.bias.data = fn(self.bias.data)
