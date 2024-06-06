import torch
import math
import torch.nn as nn
import avalanche.models as am
import torch.nn.functional as F

from torch.autograd import Variable
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from typing import Optional
from avalanche.benchmarks.scenarios import CLExperience

DEFAULT_THRESHOLD = 5e-3


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs, threshold=DEFAULT_THRESHOLD):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput


class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs, threshold=DEFAULT_THRESHOLD):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > threshold] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput


class PretrainingMultiTaskClassifier(am.MultiTaskModule):

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

    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
        return self.classifiers[str(task_label)](x.to(dtype=torch.float32))


class MultiTaskClassifier(am.MultiTaskModule):

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

    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
        return self.classifiers[str(task_label)](x)


class ElementWiseLinear(am.MultiTaskModule):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, train_str='mask', zero_out=True, bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.zero_out = zero_out

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        if train_str == 'mask':
            # Weight and bias are no longer Parameters.
            self.weight = Variable(torch.Tensor(
                out_features, in_features), requires_grad=False)
            if bias:
                self.bias = Variable(torch.Tensor(
                    out_features), requires_grad=False)
            else:
                self.register_parameter('bias', None)
        elif train_str == 'weight' or train_str == 'all' or train_str == 'no_DAP' or train_str == 'no_mask':
            self.weight = Parameter(torch.Tensor(
                out_features, in_features), requires_grad=True)
            if bias:
                self.bias = Parameter(torch.Tensor(
                    out_features), requires_grad=True)
            else:
                self.register_parameter('bias', None)

        self.masks = nn.ParameterDict({'0': self.make_mask()})

    def adaptation(self, num_class, task_label):
        if str(task_label) not in self.masks:
            self.masks[str(task_label)] = self.make_mask()

    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
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
        if self.threshold_fn == 'binarizer':
            self.mask_thresholded = Binarizer.apply(
                self.masks[str(task_label)])
        elif self.threshold_fn == 'ternarizer':
            self.mask_thresholded = Ternarizer.apply(
                self.masks[str(task_label)])

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


class ElementWiseEmbedding(am.MultiTaskModule):
    __constants__ = ['num_embeddings', 'embedding_dim', 'padding_idx', 'max_norm',
                     'norm_type', 'scale_grad_by_freq', 'sparse']

    num_embeddings: int
    embedding_dim: int
    padding_idx: Optional[int]
    max_norm: Optional[float]
    norm_type: float
    scale_grad_by_freq: bool
    weight: Tensor
    freeze: bool
    sparse: bool

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Optional[Tensor] = None, _freeze: bool = False,
                 device=None, dtype=None,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        # Weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(
            num_embeddings, embedding_dim), requires_grad=False)
        self.masks = nn.ParameterDict({'0': self.make_mask()})

        self.sparse = sparse

    def make_mask(self):
        # Initialize real-valued mask weights.
        mask_real = self.weight.data.new(self.weight.size())
        if self.mask_init == '1s':
            mask_real.fill_(self.mask_scale)
        elif self.mask_init == 'uniform':
            mask_real.uniform_(-1 * self.mask_scale, self.mask_scale)
        # mask_real is now a trainable parameter.
        return Parameter(mask_real)

    def adaptation(self, num_class, task_label):
        if str(task_label) not in self.masks:
            self.masks[str(task_label)] = self.make_mask()

    def get_weight(self, task_label):
        # For multi-head attention module
        if self.threshold_fn == 'binarizer':
            mask_thresholded = Binarizer.apply(self.masks[str(task_label)])
        elif self.threshold_fn == 'ternarizer':
            mask_thresholded = Ternarizer.apply(self.masks[str(task_label)])

        weight_thresholded = mask_thresholded * self.weight

        return weight_thresholded

    def reset_parameters(self) -> None:
        init.normal_(self.weight)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
        weight_thresholded = self.get_weight(task_label)
        return F.embedding(
            x, weight_thresholded, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)


class LoRALinear(am.MultiTaskModule):
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

        self.weight = Variable(torch.empty(
            (out_features, in_features)), requires_grad=False).to('cuda')
        self.bias = Variable(torch.empty(out_features),
                             requires_grad=False).to('cuda')

        self.fan_in_fan_out = fan_in_fan_out
        self.r = r
        self.in_features = in_features
        self.out_features = out_features

        # Actual trainable parameters
        if self.r > 0:
            self.lora_As = nn.ParameterDict(
                {'0': nn.Parameter(self.weight.new_zeros((r, in_features)))})
            self.lora_Bs = nn.ParameterDict({'0': nn.Parameter(
                self.weight.new_zeros((out_features, r)))})
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        else:
            self.lora_As = {}
            self.lora_Bs = {}
        self.reset_parameters('0')
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self, task_label):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_As[task_label], a=math.sqrt(5))
            nn.init.zeros_(self.lora_Bs[task_label])

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

        for task_label in self.lora_As:
            if mode:
                if self.merge_weights and self.merged:
                    # Make sure that the weights are not merged
                    if self.r > 0:
                        self.weight.data -= T(self.lora_Bs[task_label] @
                                              self.lora_As[task_label]) * self.scaling
                    self.merged = False
            else:
                if self.merge_weights and not self.merged:
                    # Merge the weights and mark it
                    if self.r > 0:
                        self.weight.data += T(self.lora_Bs[task_label] @
                                              self.lora_As[task_label]) * self.scaling
                    self.merged = True

    def adaptation(self, num_class, task_label):
        if str(task_label) not in self.lora_As:
            self.lora_As[str(task_label)] = nn.Parameter(
                self.weight.new_zeros((self.r, self.in_features)))
            self.lora_Bs[str(task_label)] = nn.Parameter(
                self.weight.new_zeros((self.out_features, self.r)))
            self.reset_parameters(str(task_label))

    def forward_single_task(self, x: torch.Tensor, task_label: int) -> torch.Tensor:
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_As[str(task_label)].transpose(0, 1)
                       @ self.lora_Bs[str(task_label)].transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class LoRAPiggybackLinear(LoRALinear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 r: int = 0,
                 lora_alpha: int = 1,
                 lora_dropout: float = 0,
                 fan_in_fan_out: bool = False,
                 merge_weights: bool = True,
                 training_type='posttrain',
                 bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None, **kwargs):
        super().__init__(in_features, out_features, r, lora_alpha,
                         lora_dropout, fan_in_fan_out, merge_weights, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.training_type = training_type

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        if self.training_type == 'finetune':
            self.masks_A = nn.ParameterDict({'0': self.make_mask('A')})
            self.masks_B = nn.ParameterDict({'0': self.make_mask('B')})

    def adaptation(self, num_class, task_label):
        super().adaptation(num_class, task_label)
        if self.training_type == 'finetune' and str(task_label) not in self.masks:
            self.masks_A[str(task_label)] = self.make_mask('A')
            self.masks_B[str(task_label)] = self.make_mask('B')

    def forward_single_task(self, x: Tensor, task_label: int) -> Tensor:
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.training_type == 'posttrain':
                result += (self.lora_dropout(x) @ self.lora_As[str(task_label)].transpose(0, 1)
                           @ self.lora_Bs[str(task_label)].transpose(0, 1)) * self.scaling
            elif self.training_type == 'finetune':
                thresholded_mask_A = Binarizer.apply(
                    self.masks_A[str(task_label)])
                thresholded_mask_B = Binarizer.apply(
                    self.masks_B[str(task_label)])
                result += (self.lora_dropout(x) @ (self.lora_As[str(task_label)] * thresholded_mask_A).transpose(0, 1)
                           @ (self.lora_Bs[str(task_label)] * thresholded_mask_B).transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)

        for task_label in self.lora_As:
            if mode:
                if self.merge_weights and self.merged:
                    # Make sure that the weights are not merged
                    if self.r > 0:
                        if self.training_type == 'posttrain':
                            self.weight.data -= T(self.lora_Bs[task_label] @
                                                  self.lora_As[task_label]) * self.scaling
                        elif self.training_type == 'finetune':
                            thresholded_mask_A = Binarizer.apply(
                                self.masks_A[str(task_label)])
                            thresholded_mask_B = Binarizer.apply(
                                self.masks_B[str(task_label)])
                            self.weight.data -= T((thresholded_mask_B * self.lora_Bs[task_label]) @ (
                                thresholded_mask_A * self.lora_As[task_label])) * self.scaling
                    self.merged = False
            else:
                if self.merge_weights and not self.merged:
                    # Merge the weights and mark it
                    if self.r > 0:
                        if self.training_type == 'posttrain':
                            self.weight.data += T(self.lora_Bs[task_label] @
                                                  self.lora_As[task_label]) * self.scaling
                        elif self.training_type == 'finetune':
                            thresholded_mask_A = Binarizer.apply(
                                self.masks_A[str(task_label)])
                            thresholded_mask_B = Binarizer.apply(
                                self.masks_B[str(task_label)])
                            self.weight.data += T((thresholded_mask_B * self.lora_Bs[task_label]) @ (
                                thresholded_mask_A * self.lora_As[task_label])) * self.scaling
                    self.merged = True

    def make_mask(self, lora):
        # Initialize real-valued mask weights.
        if lora == 'A':
            mask_real = self.weight.data.new(self.lora_As['0'].size())
        elif lora == 'B':
            mask_real = self.weight.data.new(self.lora_Bs['0'].size())

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
        self.bias.data = fn(self.bias.data)