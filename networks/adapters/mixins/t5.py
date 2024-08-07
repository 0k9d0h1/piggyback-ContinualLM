import logging
from typing import Iterable, Tuple

import torch.nn as nn

from ..layer import AdapterLayer
from ..model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin


logger = logging.getLogger(__name__)


class T5SelfAttentionLayerAdaptersMixin(AdapterLayer):
    def __init__(self, args):
        super().__init__("mh_adapter", None, args=args)
        self.args = args
        

class T5CrossAttentionLayerAdaptersMixin(AdapterLayer):
    def __init__(self, args):
        super().__init__("cross_adapter", None, args=args)
        self.args = args


class T5FFLayerAdaptersMixin(AdapterLayer):
    def __init__(self, args):
        super().__init__("output_adapter", None, args=args)
        self.args = args


class T5ModelAdaptersMixin(InvertibleAdaptersMixin, ModelAdaptersMixin):
    """Adds adapters to the T5Model class."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        if hasattr(self, "encoder"):
            for i, layer in enumerate(self.encoder.block):
                yield i, layer
            for i, layer in enumerate(self.decoder.block, start=len(self.encoder.block)):
                yield i, layer
        else:
            for i, layer in enumerate(self.decoder.block):
                yield i, layer

    def _init_adapter_modules(self):
        if hasattr(self, "encoder"):
            # In T5, the invertible adapters are implemented by the encoder module.
            # Therefore, relay mixin calls to the encoder here.
            self.invertible_adapters = self.encoder.invertible_adapters
            self.add_invertible_adapter = self.encoder.add_invertible_adapter
            self.get_invertible_adapter = self.encoder.get_invertible_adapter
            self.enable_invertible_adapters = self.encoder.enable_invertible_adapters
            self.invertible_adapters_forward = self.encoder.invertible_adapters_forward
            self.delete_invertible_adapter = self.encoder.delete_invertible_adapter
        super()._init_adapter_modules()
