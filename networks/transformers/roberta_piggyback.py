import torch.nn as nn
import torch
import math
import avalanche.models as am

from .layers import ElementWiseLinear, MultiTaskClassifier, PretrainingMultiTaskClassifier
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions, MaskedLMOutput, SequenceClassifierOutput
from transformers.modeling_utils import ModuleUtilsMixin
from typing import List, Optional, Tuple, Union
from avalanche.benchmarks.scenarios import CLExperience
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss


class PiggybackRobertaEmbeddings(am.MultiTaskModule):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()        
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        task_label=0,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        return self.forward_single_task(input_ids, token_type_ids, task_label, position_ids, inputs_embeds, past_key_values_length)

    def forward_single_task(
        self, input_ids=None, token_type_ids=None, task_label=0, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, past_key_values_length)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(
                    inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class PiggybackRobertaSelfAttention(am.MultiTaskModule):
    def __init__(self, config, train_str, zero_out, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = ElementWiseLinear(
            config.hidden_size, self.all_head_size, train_str, zero_out, config=config)
        self.key = ElementWiseLinear(
            config.hidden_size, self.all_head_size, train_str, zero_out, config=config)
        self.value = ElementWiseLinear(
            config.hidden_size, self.all_head_size, train_str, zero_out, config=config)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                task_label=0,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.FloatTensor] = None,
                encoder_attention_mask: Optional[torch.FloatTensor] = None,
                past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                output_attentions: Optional[bool] = False,):

        return self.forward_single_task(hidden_states, attention_mask, task_label, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)

    def forward_single_task(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        task_label=0,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states, task_label)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states, task_label))
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states, task_label))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(
                self.key(hidden_states, task_label))
            value_layer = self.transpose_for_scores(
                self.value(hidden_states, task_label))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(
                self.key(hidden_states, task_label))
            value_layer = self.transpose_for_scores(
                self.value(hidden_states, task_label))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        use_cache = past_key_value is not None
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (
            context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class PiggybackRobertaSelfOutput(am.MultiTaskModule):
    def __init__(self, config, train_str, zero_out):
        super().__init__()
        self.dense = ElementWiseLinear(
            config.hidden_size, config.hidden_size, train_str, zero_out, config=config)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, task_label) -> torch.Tensor:
        return self.forward_single_task(hidden_states, input_tensor, task_label)

    def forward_single_task(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, task_label) -> torch.Tensor:
        hidden_states = self.dense(hidden_states, task_label)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class PiggybackRobertaAttention(am.MultiTaskModule):
    def __init__(self, config, train_str, zero_out, position_embedding_type=None):
        super().__init__()
        self.self = PiggybackRobertaSelfAttention(
            config, position_embedding_type=position_embedding_type, train_str=train_str, zero_out=zero_out)
        self.output = PiggybackRobertaSelfOutput(config, train_str, zero_out)

    def adaptation(self, num_class, task_label):

        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            task_label=0,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,):

        return self.forward_single_task(hidden_states, attention_mask, task_label, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)

    def forward_single_task(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        task_label=0,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            task_label,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(
            self_outputs[0], hidden_states, task_label)
        # add attentions if we output them
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class PiggybackRobertaIntermediate(am.MultiTaskModule):
    def __init__(self, config, train_str, zero_out):
        super().__init__()
        self.dense = ElementWiseLinear(
            config.hidden_size, config.intermediate_size, train_str, zero_out, config=config)
        self.intermediate_act_fn = nn.GELU()

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward_single_task(self, hidden_states: torch.Tensor, task_label) -> torch.Tensor:
        hidden_states = self.dense(hidden_states, task_label)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class PiggybackRobertaOutput(am.MultiTaskModule):
    def __init__(self, config, train_str, zero_out):
        super().__init__()
        self.dense = ElementWiseLinear(
            config.intermediate_size, config.hidden_size, train_str, zero_out, config=config)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, task_label) -> torch.Tensor:
        return self.forward_single_task(hidden_states, input_tensor, task_label)

    def forward_single_task(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor, task_label) -> torch.Tensor:
        hidden_states = self.dense(hidden_states, task_label)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class PiggybackRobertaLayer(am.MultiTaskModule):
    def __init__(self, config, train_str, zero_out):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PiggybackRobertaAttention(config, train_str, zero_out)
        self.intermediate = PiggybackRobertaIntermediate(
            config, train_str, zero_out)
        self.output = PiggybackRobertaOutput(config, train_str, zero_out)

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        task_label=0,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        return self.forward_single_task(hidden_states, attention_mask, task_label, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value, output_attentions)

    def forward_single_task(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        task_label=0,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:
                                                  2] if past_key_value is not None else None

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            task_label,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output, task_label)
        layer_output = self.output(
            intermediate_output, attention_output, task_label)
        outputs = (layer_output,) + outputs

        return outputs


class PiggybackRobertaEncoder(am.MultiTaskModule):
    def __init__(self, config, train_str, zero_out):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([PiggybackRobertaLayer(config, train_str, zero_out)
                                   for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        task_label=0,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:

        return self.forward_single_task(hidden_states, attention_mask, task_label, head_mask, encoder_hidden_states, encoder_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict)

    def forward_single_task(
        self,
        hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        task_label=0,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                task_label,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + \
                        (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class PiggybackRobertaPooler(am.MultiTaskModule):
    def __init__(self, config, train_str, zero_out):
        super().__init__()
        self.dense = ElementWiseLinear(
            config.hidden_size, config.hidden_size, train_str, zero_out, config=config)
        self.activation = nn.Tanh()

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward_single_task(self, hidden_states: torch.Tensor, task_label) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor, task_label)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class PiggybackRobertaModel(am.MultiTaskModule):

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->PiggybackRoberta
    def __init__(self, config, args, add_pooling_layer=True, train_str='mask', zero_out=True):
        super().__init__()
        self.config = config
        self.args = args

        self.embeddings = PiggybackRobertaEmbeddings(config)
        self.encoder = PiggybackRobertaEncoder(config, train_str, zero_out)

        self.pooler = PiggybackRobertaPooler(
            config) if add_pooling_layer else None

    def get_extended_attention_mask(
        self, attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = torch.float32
    ) -> Tensor:
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                extended_attention_mask = ModuleUtilsMixin.create_extended_attention_mask_for_decoder(
                    input_shape, attention_mask, device
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (
            1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                task_label=0,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        return self.forward_single_task(input_ids, attention_mask, token_type_ids, task_label, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask,
                                        past_key_values, use_cache, output_attentions, output_hidden_states, return_dict)

    def forward_single_task(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_label=0,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, dtype=self.args.precision)

        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        # head_mask = self.get_head_mask(
        #     head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            task_label=task_label,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            task_label=task_label,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(
            sequence_output, task_label) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class PiggybackRobertaForMaskedLM(am.MultiTaskModule):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.roberta = PiggybackRobertaModel(
            config, args, add_pooling_layer=False)
        self.lm_head = PiggybackRobertaLMHead(config)

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                task_label=0,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                labels: Optional[torch.Tensor] = None,
                ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        return self.forward_single_task(input_ids, attention_mask, token_type_ids, task_label, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask,
                                        output_attentions, output_hidden_states, return_dict, labels)

    def forward_single_task(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        task_label=0,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_label=task_label,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output, task_label)

        masked_lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(prediction_scores.device)
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PiggybackRobertaLMHead(am.MultiTaskModule):
    """PiggybackRoberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = PretrainingMultiTaskClassifier(
            config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.gelu = nn.GELU()

        self.decoder = PretrainingMultiTaskClassifier(
            config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward_single_task(self, features, task_label):
        x = self.dense(features, task_label)
        x = self.gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x, task_label)

        return x


class PiggybackRobertaForSequenceClassification(am.MultiTaskModule):
    def __init__(self, config, args, initial_out_features, zero_out=True, train_str='weight'):
        super().__init__()
        self.config = config
        self.num_labels = args.class_num

        self.roberta = PiggybackRobertaModel(
            config, args, train_str=train_str, add_pooling_layer=False, zero_out=zero_out)
        self.classifier = PiggybackRobertaClassificationHead(
            config, initial_out_features)

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        task_label=0,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.forward_single_task(input_ids, attention_mask, token_type_ids, task_label, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)

    def forward_single_task(self,
                            input_ids: Optional[torch.LongTensor] = None,
                            attention_mask: Optional[torch.FloatTensor] = None,
                            token_type_ids: Optional[torch.LongTensor] = None,
                            task_label=0,
                            position_ids: Optional[torch.LongTensor] = None,
                            head_mask: Optional[torch.FloatTensor] = None,
                            inputs_embeds: Optional[torch.FloatTensor] = None,
                            labels: Optional[torch.LongTensor] = None,
                            output_attentions: Optional[bool] = None,
                            output_hidden_states: Optional[bool] = None,
                            return_dict: Optional[bool] = None,):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_label=task_label,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, task_label)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PiggybackRobertaForLoRAEndtask(am.MultiTaskModule):
    def __init__(self, config, args, initial_out_features, zero_out=True):
        super().__init__()
        self.config = config

        self.roberta = PiggybackRobertaModel(
            config, args, train_str='mask', add_pooling_layer=False, zero_out=zero_out)
        self.classifier = PiggybackRobertaClassificationHead(
            config, initial_out_features)

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        task_label=0,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.forward_single_task(input_ids, attention_mask, token_type_ids, task_label, position_ids, head_mask, inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)

    def forward_single_task(self,
                            input_ids: Optional[torch.LongTensor] = None,
                            attention_mask: Optional[torch.FloatTensor] = None,
                            token_type_ids: Optional[torch.LongTensor] = None,
                            task_label=0,
                            position_ids: Optional[torch.LongTensor] = None,
                            head_mask: Optional[torch.FloatTensor] = None,
                            inputs_embeds: Optional[torch.FloatTensor] = None,
                            labels: Optional[torch.LongTensor] = None,
                            output_attentions: Optional[bool] = None,
                            output_hidden_states: Optional[bool] = None,
                            return_dict: Optional[bool] = None,):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            task_label=task_label,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output, task_label)

        return logits


class PiggybackRobertaClassificationHead(am.MultiTaskModule):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, initial_out_features):
        super().__init__()
        self.dense = PretrainingMultiTaskClassifier(
            config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = MultiTaskClassifier(
            config.hidden_size, initial_out_features)

    def adaptation(self, num_class, task_label):
        for module in self.modules():
            if 'adaptation' in dir(module) and module is not self:
                module.adaptation(num_class, task_label)

    def forward_single_task(self, features, task_label):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x, task_label)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x, task_label)
        return x


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(
        mask) + past_key_values_length) * mask
    return incremental_indices.long() + padding_idx
