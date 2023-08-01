from sat.model import BaseMixin, BaseModel
import torch
import torch.nn as nn

from LMTuner.models.GPT2_model import GPT2AttnMixin
from sat.transformer_defaults import standard_attention
from sat.mpu.utils import split_tensor_along_last_dim
from sat.model.position_embedding.rotary_embeddings import RotaryEmbedding, rotate_half
import torch.nn.functional as F
from sat.mpu import ColumnParallelLinear
from LMTuner.scaled_rope.modelling_llama import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaNTKByPartsRotaryEmbedding,
    LlamaRotaryEmbedding,
    LlamaXposRotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.modules.mha import FlashSelfAttention
    from einops import rearrange

    have_flash_attention = True
except:
    have_flash_attention = False
    

class RotaryMixin(BaseMixin):
    def __init__(self, args, hidden_size, num_heads):
        super().__init__()
        self.config = args
        self.head_dim = hidden_size // num_heads
        self.max_position_embeddings = args.max_position_embeddings

        self._init_rope()
        self.use_flash_attention = args.use_flash_attention
        if self.use_flash_attention:
            if not have_flash_attention:
                raise RuntimeError("Flash Attention 2 not installed")
            self.flash_attention = FlashSelfAttention(causal=True)

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "xpos":
                self.rotary_emb = LlamaXposRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "ntk-by-parts":
                original_max_position_embeddings = self.config.rope_scaling[
                    "original_max_position_embeddings"
                ]
                self.rotary_emb = LlamaNTKByPartsRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    original_max_position_embeddings=original_max_position_embeddings,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def attention_forward(self, hidden_states, mask, **kw_args):
        origin = self
        self = self.transformer.layers[kw_args['layer_id']].attention
        attention_fn = standard_attention
        if 'attention_fn' in self.hooks:
            attention_fn = self.hooks['attention_fn']

        mixed_raw_layer = self.query_key_value(hidden_states)
        (
            mixed_query_layer,
            mixed_key_layer,
            mixed_value_layer,
        ) = split_tensor_along_last_dim(mixed_raw_layer, 3)

        dropout_fn = self.attention_dropout if self.training else None

        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)
        cos, sin = origin.rotary_emb(
            value_layer, seq_len=kw_args['position_ids'].max() + 1
        )
        if origin.config.rope_scaling:
            if origin.config.rope_scaling["type"] == "xpos":
                query_layer, key_layer = origin.rotary_emb.apply_rotary_pos_emb(
                    query_layer, key_layer, cos, sin, kw_args['position_ids']
                )
            else:
                query_layer, key_layer = apply_rotary_pos_emb(
                    query_layer, key_layer, cos, sin, kw_args['position_ids']
                )

        context_layer = attention_fn(
            query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)
        output = self.dense(context_layer)

        if self.training:
            output = self.output_dropout(output)
        return output


class LLaMAMlpMixin(BaseMixin):
    def __init__(self, num_layers, in_features, hidden_features):
        super().__init__()
        hidden_features = 4 * in_features if hidden_features is None else hidden_features
        self.gate_proj = nn.ModuleList([ColumnParallelLinear(
            in_features,
            hidden_features,
            gather_output=False,
            # init_method=init_method,
            bias=False,
            # params_dtype=params_dtype,
            module=self,
            name="dense_h_to_4h_gate",
            # skip_init=skip_init,
            # device=device
        ) for i in range(num_layers)])

    def mlp_forward(self, hidden_states, **kw_args):
        origin = self.transformer.layers[kw_args['layer_id']].mlp
        hidden_states = origin.activation_func(
            self.gate_proj[kw_args['layer_id']](hidden_states)) * origin.dense_h_to_4h(hidden_states)
        hidden_states = origin.dense_4h_to_h(hidden_states)
        return hidden_states


class LMMixin(BaseMixin):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lm_head = ColumnParallelLinear(
            hidden_size,
            vocab_size,
            gather_output=True,
            # init_method=init_method,
            bias=False,
            # params_dtype=params_dtype,
            module=self,
            name="lm_head",
            # skip_init=skip_init,
            # device=device
        )

    def final_forward(self, logits, **kwargs):
        return self.lm_head(logits)


from sat.model.normalization import RMSNorm


class LLaMAModel(BaseModel):
    def __init__(self, args, transformer=None, parallel_output=True, layernorm=RMSNorm,
                 activation_func=nn.functional.silu, **kwargs):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output, layernorm=layernorm,
                         activation_func=activation_func, **kwargs)
        del self.transformer.position_embeddings
        self.add_mixin("rotary", RotaryMixin(args, args.hidden_size, args.num_attention_heads))
        self.add_mixin("lm", LMMixin(args.vocab_size, args.hidden_size))
        self.add_mixin("mlp", LLaMAMlpMixin(args.num_layers, args.hidden_size, args.inner_hidden_size))
        self.add_mixin("causal", GPT2AttnMixin(args.max_sequence_length))

        self.pad_token_id = 1
    def position_embedding_forward(self, *args, **kwargs):
        return None

    def forward(self, input_ids, position_ids=None, attention_mask=None, past_key_values=None, **kwargs):
        if attention_mask is None and position_ids is None:
            attention_mask, position_ids = self.get_inputs(input_ids, attention_mask=attention_mask,
                                                           position_ids=position_ids, past_key_values=past_key_values,
                                                           **kwargs)

        return super().forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                               past_key_values=past_key_values, **kwargs)

    def get_inputs(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        if attention_mask is None:
            if past_key_values is not None and input_ids.size(0) == 1:
                attention_mask = torch.tensor([[1]], dtype=torch.long, device=input_ids.device)
            else:
                attention_mask = self.get_masks(
                    input_ids=input_ids,
                    device=input_ids.device, **kwargs
                )
        if position_ids is None:
            position_ids = []
            for _ in input_ids:
                position_ids.append(torch.arange(input_ids.size(1)))
            position_ids = torch.stack(position_ids).to(input_ids.device).to(torch.int64)
        return attention_mask, position_ids

    def get_pad_length(self, seq):
        l = 0
        while l < len(seq) and seq[l] != self.pad_token_id:
            l += 1
        return l

    def get_masks(self, input_ids, device, **kwargs):
        batch_size, seq_length = input_ids.shape
        attention_mask = torch.zeros((batch_size, seq_length), device=device).to(torch.bfloat16)

        pad_lengths = [self.get_pad_length(seq.tolist()) for seq in input_ids]
        for i, pad_length in enumerate(pad_lengths):
            attention_mask[i, :pad_length] = 1
        attention_mask = attention_mask[:, None, None, :]

        return attention_mask

    @classmethod
    def add_model_specific_args(cls, parser):
        group = parser.add_argument_group('LLaMA', 'LLaMA Configurations')
        group.add_argument('--bos-token-id', type=int, default=0)
        group.add_argument('--eos-token-id', type=int, default=1)
        group.add_argument('--pad-token-id', type=int, default=1)
        return parser
