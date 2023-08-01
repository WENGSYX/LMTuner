import torch
import torch.nn as nn
import torch.nn.functional as F
from sat.model.base_model import BaseMixin, BaseModel
import math
from sat import mpu
from transformers.activations import ACT2FN

gelu = ACT2FN["gelu_new"]


class GPT2FinalMixin(BaseMixin):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def final_forward(self, logits, **kwargs):
        return self.lm_head(logits)


class GPT2AttnMixin(BaseMixin):
    def __init__(self, max_positions):
        super().__init__()
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )

    def attention_fn(self, query_layer, key_layer, value_layer, attention_mask,
                     attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
        # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training.
        # The implementation in the paper can be done very easily, if you really need it to train very deep transformers.

        if scaling_attention_score:
            query_layer = query_layer / math.sqrt(query_layer.shape[-1])
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        query_length, key_length = query_layer.size(-2), key_layer.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]
        mask_value = torch.finfo(attention_scores.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attention_scores.dtype, device=attention_scores.device)
        attention_scores = torch.where(causal_mask, attention_scores, mask_value)
        if log_attention_weights is not None:
            attention_scores += log_attention_weights

        if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
            # if auto-regressive, skip
            attention_scores = torch.mul(attention_scores, attention_mask) - \
                               10000.0 * (1.0 - attention_mask)

        attention_probs = F.softmax(attention_scores, dim=-1)

        if attention_dropout is not None:
            if mpu.get_cuda_rng_tracker is not None:
                with mpu.get_cuda_rng_tracker().fork():
                    attention_probs = attention_dropout(attention_probs)
            else:
                attention_probs = attention_dropout(attention_probs)
        if value_layer.dtype != attention_probs.dtype:
            attention_probs = torch.tensor(attention_probs, dtype=value_layer.dtype)
        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer


class GPT2Model(BaseModel):
    def __init__(self, args, transformer=None, **kwargs):
        super(GPT2Model, self).__init__(args, transformer=transformer, activation_func=gelu, **kwargs)
        self.add_mixin("gpt2-final", GPT2FinalMixin(args.vocab_size, args.hidden_size))
        self.add_mixin("gpt2-attn", GPT2AttnMixin(args.max_sequence_length))
        self.pad_token_id = 50256

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
        return l + 1

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
        group = parser.add_argument_group('GPT2', 'GPT2 Configurations')
        # group.add_argument('--num-types', type=int)
        return parser
