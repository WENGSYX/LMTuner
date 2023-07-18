import os
import torch
from torch.optim import Optimizer
import torch.distributed as dist

import copy
from dataclasses import dataclass

import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.utils import PaddingStrategy
from transformers.trainer import *



@dataclass
class DataCollatorForCauselLM:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: Any
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    padding_side: str = 'right'

    def __call__(self, features, return_tensors=None):
        padding_side = self.padding_side

        # if return_tensors is None:
        #     return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        max_length = max(len(feature['input_ids']) for feature in features)
        if padding_side == 'right':
            input_ids = [feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_length - len(feature['input_ids']))
                         for feature in features]
            attention_mask = [[1] * len(feature['input_ids']) + [0] * (max_length - len(feature['input_ids'])) for
                              feature in features]
        elif padding_side == 'left':
            input_ids = [[self.tokenizer.pad_token_id] * (max_length - len(feature['input_ids'])) + feature['input_ids']
                         for feature in features]
            attention_mask = [[0] * (max_length - len(feature['input_ids'])) + [1] * len(feature['input_ids']) for
                              feature in features]
        else:
            raise ValueError("Invalid padding strategy:" + str(padding_side))

        features = {
            'input_ids': torch.tensor(input_ids).long(),
            'attention_mask': torch.tensor(attention_mask).long(),
            'labels': torch.tensor(np.array([feature['labels'] for feature in features])).long()
        }
        return features


@dataclass
class EvalDataCollatorForCauselLM:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: Any
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    padding_side: str = 'left'
    unconditional_normalization: bool = False

    def __call__(self, features, return_tensors=None):
        padding_side = self.padding_side

        split_size = []
        new_features = []
        assert "labels" in features[0].keys()
        for feature in features:
            split_size.append(len(feature["labels"]))
            for op_input_ids, op_labels in zip(feature["input_ids"], feature["labels"]):
                un_mask = np.zeros_like(op_labels)
                un_mask_index = np.where(op_labels == self.label_pad_token_id, 1, 0).sum() - 2
                un_mask[:un_mask_index] = 1
                new_features.append({"input_ids": op_input_ids, "labels": op_labels, "un_mask": un_mask})

        labels = [feature["labels"] for feature in new_features]
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            for feature in new_features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                    feature["un_mask"] = np.concatenate([feature["un_mask"], np.ones_like(remainder)]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
                    feature["un_mask"] = np.concatenate([np.ones_like(remainder), feature["un_mask"]]).astype(np.int64)

        max_length = max(len(feature['input_ids']) for feature in new_features)
        if padding_side == 'right':
            input_ids = [feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_length - len(feature['input_ids']))
                         for feature in new_features]
            attention_mask = [[1] * len(feature['input_ids']) + [0] * (max_length - len(feature['input_ids'])) for
                              feature in new_features]
        elif padding_side == 'left':
            input_ids = [[self.tokenizer.pad_token_id] * (max_length - len(feature['input_ids'])) + feature['input_ids']
                         for feature in new_features]
            attention_mask = [[0] * (max_length - len(feature['input_ids'])) + [1] * len(feature['input_ids']) for
                              feature in new_features]
        else:
            raise ValueError("Invalid padding strategy:" + str(padding_side))

        batched_features = {
            'input_ids': torch.tensor(input_ids).long(),
            'attention_mask': torch.tensor(attention_mask).long(),
            'labels': torch.tensor(np.array([feature['labels'] for feature in new_features])).long(),
            'split_size': split_size
        }
        if self.unconditional_normalization:
            batched_features['un_mask'] = torch.tensor(np.array([feature['un_mask'] for feature in new_features])).bool()

        return batched_features


class LearningRateScheduler:
    r"""
    Learning rate scheduler with warmup.

        :param warmup: if ``warmup`` is an integer, ``warmup`` stands for warmup steps, if ``warmup`` is a float,
            such as 0.1, then it stands for warmup_ratio.
        :param schedule: the learning rate will be adjusted according to ``schedule`` strategy,
            which can be: linear or constant.
    """

    def __init__(self,
                 warmup: float,
                 schedule: str,
                 learning_rate: float,
                 n_steps: int = 0):

        self.warmup = max(warmup, 0.)
        self.schedule = schedule
        self.initial_lr = learning_rate

        if self.warmup > 1:
            self.warmup = self.warmup / n_steps
        self.t_steps = max(2, n_steps)

        if self.schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif self.schedule == 'linear':
            self.get_lr = self._get_linear_lr
        else:
            raise NotImplementedError("Only support 'linear', 'constant'.")

    def _get_constant_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return 1

    def _get_linear_lr(self, progress):
        if progress < self.warmup:
            return progress / self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)

    def step(self, global_step):
        progress = global_step / self.t_steps
        return self.initial_lr * self.get_lr(progress)


class WandbLogger:
    """
    使用 wandb 记录信息的类。

    :param training_args: Trainer 的参数
    """

    def __init__(self, training_args):
        self.training_args = training_args
        # report_to is a list
        self.able = "wandb" in getattr(training_args, "report_to", [])
        if self.able and 'wandb' not in sys.modules:
            raise ModuleNotFoundError(
                "Detected Wandb not installed while you have set "
                "`report_to=['wandb']` in your training config. Please "
                "either set `report_to` to another value or install wandb.")

    def init(self, *args, **kwargs):
        if self.able:
            wandb.init(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.able:
            wandb.log(*args, **kwargs)

    def set_summary(self, key, value):
        if self.able:
            wandb.run.summary[key] = value


class DynamicLossScaler:
    def __init__(self,
                 init_scale=2 ** 32,
                 scale_factor=2.,
                 scale_window=1000,
                 min_scale=1,
                 delayed_shift=1,
                 consecutive_hysteresis=False,
                 raise_error_at_min_scale=True,
                 dtype=torch.half):
        self.cur_scale = init_scale
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis
        self.raise_error_at_min_scale = raise_error_at_min_scale
        self.dtype = dtype
        self.has_overflow_serial = False

    @property
    def loss_scale(self):
        return self.cur_scale

    # `x` is a torch.Tensor
    def _has_inf_or_nan(self, x):
        try:
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum in [float('inf'), -float('inf')] or cpu_sum != cpu_sum:
                return True
            return False

    # `overflow` is boolean indicating whether the gradient overflowed
    def update_scale(self, overflow):
        if overflow:
            # self.cur_scale /= self.scale_factor
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                if (self.cur_scale == self.min_scale) and self.raise_error_at_min_scale:
                    raise Exception(
                        "Current loss scale already at minimum - cannot decrease scale anymore. Exiting run.")
                else:
                    next_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
                    if torch.distributed.get_rank() == 0:
                        overflow_msg = f"[deepspeed] OVERFLOW! Rank {torch.distributed.get_rank()} Skipping step."
                        if self.dtype == torch.half:
                            overflow_msg += f" Attempted loss scale: {int(self.cur_scale)}, reducing to {int(next_scale)}"
                        print(overflow_msg)
                    self.cur_scale = next_scale
            else:
                if torch.distributed.get_rank() == 0:
                    overflow_msg = f"[deepspeed] OVERFLOW! Rank {torch.distributed.get_rank()} Skipping step."
                    if self.dtype == torch.half:
                        overflow_msg += f" Attempted loss scale: {int(self.cur_scale)}, but hysteresis is {self.cur_hysteresis}. Reducing hysteresis to {self.cur_hysteresis - 1}"
                    print(overflow_msg)
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                if torch.distributed.get_rank() == 0:
                    hysteresis_msg = f"Consecutive hysteresis is enabled. Restoring hysteresis to {self.delayed_shift}"
                    print(hysteresis_msg)
                self.cur_hysteresis = self.delayed_shift
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1


def get_loss(logits, labels, clip_loss_value=None):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    # Flatten the tokens
    if clip_loss_value is not None:
        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                        shift_labels.view(-1).cuda())
        loss.data.clamp_(min=-clip_loss_value, max=clip_loss_value)
        loss = loss.mean()
    else:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(shift_labels.shape[0] * shift_labels.shape[1], -1),
                        shift_labels.view(-1).cuda())
    return loss


class LOMO(Optimizer):
    """
    一个自定义的优化器类LOMO，用于在分布式训练中的梯度更新。

    该类实现两个梯度更新函数 :meth:`fuse_update` 和 :meth:`fuse_update_zero3`，分别用于非ZeRO和ZeRO模式下的梯度更新。

    :param model: 待优化的模型
    :param lr: 学习率，默认值为1e-3
    :param clip_grad_norm: 梯度裁剪的范数阈值

        .. note::

            clip_grad_norm须为正数

    :param clip_grad_value: 梯度裁剪的值域阈值
    """

    def __init__(self, model, lr=1e-3, clip_grad_norm=None, clip_grad_value=None,args=None):
        self.model = model
        self.lr = lr
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = dist.get_world_size()
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

        # for grad norm
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError(f"clip_grad_norm should be positive, got {self.clip_grad_norm}.")
        self.gather_norm = False
        self.grad_norms = []
        self.clip_coef = None
        self.args = args

        # check if zero3 is enabled
        p0 = list(self.model.parameters())[0]
        if hasattr(p0, 'ds_tensor'):  # zero3 is enabled
            self.grad_func = self.fuse_update_zero3()
        else:
            self.grad_func = self.fuse_update()
        self.grad_func = self.fuse_update_zero3()
        # check if fp16 is enabled
        if p0.dtype == torch.float16:
            self.loss_scaler = DynamicLossScaler(
                init_scale=2 ** 16,
            )  # TODO: add args
            if self.clip_grad_norm is None:
                raise ValueError(
                    "Loss scaling is recommended to be used with grad norm to get better performance."
                )
        else:
            self.loss_scaler = None

        # register hook function, which will be called through the backward process
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.register_hook(self.grad_func)
        defaults = dict(lr=lr, clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)
        super(LOMO, self).__init__(self.model.parameters(), defaults)

    def fuse_update(self):
        """
        在非ZeRO模式下更新模型参数的梯度。

        :return: func，一个闭包函数，用于更新模型参数的梯度
        """

        def func(x):
            """
            闭包函数，用于更新模型参数的梯度。
            """
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        if self.loss_scaler:
                            if self.loss_scaler.has_overflow_serial or self.loss_scaler._has_inf_or_nan(p.grad):
                                # if the overflow is detected, drop the gradient
                                p.grad = None
                                self.loss_scaler.has_overflow_serial = True
                                break
                        grad_fp32 = p.grad.to(torch.float32)
                        p.grad = None
                        if self.loss_scaler:
                            grad_fp32.div_(self.loss_scaler.loss_scale)
                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm compuation and parameter update, respectively.
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:
                            if self.clip_grad_value is not None and self.clip_grad_value > 0:
                                # Clipping gradients by their value
                                grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                            if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                                # Normalize the gradient according to its norm (computed in another pass)
                                grad_fp32.mul_(self.clip_coef)
                            p_fp32 = p.data.to(torch.float32)
                            try:
                                p_fp32.add_(grad_fp32, alpha=-self.lr)
                            except:
                                print(p_fp32.shape,grad_fp32.shape)
                                p_fp32.add_(grad_fp32, alpha=-self.lr)
                            p.data.copy_(p_fp32)

            return x

        return func

    def fuse_update_zero3(self):
        """
        在ZeRO模式下更新模型参数的梯度。

        :return: func，一个闭包函数，用于更新模型参数的梯度。
        """
        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG, async_op=False)
                        if self.loss_scaler:
                            if self.loss_scaler.has_overflow_serial or self.loss_scaler._has_inf_or_nan(p.grad):
                                # if the overflow is detected, drop the gradient
                                p.grad = None
                                self.loss_scaler.has_overflow_serial = True
                                break

                        grad_fp32 = p.grad.to(torch.float32)
                        p.grad = None
                        param_fp32 = p.ds_tensor.to(torch.float32)
                        if self.loss_scaler:
                            grad_fp32.div_(self.loss_scaler.loss_scale)

                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm compuation and parameter update, respectively.
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:  # update param
                            if self.args.models == 'GLM-130B':
                                one_dim_grad_fp32 = grad_fp32.view(-1)
                                partitioned_grad_fp32 = one_dim_grad_fp32

                                if self.clip_grad_value is not None:
                                    # Clipping gradients by their value
                                    partitioned_grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                                if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                                    # Normalize the gradient according to its norm (computed in another pass)
                                    partitioned_grad_fp32.mul_(self.clip_coef)

                                partitioned_p = param_fp32

                                try:
                                    partitioned_p.add_(partitioned_grad_fp32, alpha=-self.lr)
                                except:
                                    print('pp',partitioned_p.shape)
                                    print('pg',partitioned_grad_fp32.shape)
                                    partitioned_p.add_(partitioned_grad_fp32, alpha=-self.lr)

                                p.ds_tensor[:] = partitioned_p
                            else:

                                one_dim_grad_fp32 = grad_fp32.view(-1)
                                partition_size = p.ds_tensor.numel()
                                start = partition_size * self.local_rank
                                end = min(start + partition_size, grad_fp32.numel())
                                partitioned_grad_fp32 = one_dim_grad_fp32.narrow(0, start, end - start)

                                if self.clip_grad_value is not None:
                                    # Clipping gradients by their value
                                    partitioned_grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                                if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                                    # Normalize the gradient according to its norm (computed in another pass)
                                    partitioned_grad_fp32.mul_(self.clip_coef)

                                partitioned_p = param_fp32.narrow(0, 0, end - start)

                                partitioned_p.add_(partitioned_grad_fp32.to(partitioned_p.device), alpha=-self.lr)
                                p.ds_tensor[: end - start] = partitioned_p
            return x

        return func

    def fused_backward(self, loss, lr):
        """
        执行一步反向传播并更新模型的梯度。

        :param loss: 模型的loss值
        :param lr: 学习率
        """
        self.lr = lr
        # Users need call grad_norm themselves and then call backward_step
        if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is None:
            raise ValueError(
                "clip_grad_norm is not None, but clip_coef is None. "
                "Please call optimizer.grad_norm() before optimizer.fused_backward()."
            )
        if self.loss_scaler:
            loss = loss * self.loss_scaler.loss_scale
        loss.backward()
        # update the last parameter since the last parameter in the computaiton graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything. 
        self.grad_func(0)

    def grad_norm(self, loss):
        """
        计算梯度的范数。

        :param loss: 模型的loss值
        """
        self.gather_norm = True
        self.grad_norms = []
        if self.loss_scaler:
            self.loss_scaler.has_overflow_serial = False
            loss = loss * self.loss_scaler.loss_scale
        loss.backward(retain_graph=True)
        # update the last parameter since the last parameter in the computaiton graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything. 
        self.grad_func(0)

        if self.loss_scaler and self.loss_scaler.has_overflow_serial:
            self.loss_scaler.update_scale(overflow=True)
            with torch.no_grad():  # clear gradients
                for n, p in self.model.named_parameters():
                    p.grad = None
            return


        with torch.no_grad():
            # The norm is computed over all gradients together, as if they were
            # concatenated into a single vector. Gradients are modified in-place.
            self.grad_norms = torch.stack(self.grad_norms)

            total_norm = torch.norm(self.grad_norms, 2.0)
            self.clip_coef = float(self.clip_grad_norm) / (total_norm + 1e-6)
            self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
        self.gather_norm = False

