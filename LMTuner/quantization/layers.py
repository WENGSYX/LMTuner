import torch
from torch.nn.parameter import Parameter

from sat.mpu import copy_to_model_parallel_region
from sat.mpu import gather_from_model_parallel_region
from sat.mpu import reduce_from_model_parallel_region
from sat.mpu import scatter_to_model_parallel_region
from sat.mpu import ColumnParallelLinear, RowParallelLinear

from .functional import W8A16Linear
from LMTuner.kernels import compress_int4_weight
from typing import Optional, TypeVar, Union, overload
T = TypeVar("T", bound="torch.nn.Module")

import torch.nn.functional as F
from torch import Tensor, device, dtype, nn

import bitsandbytes as bnb
import bitsandbytes.functional
from bitsandbytes.autograd._functions import get_inverse_transform_indices, undo_layout
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.utils import OutlierTracer, find_outlier_dims

class Params4bit(torch.nn.Parameter):
    def __new__(cls, data=None, requires_grad=True, quant_state=None, blocksize=64, compress_statistics=True, quant_type='fp4'):
        if data is None:
            data = torch.empty(0)

        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.data = data
        return self

    def cuda(self, device):
        w = self.data.contiguous().half().cuda(device)
        w_4bit, quant_state = bnb.functional.quantize_4bit(w, blocksize=self.blocksize, compress_statistics=self.compress_statistics, quant_type=self.quant_type)
        self.data = w_4bit
        self.quant_state = quant_state

        return self

    @overload
    def to(self: T, device: Optional[Union[int, device]] = ..., dtype: Optional[Union[dtype, str]] = ..., non_blocking: bool = ...,) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if (device is not None and device.type == "cuda" and self.data.device.type == "cpu"):
            return self.cuda(device)
        else:
            s = self.quant_state
            if s is not None:
                # make sure the quantization state is on the right device
                s[0] = s[0].to(device)
                if self.compress_statistics:
                    # TODO: refactor this. This is a nightmare
                    # for 4-bit:
                    # state = [qabsmax, input_shape, A.dtype, blocksize, [offset, state2], quant_type]
                    # state2 = [absmax, input_shape, A.dtype, blocksize, None, quant_type]
                    #s[-2][0] = s[-2][0].to(device) # offset
                    #s[-2][1][0] = s[-2][1][0].to(device) # nested absmax

                    # for 8-bit
                    s[-2][0] = s[-2][0].to(device) # offset
                    s[-2][1][0] = s[-2][1][0].to(device) # nested quantiation state statitics
                    s[-2][1][1] = s[-2][1][1].to(device) # nested quantiation codebook
            new_param = Params4bit(super().to(device=device, dtype=dtype, non_blocking=non_blocking),
                                  requires_grad=self.requires_grad, quant_state=self.quant_state,
                                   blocksize=self.blocksize, compress_statistics=self.compress_statistics,
                                   quant_type=self.quant_type)

            return new_param


class QuantizedColumnParallelLinearNF4(ColumnParallelLinear):
    def __init__(self, weight_bit_width: int, weight=None, *args, **kwargs):
        super(QuantizedColumnParallelLinearNF4, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

        del self.weight

        self.weight = Params4bit(weight.data, requires_grad=True, compress_statistics=True,
                                 quant_type='nf4')

    def forward(self, input_):
        input_parallel = copy_to_model_parallel_region(input_)
        if self.bias is not None and self.bias.dtype != input_.dtype:
            self.bias.data = self.bias.data.to(input_.dtype)

        if getattr(self.weight, 'quant_state', None) is None:
            print('FP4 quantization state not initialized. Please call .cuda() or .to(device) on the LinearFP4 layer first.')


        output_parallel = bnb.matmul_4bit(input_, self.weight.t(), bias=None, quant_state=self.weight.quant_state)
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel

        return output


class QuantizedRowParallelLinearNF4(RowParallelLinear):
    def __init__(self, weight_bit_width: int, weight=None, *args, **kwargs):
        super(QuantizedRowParallelLinearNF4, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

        del self.weight

        self.weight = Params4bit(weight.data, requires_grad=True, compress_statistics=True,
                                 quant_type='nf4')

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = bnb.matmul_4bit(input_, self.weight.t(), bias=None, quant_state=self.weight.quant_state)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output






class QuantizedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, weight_bit_width: int, weight=None, *args, **kwargs):
        super(QuantizedColumnParallelLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

        shape = self.weight.shape
        del self.weight

        if weight is None:
            self.weight = torch.empty(
                shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs["device"]
            )
            self.weight_scale = torch.empty(shape[0], dtype=kwargs["params_dtype"], device=kwargs["device"])
        else:
            self.weight_scale = (weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).half()
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(torch.int8)
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)

        self.weight = Parameter(self.weight.to(kwargs["device"]), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(kwargs["device"]), requires_grad=False)

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = W8A16Linear.apply(input_parallel, self.weight, self.weight_scale, self.weight_bit_width)
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        return output


class QuantizedRowParallelLinear(RowParallelLinear):
    def __init__(self, weight_bit_width: int, weight=None, *args, **kwargs):
        super(QuantizedRowParallelLinear, self).__init__(*args, **kwargs)
        self.weight_bit_width = weight_bit_width

        shape = self.weight.shape
        del self.weight

        if weight is None:
            self.weight = torch.empty(
                shape[0], shape[1] * weight_bit_width // 8, dtype=torch.int8, device=kwargs["device"]
            )
            self.weight_scale = torch.empty(shape[0], dtype=kwargs["params_dtype"], device=kwargs["device"])
        else:
            self.weight_scale = (weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)).half()
            self.weight = torch.round(weight / self.weight_scale[:, None]).to(torch.int8)
            if weight_bit_width == 4:
                self.weight = compress_int4_weight(self.weight)

        self.weight = Parameter(self.weight.to(kwargs["device"]), requires_grad=False)
        self.weight_scale = Parameter(self.weight_scale.to(kwargs["device"]), requires_grad=False)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = W8A16Linear.apply(input_parallel, self.weight, self.weight_scale, self.weight_bit_width)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        return output
