import argparse
import torch
import time

from LMTuner.quantization import quantize
from LMTuner.argments import get_args
from sat import get_tokenizer
from sat.arguments import initialize_distributed
from sat.training.model_io import load_checkpoint
from sat.model import GLM130B
from sat.mpu import get_model_parallel_world_size, get_model_parallel_rank, get_model_parallel_group


def add_bminf_args(parser):
    """Arguments for BMInf"""
    group = parser.add_argument_group("BMInf")

    group.add_argument("--bminf", action="store_true", help="Use BMInf to support low resource evaluation")
    group.add_argument("--bminf-memory-limit", type=int, default=20, help="Max memory for model per GPU (in GB)")
    return parser


def add_quantization_args(parser):
    group = parser.add_argument_group("Quantization")

    group.add_argument("--quantization-bit-width", type=int, default=None)
    group.add_argument("--from-quantized-checkpoint", action="store_true", help="Loading from a quantized checkpoint")


def add_initialization_args(parser):
    group = parser.add_argument_group("Initialization")

    group.add_argument(
        "--sequential-initialization",
        action="store_true",
        help="Initialize sequentially in tensor parallel group (reduce CPU RAM for initialization)",
    )


def add_rope_scaling_args(parser):
    group = parser.add_argument_group("rope_scaling")

    group.add_argument("--scaling_type", type=str, default="dynamic", help="ntk-by-parts, linear, dynamic, xpos")
    group.add_argument("--scaling_factor", type=float, default=4.0)
    group.add_argument("--max_position_embeddings", type=int, default=2048,
                       help="The maximum sequence length of the model.")
    group.add_argument("--position_interpolation_scale", type=float, default=1)
    group.add_argument("--original_max_position_embeddings", type=int, default=2048)
    group.add_argument("--ntk_alpha", type=float, default=None)
    group.add_argument("--part_ntk_scale", type=float, default=None)
    group.add_argument("--use_xpos", action="store_true")
    group.add_argument("--use_flash_attention", action="store_true")


def initialize(extra_args_provider):
    parser = argparse.ArgumentParser(add_help=False)
    add_bminf_args(parser)
    add_quantization_args(parser)
    add_initialization_args(parser)
    add_rope_scaling_args(parser)
    GLM130B.add_model_specific_args(parser)
    extra_args_provider(parser)
    known, args_list = parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.do_train = False
    initialize_distributed(args)
    if args.rope_scaling:
        args.rope_scaling = {"type": args.scaling_type, "factor": args.scaling_factor,
                             "original_max_position_embeddings": args.original_max_position_embeddings}
    else:
        args.rope_scaling = None
    return args