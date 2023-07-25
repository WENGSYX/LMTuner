import argparse
import torch
import time

from lingo.quantization import quantize
from lingo.argments import get_args
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


def initialize_model_and_tokenizer(args):
    tokenizer = get_tokenizer(args)

    #torch.distributed.barrier()
    start = time.time()

    for i in range(get_model_parallel_world_size()):
        if get_model_parallel_rank() == i:
            # Initialize model
            model = GLM130B(args).half()

            if args.from_quantized_checkpoint:
                assert args.quantization_bit_width is not None
                # Quantize model before moving to GPU
                model = quantize(model, args.quantization_bit_width)

            # Load checkpoint
            load_checkpoint(model, args)

            if args.quantization_bit_width is not None and not args.from_quantized_checkpoint:
                # Quantize model before moving to GPU
                model = quantize(model, args.quantization_bit_width)

            if args.bminf:
                import bminf

                if torch.distributed.get_rank() == 0:
                    print(f"> BMInf activated, memory limit: {args.bminf_memory_limit} GB")
                with torch.cuda.device(args.device):
                    model = bminf.wrapper(model, quantization=False, memory_limit=args.bminf_memory_limit << 30)
            else:
                model = model.to(args.device)
        if args.sequential_initialization:
            #torch.distributed.barrier(group=get_model_parallel_group())
            pass
    print(2)
    #torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(f"> Model initialized in {time.time() - start:.1f}s")

    torch.cuda.empty_cache()
    #model.eval()

    # generate rotary embedding cache
    original_parallel_output = model.transformer.parallel_output
    model.transformer.parallel_output = False

    import re
    raw_text = '窗前明月光，疑是'

    generation_mask = "[gMASK]"
    if "[MASK]" in raw_text:
        generation_mask = "[MASK]"
    elif "[sMASK]" in raw_text:
        generation_mask = "[sMASK]"
    use_gmask = "[MASK]" not in raw_text and "[sMASK]" not in raw_text

    mask_pattern = r"\[[sg]?MASK\]"
    text_list = re.split(mask_pattern, raw_text)
    pattern_list = re.compile(mask_pattern).findall(raw_text)
    seq = []
    for i in range(len(pattern_list)):
        pattern = pattern_list[i]
        sub_text = text_list[i]
        seq.extend(tokenizer.tokenize(sub_text))
        seq.append(tokenizer.get_command(pattern))

    seq.extend(tokenizer.tokenize(text_list[-1]))

    if "MASK]" not in raw_text:
        seq += [tokenizer.get_command(generation_mask)]
        raw_text += " " + generation_mask
    if not raw_text.endswith("MASK]"):
        seq = seq + [tokenizer.get_command("eos")]

    # detect mask position
    mask_token = tokenizer.get_command(generation_mask)
    mask_position = seq.index(mask_token)

    output_list = []

    input_seq = torch.cuda.LongTensor(
        [seq + [tokenizer.get_command("sop")]],
        device=args.device,
    )
    print(len(seq))
    seq = input_seq
    context_length = seq.shape[1]
    tokens = torch.nn.functional.pad(seq, (0, 256), mode="constant", value=-1)
    attention_mask = torch.ones((1, tokens.shape[-1], tokens.shape[-1]), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., : context_length - 1] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    position_ids = torch.arange(tokens.shape[-1], dtype=torch.long, device=tokens.device)
    if 1:
        position_ids[context_length - 1:] = mask_position

    position_ids = position_ids.unsqueeze(0)



    with torch.no_grad():
        a, *_ = model(
            tokens.to(torch.int64),
            position_ids.to(torch.int64),
            attention_mask.to(torch.bool),
        )
    print(a)
    print(a.shape)
    print(tokenizer.detokenize(a.argmax(2).tolist()[0]))
    model.transformer.parallel_output = original_parallel_output
    #torch.distributed.barrier()

    return model, tokenizer
