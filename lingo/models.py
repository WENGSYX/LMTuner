from sat import AutoModel
from sat.model.finetune import PTuningV2Mixin
from lingo.lora import LoraMixin
from sat.training.model_io import load_checkpoint
from sat.mpu import get_model_parallel_world_size, get_model_parallel_rank, get_model_parallel_group

from lingo.quantization import quantize
from transformers import AutoTokenizer, LlamaTokenizer






def get_model_and_tokenizer(args):
    if args.models == 'ChatGLM-6B':
        args.models = 'chatglm-6b'
        from sat.model.official import ChatGLMModel as MODEL
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)

        def preprocess_function_train(examples):
            prefix = args.source_prefix if args.source_prefix is not None else ""

            model_inputs = {
                "input_ids": [],
                "labels": [],
            }
            for i in range(len(examples[args.prompt_column])):
                if examples[args.prompt_column][i] and examples[args.response_column][i]:
                    prompt, answer = examples[args.prompt_column][i], examples[args.response_column][i]
                    prompt = prefix + prompt
                    a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                    b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                    seq_length = len(a_ids) + len(b_ids) + 2

                    input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
                    context_length = input_ids.index(tokenizer.bos_token_id)
                    mask_position = context_length - 1
                    labels = [-100] * context_length + input_ids[mask_position + 1:]

                    pad_len = args.max_seq_length - len(input_ids)
                    input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                    labels = labels + [tokenizer.pad_token_id] * pad_len

                    if 130004 in input_ids[:args.max_seq_length-1]:
                        model_inputs["input_ids"].append(input_ids[:args.max_seq_length])
                        model_inputs["labels"].append(labels[:args.max_seq_length])

            return model_inputs
    elif args.models == 'ChatGLM2-6B':
        from sat.model.official import ChatGLM2Model as MODEL
        args.models = 'chatglm2-6b'
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm2-6b', trust_remote_code=True)

        def preprocess_function_train(examples):

            prefix = args.source_prefix if args.source_prefix is not None else ""

            model_inputs = {
                "input_ids": [],
                "labels": [],
            }
            for i in range(len(examples[args.prompt_column])):
                if examples[args.prompt_column][i] and examples[args.response_column][i]:
                    prompt, answer = examples[args.prompt_column][i], examples[args.response_column][i]
                    prompt = prefix + prompt
                    a_ids = tokenizer.encode(text=prompt, add_special_tokens=True)
                    b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                    seq_length = len(a_ids) + len(b_ids) + 1

                    input_id = a_ids + b_ids + [tokenizer.eos_token_id]
                    label = [-100] * len(a_ids) + b_ids + [tokenizer.eos_token_id]

                    input_ids = input_id + (args.max_seq_length - seq_length) * [tokenizer.pad_token_id]
                    labels = label + (args.max_seq_length - seq_length) * [-100]


                    model_inputs["input_ids"].append(input_ids[:args.max_seq_length])
                    model_inputs["labels"].append(labels[:args.max_seq_length])

            return model_inputs


    elif args.models == 'GLM-130B':
        from sat.model.official import GLM130B as MODEL
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)

        def preprocess_function_train(examples):
            prefix = args.source_prefix if args.source_prefix is not None else ""

            model_inputs = {
                "input_ids": [],
                "labels": [],
            }
            for i in range(len(examples[args.prompt_column])):
                if examples[args.prompt_column][i] and examples[args.response_column][i]:
                    prompt, answer = examples[args.prompt_column][i], examples[args.response_column][i]
                    prompt = prefix + prompt
                    a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                    b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

                    seq_length = len(a_ids) + len(b_ids) + 2

                    input_id = a_ids + [130001, 130004] + b_ids + [130005]
                    label = [-20100] * (len(a_ids) + 1) + b_ids + [130005]

                    input_ids = [i + 20000 for i in input_id] + (args.max_seq_length - seq_length) * [20003]
                    labels = [i + 20000 for i in label] + (args.max_seq_length - seq_length) * [-100]

                    if 150004 in input_ids[:args.max_seq_length-1]:
                        model_inputs["input_ids"].append(input_ids[:args.max_seq_length])
                        model_inputs["labels"].append(labels[:args.max_seq_length])

            return model_inputs

    elif args.models in ['llama-7B', 'llama-13B', 'llama-33B', 'llama-65B']:
        from sat.model.official import LLaMAModel as MODEL
        tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf', trust_remote_code=True)

    else:
        print(f'[ERROR] No Support {args.models}')
        exit(0)


    args.dataset_function = preprocess_function_train

    class FineTuneModel(MODEL):
        def __init__(self, args, transformer=None, parallel_output=True, **kw_args):
            super().__init__(args, transformer=transformer, parallel_output=parallel_output, **kw_args)
            self.args = args
            if args.use_lora:
                # If you use lora on other "normal" Transformer, just use it with head_first=False (by default)
                self.add_mixin("lora", LoraMixin(args.num_layers, args.lora_rank, qlora=False, head_first=False,
                                                 num_attention_heads=args.num_attention_heads,
                                                 hidden_size_per_attention_head=args.hidden_size // args.num_attention_heads),
                               reinit=True)
            if args.use_ptuning:
                self.add_mixin("ptuning", PTuningV2Mixin(args.num_layers, args.hidden_size // args.num_attention_heads,
                                                         args.num_multi_query_heads, args.pre_seq_len))

        @classmethod
        def add_model_specific_args(cls, parser):
            group = parser.add_argument_group('ChatGLM-finetune', 'ChatGLM finetune Configurations')
            group.add_argument('--pre_seq_len', type=int, default=8)
            group.add_argument('--lora_rank', type=int, default=10)
            group.add_argument('--use_ptuning', action="store_true")
            group.add_argument('--use_lora', action="store_true")
            group.add_argument('--batch_size', type=int, default=1)
            return super().add_model_specific_args(parser)

        def disable_untrainable_params(self):
            enable = []
            if self.args.use_ptuning:
                enable.extend(['ptuning'])
            if self.args.use_lora:
                enable.extend(['matrix_A', 'matrix_B'])

            if self.args.use_ptuning or self.args.use_lora:
                for n, p in self.named_parameters():
                    flag = False
                    for e in enable:
                        if e.lower() in n.lower():
                            flag = True
                            break
                    if not flag:
                        p.requires_grad_(False)
            else:
                for n, p in self.named_parameters():
                    p.requires_grad_(True)

    if args.models != 'GLM-130B':
        model, args = FineTuneModel.from_pretrained(args.models, args)

    else:
        for i in range(get_model_parallel_world_size()):
            if get_model_parallel_rank() == i:
                # Initialize model
                model = FineTuneModel(args).half()

                if args.from_quantized_checkpoint:
                    assert args.quantization_bit_width is not None
                    # Quantize model before moving to GPU
                    model = quantize(model, args.quantization_bit_width, lora=args.use_lora)

                # Load checkpoint
                load_checkpoint(model, args)

                if args.quantization_bit_width is not None and not args.from_quantized_checkpoint:
                    # Quantize model before moving to GPU
                    model = quantize(model, args.quantization_bit_width, lora=args.use_lora)

                model = model.to(args.device)
    args.mode = 'finetune'
    return model, tokenizer, args