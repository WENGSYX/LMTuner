import os
import torch
from sat import AutoModel
from transformers import LlamaTokenizer, AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.generation.autoregressive_sampling import filling_sequence
from sat.generation.sampling_strategies import BaseStrategy, BeamSearchStrategy
from lingo.models import get_model_and_tokenizer
from lingo.models.model_io import load_checkpoint
from sat.mpu import get_model_parallel_world_size, get_model_parallel_rank, get_model_parallel_group, \
    initialize_model_parallel
from typing import Optional


def _format_data(query: str, history: Optional[list], prefix: Optional[str] = "") -> str:
    r"""
    Supports: https://huggingface.co/tatsu-lab/alpaca-7b-wdiff
    """
    prompt = prefix
    if history:
        for old_query, response in history:
            prompt += "Human: {}\nAssistant: {}\n".format(old_query, response)
    prompt += "Human: {}\nAssistant: ".format(query)
    return prompt


def generate_text(model, tokenizer, prompts,
                  num_beams=1, max_length: int = 300, top_p=0.7, top_k=0, temperature=0.95, strategy=None):

    if strategy == 'beam_serach':
        strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=top_k,
                                      end_tokens=[tokenizer.eos_token_id], num_beams=num_beams, consider_end=True)

    else:
        strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[tokenizer.eos_token_id])

    for prompt in prompts:
        inputs = \
            tokenizer([prompt], return_tensors="pt").to(model.parameters().__next__().device)[
                'input_ids'][0]
        seq = torch.cat(
            [inputs, torch.tensor([-1] * (max_length - len(inputs)), device=inputs.device)], dim=0
        ).to(get_model_parallel_rank())

        output = filling_sequence(
            model, seq,
            batch_size=1,
            strategy=strategy
        )[0]

        output_list = list(output)

        response = tokenizer.decode(output_list[0])
        print(f"{response}")
        with open(prompt,'w',encoding='utf-8') as f:
            f.write(response)


    return response


def chat(model, tokenizer, prompt,
         num_beams=1, max_length: int = 256, top_p=0.7, top_k=0, temperature=0.95, strategy=None):
    inputs = \
    tokenizer([prompt], return_tensors="pt").to(model.parameters().__next__().device)[
        'input_ids'][0]
    seq = torch.cat(
        [inputs, torch.tensor([-1] * (max_length - len(inputs)), device=inputs.device)], dim=0
    ).to(get_model_parallel_rank())
    if strategy == 'beam_serach':
        strategy = BeamSearchStrategy(temperature=temperature, top_p=top_p, top_k=top_k,
                                      end_tokens=[tokenizer.eos_token_id], num_beams=num_beams, consider_end=True)

    else:
        strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[tokenizer.eos_token_id])

    output = filling_sequence(
        model, seq,
        batch_size=1,
        strategy=strategy
    )[0]

    output_list = list(output)

    response = tokenizer.decode(output_list[0])
    print(f"Assistant: {response.replace(prompt, '')}")
    return response


def predict_and_print(model, tokenizer, query, history: list):
    prompt = _format_data(query, history)

    response = ""

    response = chat(model, tokenizer, prompt, max_length=args.max_length, num_beams=args.num_beams, top_p=args.top_p,
                    temperature=args.temperature, top_k=args.top_k)
    print()
    history = history + [(query, response)]
    return history


def read_json(path):
    import json
    with open(path, 'r') as f:
        return json.load(f)


def main(args):
    # load model
    model, tokenizer, args = get_model_and_tokenizer(args)
    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())
    batch = tokenizer(
        "你好",
        return_tensors="pt",
    )
    batch['position_ids'] = torch.arange(batch['input_ids'].shape[1]).unsqueeze(0)
    batch = {k: v.cuda() for k, v in batch.items()}

    logits = model(**batch)
    print(logits)
    torch.save(logits,'sat_logit.pt')

    #model.to(get_model_parallel_rank())

    if args.chat:
        history = []
        print("Input for chat， input clear to clear the history，stop to end the program.")
        while True:
            try:
                query = input("\nInput: ")
            except UnicodeDecodeError:
                print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.")
                continue
            except Exception:
                raise

            if query.strip() == "stop":
                break

            if query.strip() == "clear":
                history = []
                print("History has been removed.")
                continue

            try:
                history = predict_and_print(model, tokenizer, query, history)
            except IndexError:
                print("The model is unable to generate a response beceaue the query is too long.")
                history = []
                print('History has been removed.')

    else:
        prompts = [
            # For these prompts, the expected answer is the natural continuation of the prompt
            "The capital of China is",
            "Simply put, the theory of relativity states that ",
            """你好，中国。你好，浙江。你好，""",
            # Few shot prompt (providing a few examples before asking model to complete more);
            """Translate English to Chinese:

            sea otter => 海獭
            peppermint => 薄荷
            plush girafe => 长颈鹿
            cheese =>""",
            "问：用鲁迅的风格，以“今天的紫菜汤有点咸了”开头，写一首四行诗\n答：",
            "问：研究红楼梦我该学习的五个要点是什么？\n答：",
        ]

        generate_text(model, tokenizer, prompts)


if __name__ == "__main__":
    from lingo.initialize import initialize
    def add_generation_specific_args(py_parser):
        py_parser.add_argument('--max_source_length', type=int)
        py_parser.add_argument('--max_target_length', type=int)
        py_parser.add_argument('--max_seq_length', type=int)
        py_parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True)
        py_parser.add_argument('--source_prefix', type=str, default="")
        py_parser.add_argument('--prompt_column', type=str,default='input')
        py_parser.add_argument('--response_column', type=str,default='output')

        py_parser.add_argument('--pre_seq_len', type=int, default=8)
        py_parser.add_argument('--lora_rank', type=int, default=10)
        py_parser.add_argument('--use_ptuning', action="store_true")
        py_parser.add_argument('--use_lora', type=bool,default=False)
        py_parser.add_argument('--use_lomo', type=bool,default=False)

        py_parser.add_argument('--models', type=str, default="")
        py_parser.add_argument('--dataset', type=str, default="")
        py_parser.add_argument('--finetune', type=bool, default=True)
        py_parser.add_argument('--wandb', type=bool, default=False)
        py_parser.add_argument('--quantization_bit', type=int, default=0)

        py_parser.add_argument("--max_length", type=int, default=512)
        py_parser.add_argument('--chat', type=bool, default=False)

        py_parser.add_argument('--rope_length_generalization', type=bool, default=False)

    args = initialize(extra_args_provider=add_generation_specific_args)
    args.num_beams = 1
    args.top_p = 0.95
    args.top_k = 10
    args.temperature = 0.8
    args.mode = 'inference'
    args.max_length = 300
    main(args)