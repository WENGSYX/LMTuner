from LMTuner import Let_Tune, get_cmd, launch_cmd
from LMTuner.models import get_model_and_tokenizer
from LMTuner.initialize import initialize
import json
import argparse
import os
import math
from tqdm import trange



if __name__ == '__main__':
    def add_generation_specific_args(py_parser):
        py_parser.add_argument('--max_source_length', type=int)
        py_parser.add_argument('--max_target_length', type=int)
        py_parser.add_argument('--max_seq_length', type=int)
        py_parser.add_argument('--ignore_pad_token_for_loss', type=bool, default=True)
        py_parser.add_argument('--source_prefix', type=str, default="")
        py_parser.add_argument('--prompt_column', type=str, default='input')
        py_parser.add_argument('--response_column', type=str, default='output')

        py_parser.add_argument('--pre_seq_len', type=int, default=8)
        py_parser.add_argument('--lora_rank', type=int, default=10)
        py_parser.add_argument('--use_ptuning', action="store_true")
        py_parser.add_argument('--use_lora', type=bool, default=False)
        py_parser.add_argument('--use_lomo', type=bool, default=False)
        py_parser.add_argument('--lora_save', type=str, default='')
        py_parser.add_argument('--lora_load', type=str, default='')

        py_parser.add_argument('--models', type=str, default="")
        py_parser.add_argument('--dataset', type=str, default="")
        py_parser.add_argument('--finetune', type=bool, default=True)
        py_parser.add_argument('--wandb', type=bool, default=False)
        py_parser.add_argument('--quantization_bit', type=int, default=0)

        py_parser.add_argument('--rope_scaling', type=bool, default=False)
        py_parser.add_argument('--ARGS', type=str, default='')
        py_parser.add_argument('--Inference', type=bool, default=True)
        py_parser.add_argument('--strategy', type=str, default='greedy_search')
        py_parser.add_argument('--generate_file',type=str,default='')

        return py_parser


    args = initialize(extra_args_provider=add_generation_specific_args)

    args.strategy = 'beam_serach'
    args.top_p = 0.95
    args.top_k = 10
    args.temperature = 0.8
    model, tokenizer, args = get_model_and_tokenizer(args)

    if args.generate_file != '':
        if os.path.exists(args.generate_file):
            data = [json.loads(i)['input'] for i in open(args.generate_file,encoding='utf-8').readlines()]
        else:
            data = ['Hello']

        responses = []
        response = model.generate(data, tokenizer)
        responses.extend(response)

        with open('./output.jsonl','w',encoding='utf-8') as f:
            for i in range(len(responses)):
                f.write(json.dumps({'input':data[i],'output':responses[i]}))
                f.write('\n')

    else:
        while True:
            inputs = input('[INPUT] : ')
            print(model.generate([inputs], tokenizer))
