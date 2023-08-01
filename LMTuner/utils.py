import random

import openai
import pynvml
import time
import json
from LMTuner.setting import *
from LMTuner.deepspeed_file import get_deepspeed
from LMTuner.dataset import LingoDataset
from transformers import AutoTokenizer, LlamaTokenizer, GPT2Tokenizer
import subprocess
import os
import math


system = """You are an AI assistant capable of training large language models. You need to have a concise conversation with the user. If there are default options in ARGS, you do not need to reset them. But when you're ready to end the conversation, please call the Stop_Conversation function. Firstly, you need to ask the user questions in order to complete all the information about ARGS and recommend suitable models and method configurations for the user. For each key in ARGS that has a non-None value, you need to ask the user for confirmation. For keys that have default values, you can recommend the default value to the user, but you should also allow the user to modify the value according to their needs. 

ARGS['model']: Default: None; Here are some available foundation models to choose from: GLM-130B, ChatGLM-6B,ChatGLM2-6B, llama-7b, lama-13b, llama-33b, llama-65b, gpt2. Please note that the selection of the model should be based on the specific number and memory information of the GPUs, combined with the training method. The GLM-130B, ChatGLM-6B, and ChatGLM2-6B models are suitable for Chinese data. If the graphics memory is insufficient, we recommend using the 6B model. If the graphics memory is sufficient, we recommend the GLM-130B model. In addition, the llama-7b, llama-13b, llama-33b, llama-65b, and gpt2 models are suitable for English data. Among them, gpt2 is the smallest model, and only requires 6G GPU memory for full parameter fine-tuning.

ARGS['data']: Default: None; You can either select one from the data list or fill in a local file address. Now you have some existing datasets, including: English Common: GAIR/lima; Chinese Medical: WENGSYX/Lingo-Medical-v1; Chinese Law: WENGSYX/Lingo-law-v1; Chinese Common: WENGSYX/Lingo-Chinese-v1. You have some ready-made datasets that can be added to ARGS ["dataset"], and also support adding user own data. You need to teach users to configure a JSONL file, where each line is a dictionary, and the dictionary keys are "input" and "output", respectively, and please append the absolute path of the dataset in ARGS['dataset'].

ARGS['method']: Default: None;  If the user does not have a preference for a specific model, recommendations can be made based on the available GPU memory. (LoRA: Low-Rank Adaptation; QLoRA: Quantized LoRA; LOMO: LOw-Memory Optimization; None: Full parameter fine-tune with AdamW) Among them, LoRA and QLoRA only fine tune a portion of the parameters in the model to reduce the required graphics memory, while LOMO is a special SGD optimizer that can fine tune all parameters under low graphics memory requirements) A 6GB GPU memory supports training of the 6B model with QLoRA mode. A 12GB GPU memory supports training of the 6B model with QLoRA/LoRA/LOMO modes or the 13B model with QLoRA mode. A 24GB GPU memory supports training of the 33B model with QLoRA mode or the 13B model with QLoRA/LoRA/LOMO modes. A 48GB GPU memory supports training of the 65B model with QLoRA mode or the 33B model with QLoRA/LoRA/LOMO modes, or full parameter fine-tuning of the 7B model (using AdamW optimizer). An 80GB GPU memory supports training of the 65B model with QLoRA/LoRA/LOMO modes or full parameter fine-tuning of the 13B model (using AdamW optimizer). Four 48GB GPU memory cards support training of the 130B model with QLoRA mode, while eight 48GB GPU memory cards support training of the 130B model with QLoRA/LoRA/LOMO modes.

ARGS['learning rate']: Default 1e-5; If using LoRA or QLoRA as the method, please set it to 2e-4. Otherwise, set it to 1e-5.

ARGS['epoch']: Defalt 10,
ARGS['lora rank']: Default 16,
ARGS['batch size']: Default 6,
ARGS['max length']: The sequence max length.
ARGS['gradient accumulation']: Default 1
ARGS['GPU Number']: Default The number of GPUs on the local machine.
ARGS['GPU Memory']: Default The memory size of the first GPU on the local machine.
ARGS['quantization']: Default None; if ARGS['method']=='QLoRA', INT4 or INT8 quantization can be used, but it is recommended to use INT4 quantization. You need to set 4 or 8 for QLoRA.

ARGS['train this machine']: Default True; Training will start automatically.
ARGS['save interval']: Default 1000; Save the weights every N steps.
ARGS['save path']: Default './checkpoints'
ARGS['gradient accumulation']: Default 1; Gradient accumulation steps during training.
ARGS['rope scaling']: Default False; Enabling rope scaling allows for positional interpolation of Llama-series models, enabling larger sequence lengths (up to 8000 tokens). However, it should be noted that this approach requires longer training times.
Once all the values for ARGS have been confirmed, you should call the function to indicate the final values of ARGS, but don't let the user know about ARGS and the function you are calling.

You also need to ask whether the model should be trained on the local machine. If GPU Number=0, then training on the local machine is not allowed, and GPU recommendations for a server should be made based on the user's model: RTX3090 memory=24G/RTX4090 memory=24G/A6000 memory=48G/A100 memory=80G."""

def GPT4(ARGS, text, history=[]):

    message = [
        {"role": "system",
         "content": system + "\n\nARGS = " + str(ARGS)}
    ]
    message.extend(history)
    if text != None:
        message.append({"role": "user",
                        "content": text})
        history.append({"role": "user",
                        "content": text})

    result = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=message,
        function_call="auto",
        functions=[
            {
                "name": "Set_ARGS",
                "description": "Change ARGS by specifying Key and Value",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Key": {
                            "type": "string",
                            "description": "the Key of ARGS"
                        },
                        "Value": {
                            "type": "string",
                            "description": "the Value of ARGS"
                        }
                    },
                    "required": ["Key", 'Value']
                }
            },
            {
                "name": "Stop_Conversation",
                "description": "Call this function to end the conversation when asking the user is done and all ARGS values are not None.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stop": {
                            "type": "string",
                            "description": "Input any string to indicate stopping the conversation."
                        },
                    },
                    "required": ["stop"]
                }
            }
        ]
    )

    if result['choices'][0]['message']['content'] != None:
        history.append({'role': "assistant",
                        'content': result['choices'][0]['message']['content']})
    return result['choices'][0]['message']['content'], history, result


def print_stream(text, sleep=0.02):
    for i in text:
        print(i, end='', flush=True)
        time.sleep(0.008)
    print('\n', end='')


def let_lingo_choice(ARGS):

    print_stream(
        '\033[0;33m[INFO] I see that you"re not too keen on chatting with me, but that"s alright. Let me ask you some questions next as part of the basic setup.\033[0m')
    print('')

    Model_Question = ['[CHOICE] Which model do you want to train?','1.GLM-130B','2.ChatGLM-6B','3.ChatGLM2-6B','4.Llama-7B','5.Llama-13B','6.Llama-33B','7.Llama-65B','8.Llama2-7B','9.Llama2-13B','10.Llama2-70B']
    Model_Choices = {'1':'GLM-130B','2':'ChatGLM-6B','3':'ChatGLM2-6B','4':'Llama-7B','5':'Llama-13B','6':'Llama-33B','7':'Llama-65B','8':'Llama2-7B','9':'Llama2-13B','10':'Llama2-70B'}
    ARGS['model'] = choice(Model_Question,Model_Choices)

    Data_Question = ['[CHOICE] May I know on which dataset you would like to train? We have provided the following pre-existing data.','1.English','2.Chinese','3.Chinese Medical','4.Chinese Law','5.Custom']
    Data_Choices = {'1':'GAIR/lima','2':'WENGSYX/LMTuner-dataset-v1','3':'WENGSYX/Lingo-Medical-v1','4':'WENGSYX/Lingo-law-v1','5':'Custom'}
    ARGS_data = choice(Data_Question,Data_Choices)
    if ARGS_data == 'Custom':
        print_stream('\033[0;36m[INFO] you need to configure a JSONL file, where each line is a dictionary, and the dictionary keys are "input" and "output", respectively, and please input the absolute path of the dataset.\033[0m')
        path = input('[Answer] : ')
        sueecss, messages, number = check_data(ARGS,path)
        if sueecss:
            print_stream(f'\033[0;36m[INFO] The following is the sequence length distribution of the dataset you provided:\033[0m')
            for message in messages:
                print_stream(f'\033[0;36m{message}\033[0m')
            print_stream(f'\033[0;36m[INFO] We suggest you set the sequence length to {number}.\033[0m')
            ARGS['max length'] = number
        else:
            print_stream(messages)
        ARGS['data'] = path
    else:
        sueecss, messages, number = check_data(ARGS, ARGS_data)
        if sueecss:
            print_stream(f'\033[0;36m[INFO] The following is the sequence length distribution of the dataset you provided:\033[0m')
            for message in messages:
                print_stream(f'\033[0;36m{message}\033[0m')
            print_stream(f'\033[0;36m[INFO] We suggest you set the sequence length to {number}.\033[0m')
            ARGS['max length'] = number
        else:
            print_stream(messages)
        ARGS['data'] = ARGS_data
    print('')

    Method_Question = ['[CHOICE] Do you want to use parameter-efficient fine-tuning or memory-efficient fine-tuning?','1. Tiger Optimizer (It has the best performance, but may require more memory.)','2. LoRA (Parameter-efficient)','3. QLoRA (Parameter-efficient)','4. LOMO (Memory-efficient)']
    Method_Choices = {'1':'','2':'LoRA','3':'QLoRA','4':'LOMO'}
    ARGS['method'] = choice(Method_Question,Method_Choices)

    for key in ['GPU Number','learn rate','epoch','lora rank','batch size','max length','train this machine','save interval','save path','gradient accumulation','rope scaling']:
        print_stream(f'\033[0;36m[CHOICE] Please tell me the\033[0m {key}\033[0;36m. It is recommended to: \033[0m{ARGS[key]}')
        input_text = input('[Answer] : ')
        if input_text:
            if input_text in ['True','False']:
                input_text = {'True':True,'False':False}
            ARGS[key] = input_text
        print('')
    ARGS['train continue'] = False
    return ARGS

def check_data(ARGS,path):
    try:
        if path in LINGO_SUPPORT_DATASET:
            data = LingoDataset(path)
            data = data.turn_conversations_to_io()
        else:
            data = [json.loads(i) for i in open(path, encoding='utf-8').readlines()]
    except Exception as e:
        # 捕获异常并打印错误信息
        return 0,'[Warm] Format error or file not found locally.',None

    try:
        tokenizer = get_tokenizer(ARGS)
        random.shuffle(data)
        number = []
        for d in data[:1000]:
            number.append(len(tokenizer(d['input']+d['output']).input_ids))

        max_num = max(number)
        min_num = min(number)
        interval = math.ceil((max_num - min_num) / 10)
        count = [0] * 10
        for num in number:
            idx = math.floor((num - min_num) / interval)
            if idx == 10:
                idx = 9
            count[idx] += 1

        message_num = []
        for i in range(10):
            message_num.append(len(f"{i * interval} - {min(max_num, (i + 1) * interval)}"))
        message_num_max = max(message_num)

        message = []
        for i in range(10):
            message.append(f"{i * interval} - {min(max_num, (i + 1) * interval)}{' ' * (message_num_max - message_num[i])}: {'█' * int(count[i]/20)}")

        number.sort(reverse=True)
        return 1,message,number[70]+15
    except Exception as e:
        # 捕获异常并打印错误信息
        return 0, f'[Warm] {e}', None

def choice(questions,choices):
    for i in questions:
        print_stream(f'\033[0;36m{i}\033[0m')

    while True:
        input_text = input('[Answer] : ')
        if input_text in choices:
            print('')
            return choices[input_text]
        print_stream(f'[INFO] Please input the number, such as: 1')


def let_lingo_conversation(ARGS):
    print_stream(
        '\033[0;36m[AI] Hello there! I"m your AI assistant, and I"m here to help you train your model. Before we get started, it"s important to have a clear plan and goal in mind. \033[0m')
    data_input = input('[Answer] : ')

    history = []
    Now_Break = False
    while True:
        response, history, result = GPT4(ARGS, data_input, history)
        if result['choices'][0]['finish_reason'] == 'function_call':
            if result['choices'][0]['message']['function_call']['name'] == 'Stop_Conversation':
                Now_Break = True
            else:
                try:
                    dict_json = json.loads(result['choices'][0]['message']['function_call']['arguments'])
                    ARGS[dict_json['Key']] = dict_json['Value']
                    if dict_json['Key'] == 'data':
                        sueecss, messages, number = check_data(ARGS, dict_json['Value'])
                        if sueecss:
                            print_stream(
                                f'\033[0;36m[INFO] The following is the sequence length distribution of the dataset you provided:\033[0m')
                            for message in messages:
                                print_stream(f'\033[0;36m{message}\033[0m')
                            print_stream(f'\033[0;36m[INFO] We suggest you set the sequence length to {number}.\033[0m')
                            ARGS['max length'] = number
                        else:
                            print_stream(messages)
                        print()
                        ARGS['max length'] = number

                except:
                    pass
                history.append({'role': "function", 'name': 'Set_ARGS',
                                'content': 'ARGS["{}"] = {}\n It"s ok'.format(dict_json['Key'], dict_json['Value'])})
                data_input = None
        else:
            responses = response.split('\n')
            for response_item in responses:
                print_stream(
                    '\033[0;36m[AI] {} \033[0m'.format(response_item))
            data_input = input('[Answer] : ')


        if Now_Break:
            ARGS['train continue'] = False
            break
    return ARGS


def launch_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
    return p.returncode


def get_cmd(ARGS):
    cmd = "deepspeed "
    glm_130b_python_code = ''
    wandb_python_code = ''

    ARGS['GPU Number'] = int(ARGS['GPU Number'])
    if ARGS['model'] == 'GLM-130B' and ARGS['GPU Number'] >= 4:
        cmd += '--include localhost:{} '.format(','.join([str(i) for i in list(range(ARGS['GPU Number'] // 4 * 4))]))
    else:
        cmd += '--include localhost:{} '.format(','.join([str(i) for i in list(range(ARGS['GPU Number']))]))

    if ARGS['method']=='QLoRA' and ARGS['quantization'] not in [4,8]:
        ARGS['quantization'] = 4

    cmd += "main.py --seed 1234 "
    if ARGS['model'] == 'GLM-130B':
        print_stream(
            '\033[0;36m[AI] Due to the need to manually download the weight of GLM-130B, please follow my instructions (See in https://github.com/THUDM/GLM-130B#model-weights)\033[0m')
        print_stream(
            '\033[0;36m[AI] Download the GLM-130B checkpoint from: https://docs.google.com/forms/d/e/1FAIpQLSehr5Dh_i3TwACmFFi8QEgIVNYGmSPwV0GueIcsUev0NEfUug/viewform?usp=sf_link \033[0m')
        print('')



        print_stream('\033[0;36m[AI] Merge them and extract it: \033[0m')
        print_stream('\033[0;36m[AI] cat glm-130b-sat.tar.part_* > glm-130b-sat.tar \033[0m')
        print_stream('\033[0;36m[AI] tar xvf glm-130b-sat.tar \033[0m')
        print('')

        if ARGS['GPU Number'] >= 8:
            cmd += '--model-parallel-size {} '.format(8)
            target_tp = 8
        elif ARGS['GPU Number'] >= 4:
            cmd += '--model-parallel-size {} '.format(4)
            target_tp = 4
        elif ARGS['GPU Number'] >= 2:
            cmd += '--model-parallel-size {} '.format(2)
            target_tp = 2
        else:
            target_tp = 1

        if ARGS['method']=='QLoRA':
            qlora = f'--quantization-bit-width {ARGS["Quantization"]}'
            cmd += '--from-quantized-checkpoint '
        else:
            qlora = ''

        if ARGS['train continue'] == False:
            print_stream('\033[0;36m[AI] Can you tell me the absolute location of the glm-130b-sat folder? \033[0m')
            input_folder = input('[Answer] : ')
            print_stream(
                '\033[0;36m[AI] We are going to convert the model weights next. You need to tell me the absolute location of the new file. \033[0m')
            output_folder = input('[Answer] : ')

            print_stream(
                '\033[0;36m[AI] Let"s conversion the checkpoint to change the tensor parallel (tools/convert_tp.py -> https://github.com/THUDM/GLM-130B): \033[0m')

            print_stream(
                f'\033[0;36m[AI] python tools/convert_tp.py --input-folder {input_folder} --output-folder {output_folder} --target-tp {target_tp} {qlora} \033[0m')
            glm_130b_python_code = f'python tools/convert_tp.py --input-folder {input_folder} --output-folder {output_folder} --target-tp {target_tp} {qlora}'
            cmd += f'--load {output_folder} '
            ARGS['load'] = output_folder
            print_stream('\033[0;36m[AI] If you have completed these operations, please let me know at any time. \033[0m')
            input('[Answer] : ')
            print('')
        else:
            cmd += f'--load {ARGS["load"]} '

        cmd += f'--num-layers 70 --hidden-size 12288 --inner-hidden-size 32768 --vocab-size 150528 --layernorm-order post --num-attention-heads 96 --models GLM-130B --mode finetune '


        log_interval = 1
    else:
        cmd += '--model {} '.format(ARGS['model'])
        log_interval = 10

        if 'load' in ARGS:
            if ARGS['load'] != '':
                cmd += '--load {} '.format(ARGS['load'])
        else:
            ARGS['load'] = ''

    if ARGS['method'] == 'LoRA':
        cmd += '--use_lora 1 --lora_rank {} '.format(ARGS['lora rank'])
    if ARGS['method'] == 'QLoRA':  # TODO 使用QLoRA时，CPU-offload必须设置为false
        cmd += '--use_lora 1 --quantization-bit {} --lora_rank {} '.format(ARGS['quantization'], ARGS['lora rank'])
    if ARGS['method'] == 'LOMO':  # TODO LOMO必须要ZeRO-Stage3和bf16
        cmd += '--use_lomo 1 '
    if ARGS['rope scaling'] == True:
        cmd += '--rope_scaling True '


    if ARGS['train continue'] == False:
        print_stream(
            '\033[0;36m[AI] By the way, do you want to use Wandb to record logs? \033[0m')
        print_stream(
            '\033[0;36m[AI] If so, please provide Wandb"s API KEY, which may be located in https://wandb.ai/settings. Of course, if you enter No, we can skip this step \033[0m')
        wandb_api = input(
            '[Answer] : ')
        if len(wandb_api) >= 10:
            if ARGS['train this machine'] == True or ARGS['train this machine'] == 'True':

                if launch_cmd(f'wandb login {wandb_api}') == 0:
                    cmd += '--wandb 1'
                    ARGS['wandb'] = wandb_api
                else:
                    ARGS['wandb'] = False
                    print_stream(
                        '\033[0;36m[WARM] The wandb raise ERROR, but we can continue without wandb. \033[0m')
            else:
                wandb_python_code = f'wandb login {wandb_api}'
        else:
            ARGS['wandb'] = False

    elif 'wandb' in ARGS:
        if ARGS['wandb']:
            cmd += '--wandb 1'



    cmd += f'--fp16 --dataset {ARGS["data"]} --train-data {ARGS["data"]} --valid-data {ARGS["data"]} --max_seq_length {ARGS["max length"]} --epochs {ARGS["epoch"]}  --train-iters 0  --no-load-rng --warmup .02 --checkpoint-activations --save-interval {ARGS["save interval"]} --save "{ARGS["save path"]}" --split 1 --eval-interval 1000000 --eval-batch-size 2 --lr {ARGS["learning rate"]} --num-workers 0 --log-interval {log_interval} '

    cmd += '--deepspeed_config ./ds_config.json'

    ds_config = get_deepspeed(ARGS)
    with open('./ds_config.json','w',encoding='utf-8') as f:
        f.write(ds_config)

    return cmd,glm_130b_python_code,wandb_python_code,ARGS

def write_readme(ARGS,cmd,glm_130b_python_code,wandb_python_code):
    readme_lines = []

    readme_lines.append('# Train the model on another server using Lingo')
    readme_lines.append('***Welcome to Lingo***, we can provide you with various training methods you want')
    readme_lines.append('')
    readme_lines.append('### Installation')
    readme_lines.append(
        'For details, please refer to [Github Guide](https://github.com/WENGSYX/Lingo/blob/main/QA/readme.md)')
    readme_lines.append('')

    readme_lines.append('### Operation')
    if ARGS['model'] == 'GLM-130B':
        readme_lines.append('#### Download GLM-130B')
        readme_lines.append(
            'You can download the weights of GLM-130B here: https://docs.google.com/forms/d/e/1FAIpQLSehr5Dh_i3TwACmFFi8QEgIVNYGmSPwV0GueIcsUev0NEfUug/viewform?usp=sf_link')
        readme_lines.append('')
        readme_lines.append('#### Merge GLM-130B')
        readme_lines.append('Then you need to merge the compressed files and unzip them into a folder')
        readme_lines.append('```bash')
        readme_lines.append('cat glm-130b-sat.tar.part_* > glm-130b-sat.tar')
        readme_lines.append('tar xvf glm-130b-sat.tar')
        readme_lines.append('```')
        readme_lines.append('')
        readme_lines.append('#### Convert GLM-130B weights')
        readme_lines.append('Finally, you need to convert the GLM-130B weights as needed:')
        readme_lines.append('```bash')
        readme_lines.append(glm_130b_python_code)
        readme_lines.append('```')
        readme_lines.append('')
    if wandb_python_code:
        readme_lines.append('#### Login wandb')
        readme_lines.append('Please run the following code in shell to login wandb')
        readme_lines.append('```bash')
        readme_lines.append(wandb_python_code)
        readme_lines.append('```')
    readme_lines.append('##### Copy ds_config')
    readme_lines.append(
        'Please move the originally generated ***ds_config.json*** to the now training path.')
    readme_lines.append('')
    readme_lines.append('##### Training operation')
    readme_lines.append('Please manually execute the following code on the server to start training:')
    readme_lines.append('```bash')
    readme_lines.append(cmd)
    readme_lines.append('```')
    readme_lines.append('')
    readme_finally = '### Cite\n\nThis project is an accompanying project of [Neural Comprehension](https://github.com/WENGSYX/Neural-Comprehension). If you are interested in our project, please feel free\nto quote.\n\n```\n@misc{weng2023mastering,\n      title={Mastering Symbolic Operations: Augmenting Language Models with Compiled Neural Networks}, \n      author={Yixuan Weng and Minjun Zhu and Fei Xia and Bin Li and Shizhu He and Kang Liu and Jun Zhao},\n      year={2023},\n      eprint={2304.01665},\n      archivePrefix={arXiv},\n      primaryClass={cs.CL}\n}\n```\n\n### Disclaimers\n**The resources related to this project are for academic research purposes only and are strictly prohibited from commercial use** When using parts involving third-party code, please strictly follow the corresponding open source protocol. The content generated by the model is affected by factors such as model calculation, randomness, and loss of quantification accuracy. This project does not guarantee its accuracy. For any content output by the model, this project does not assume any legal responsibility, nor is it liable for any losses that may arise from the use of relevant resources and output results\n'

    readme_lines.append(readme_finally)

    return readme_lines

def get_tokenizer(args):
    if args['model'] == 'ChatGLM-6B':
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)
    elif args['model'] == 'ChatGLM2-6B':
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm2-6b', trust_remote_code=True)
    elif args['model'] == 'GLM-130B':
        tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm-6b', trust_remote_code=True)

    elif args['model'].lower() in ['llama-7b', 'llama-13b', 'llama-33b', 'llama-65b']:
        tokenizer = LlamaTokenizer.from_pretrained('decapoda-research/llama-7b-hf', trust_remote_code=True)

    elif args['model'].lower() in ['llama2-7b', 'llama2-13b', 'llama2-70b']:
        tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', trust_remote_code=True)

    elif args['model'] in ['gpt2']:
        tokenizer = AutoTokenizer.from_pretrained('gpt2')

    elif args['model'] in ['gpt-neo-1.3b']:
        tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')

    else:
        print(f'[ERROR] No Support {args["model"]}')
        exit(0)
    return tokenizer