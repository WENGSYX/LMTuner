import openai
import pynvml
import time
import json
from lingo.setting import *
from lingo.deepspeed_file import get_deepspeed
import subprocess




def GPT4(ARGS, text, history=[]):
    system = 'You are an AI assistant capable of training large language models. Firstly, you need to ask the user questions in order to complete all the information about ARGS and recommend suitable models and method configurations for the user. For each key in ARGS that has a non-None value, you need to ask the user for confirmation. For keys that have default values, you can recommend the default value to the user, but you should also allow the user to modify the value according to their needs. \n\n'
    system += "Once all the values for ARGS have been confirmed, you should call the function to indicate the final values of ARGS, but don't let the user know about ARGS and the function you are calling.\n\n"
    system += 'Now you have some existing datasets, including:\nCommon: Lingo-Dataset-v1\nMedical: Lingo-Medical-v1\nLaw: Lingo-law-v1\nChinese: Lingo-Chinese-v1\nEnglish: Lingo-English-v1\n\nYou have some ready-made datasets that can be added to ARGS ["dataset"], and also support adding your own data. You need to teach users to configure a JSONL file, where each line is a dictionary, and the dictionary keys are "input" and "output", respectively, and please append the absolute path of the dataset in ARGS[\'dataset\']'
    system += 'Here are some available models to choose from: \nGLM-130B, \nChatGLM-6B,\nChatGLM2-6B, \nllama-7b, \nlama-13b, \nllama-33b, \nllama-65b.\n If the user does not have a preference for a specific model, recommendations can be made based on the available GPU memory. (LoRA: Low-Rank Adaptation; QLoRA: Quantized LoRA; LOMO: LOw-Memory Optimization. Please use code \'ARGS[\'QLoRA\'] = True\' to set those.) A 6GB GPU memory supports training of the 6B model with QLoRA mode. A 12GB GPU memory supports training of the 6B model with QLoRA/LoRA/LOMO modes or the 13B model with QLoRA mode. A 24GB GPU memory supports training of the 33B model with QLoRA mode or the 13B model with QLoRA/LoRA/LOMO modes. A 48GB GPU memory supports training of the 65B model with QLoRA mode or the 33B model with QLoRA/LoRA/LOMO modes, or full parameter fine-tuning of the 7B model (using AdamW optimizer). An 80GB GPU memory supports training of the 65B model with QLoRA/LoRA/LOMO modes or full parameter fine-tuning of the 13B model (using AdamW optimizer). Four 48GB GPU memory cards support training of the 130B model with QLoRA mode, while eight 48GB GPU memory cards support training of the 130B model with QLoRA/LoRA/LOMO modes.\n\n'
    system += "Once all the values for ARGS['method'] have been confirmed, you should call the function to indicate the final values of ARGS, but don't let the user know about ARGS and the function you are calling."
    system += "At the beginning, ARGS['GPU Number'] represents the number of GPUs on the local machine, and ARGS['GPU Memory'] represents the memory size of the first GPU on the local machine. You also need to ask whether the model should be trained on the local machine. If GPU Number=0, then training on the local machine is not allowed, and GPU recommendations for a server should be made based on the user's model: RTX3090 memory=24G/RTX4090 memory=24G/A6000 memory=48G/A100 memory=80G.\n\n"
    system += "if ARGS['method']=='QLoRA', INT4 or INT8 quantization can be used, but it is recommended to use INT4 quantization. The memory usage of INT4 is half that of INT8 or 1/4 of normal FP16 training. If QLoRA is selected, please be sure \"ARGS['Quantization']=4\" or \"ARGS['Quantization']=8\".\n\n"
    system += "Please ask your questions slowly and avoid asking too many questions at once. Also, please provide assistance to users in a friendly and engaging manner.\n\n"

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
            }
        ]
    )

    if result['choices'][0]['message']['content'] != None:
        history.append({'role': "assistant",
                        'content': result['choices'][0]['message']['content']})
    return result['choices'][0]['message']['content'], history, result


text = r""""""


def print_stream(text, sleep=0.02):
    for i in text:
        print(i, end='', flush=True)
        time.sleep(0.02)
    print('\n', end='')


def let_lingo():
    try:
        pynvml.nvmlInit()

        ARGS = {'data': None,
                'model': None,
                'method': None,
                'learn rate': 1e-5,
                'epoch': 10,
                'lora_rank': 16,
                'batch size': 6,
                'max length': 1024,
                'GPU Number': pynvml.nvmlDeviceGetCount(),
                'GPU Memory': str(round(pynvml.nvmlDeviceGetMemoryInfo(0).free / (1024 * 1024), 2)) + 'MB',
                'Quantization': None,
                'Train_This_Machine': None,
                'save interval': 1000,
                'save path': './checkpoints'
                }
    except:
        ARGS = {'data': None,
                'model': None,
                'method': None,
                'learn rate': 1e-5,
                'epoch': 10,
                'lora_rank': 16,
                'batch size': 6,
                'max length': 1024,
                'GPU Number': 4,
                'GPU Memory': '24GB',
                'Quantization': 0,
                'Train_This_Machine': None,
                'save interval': 1000,
                'save path': './checkpoints'
                }

    print_stream('\033[0;33m[INFO] This is a library for training language models with ease. \033[0m', 0.005)
    print_stream(
        '\033[0;33m[INFO] In conversations with Lingo, the language model will be trained automatically according to your needs, without requiring any effort on your part 😊\033[0m',
        0.005)
    print_stream('\033[0;33m[INFO] Would you like to command Lingo through casual conversation? \033[0m')

    if openai.api_key != None:
        user_conversation = input(
            '\033[0;33m[INFO] If yes, please type (Yes), let"s go~, If not, please type (No)\033[0m')
        print('')
    else:
        user_conversation = input(
            '\033[0;33m[INFO] If yes, please provide your OpenAI API KEY. If not, please type (No)\033[0m')
        print('')

    if user_conversation == 'No':
        let_lingo_choice(ARGS)

    else:
        openai.api_key = user_conversation
        let_lingo_conversation(ARGS)


def let_lingo_choice():
    """
    print_stream(
        '\033[0;33m[INFO] I see that you"re not too keen on chatting with me, but that"s alright. Let me ask you some questions next as part of the basic setup.\033[0m')
    print('')
    print_stream('\033[0;36m[CHOICE] Which model do you want to train?\033[0m')
    print_stream('\033[0;36m1.GLM-130B\033[0m')
    print_stream('\033[0;36m2.ChatGLM-6B\033[0m')
    print_stream('\033[0;36m3.Llama-7B\033[0m')
    print_stream('\033[0;36m4.Llama-13B\033[0m')
    print_stream('\033[0;36m5.Llama-33B\033[0m')
    print_stream('\033[0;36m6.Llama-33B\033[0m')
    """
    print_stream('\033[0;36mUnsupport!\033[0m')
    exit(0)


def let_lingo_conversation(ARGS):
    print_stream(
        '\033[0;36m[AI] Hello there! I"m your AI assistant, and I"m here to help you train your model. Before we get started, it"s important to have a clear plan and goal in mind. \033[0m')
    data_input = input('\033[0;36m[Answer] \033[0m')

    history = []
    while True:
        response, history, result = GPT4(ARGS, data_input, history)
        if result['choices'][0]['finish_reason'] == 'function_call':
            try:
                dict_json = json.loads(result['choices'][0]['message']['function_call']['arguments'])
                ARGS[dict_json['Key']] = dict_json['Value']
            except:
                pass
            history.append({'role': "function", 'name': 'Set_ARGS',
                            'content': 'ARGS["{}"] = {}\n It"s ok'.format(dict_json['Key'], dict_json['Value'])})
            data_input = None
        else:
            print_stream(
                '\033[0;36m[AI] {} \033[0m'.format(response))
            data_input = input('\033[0;36m[Answer] \033[0m')

        Now_Break = True
        for value in ARGS.values():
            if None == value:
                Now_Break = False
        if Now_Break:
            break


    cmd = get_cmd(ARGS)
    if ARGS['Train_This_Machine']:
        if launch_cmd(cmd) == 0:
            print_stream(
                '\033[0;36m[AI] Now we have successed for trian the model !\033[0m')

    else:
        write_readme(ARGS,cmd)
        print_stream(
            f'\033[0;36m[AI] You will train the model in other machines with GPUs, and I have written a readme ({os.path.dirname(os.path.abspath (__file__))}/readme.md) for you. You can refer to its commands. \033[0m')




def launch_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True)
    p.wait()
    return p.returncode


def get_cmd(ARGS):
    cmd = "deepspeed "
    if ARGS['model'] == 'GLM-130B' and ARGS['GPU Number'] >= 4:
        cmd += '--include localhost:{} '.format(','.join([str(i) for i in list(range(ARGS['GPU Number'] // 4 * 4))]))
    else:
        cmd += '--include localhost:{} '.format(','.join([str(i) for i in list(range(ARGS['GPU Number']))]))

    if ARGS['method']=='QLoRA' and ARGS['Quantization'] not in [4,8]:
        ARGS['Quantization'] = 8

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
        print_stream('\033[0;36m[AI] Can you tell me the absolute location of the glm-130b-sat folder? \033[0m')
        input_folder = input('\033[0;36m[Answer] \033[0m')
        print_stream(
            '\033[0;36m[AI] We are going to convert the model weights next. You need to tell me the absolute location of the new file. \033[0m')
        output_folder = input('\033[0;36m[Answer] \033[0m')

        print_stream(
            '\033[0;36m[AI] Let"s conversion the checkpoint to change the tensor parallel (tools/convert_tp.py -> https://github.com/THUDM/GLM-130B): \033[0m')

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
        print_stream(
            f'\033[0;36m[AI] python tools/convert_tp.py --input-folder {input_folder} --output-folder {output_folder} --target-tp {target_tp} {qlora} \033[0m')
        input('\033[0;36m[AI] If you have completed these operations, please let me know at any time. \033[0m')
        print('')
        cmd += f'--num-layers 70 --hidden-size 12288 --inner-hidden-size 32768 --vocab-size 150528 --layernorm-order post --num-attention-heads 96 --models GLM-130B '
        cmd += f'--load {output_folder} '

        log_interval = 1
    else:
        cmd += '--model {} '.format(ARGS['model'])
        log_interval = 10

    if ARGS['method'] == 'LoRA':
        cmd += '--use_lora 1 --lora_rank {} '.format(ARGS['lora_rank'])
    if ARGS['method'] == 'QLoRA':  # TODO 使用QLoRA时，CPU-offload必须设置为false
        cmd += '--use_lora 1 --quantization-bit {} --lora_rank {} '.format(ARGS['Quantization'], ARGS['lora_rank'])
    if ARGS['method'] == 'LOMO':  # TODO LOMO必须要ZeRO-Stage3和bf16
        cmd += '--use_lomo 1'

    print_stream(
        '\033[0;36m[AI] By the way, do you want to use Wandb to record logs? \033[0m')
    wandb_api = input(
        '\033[0;36m[AI] If so, please provide Wandb"s API KEY, which may be located in https://wandb.ai/settings. Of course, if you enter No, we can skip this step \033[0m')
    if len(wandb_api) >= 10:
        if launch_cmd('wandb login 1234') == 0:
            cmd += '--wandb 1'
        else:
            print_stream(
                '\033[0;36m[WARM] The wandb raise ERROR, but we can continue without wandb. \033[0m')

    if ARGS['data'] in LINGO_SUPPORT_DATASET:
        dataset = "WENGSYX/" + ARGS["data"]
    else:
        dataset = ARGS['data']

    cmd += f'--fp16 --dataset {dataset} --train-data {ARGS["data"]} --valid-data {ARGS["data"]} --max_seq_length {ARGS["max length"]} --epochs {ARGS["epoch"]}  --train-iters 0  --no-load-rng --warmup .02 --checkpoint-activations --save-interval {ARGS["save interval"]} --save "{ARGS["save path"]}" --split 1 --eval-interval 1000000 --eval-batch-size 2 --lr {ARGS["learn rate"]} --num-workers 0 --log-interval {log_interval} '

    cmd += '--deepspeed_config ./ds_config.json'

    ds_config = get_deepspeed(ARGS)
    with open('./ds_config.json','w',encoding='utf-8') as f:
        f.write(ds_config)

    return cmd

def write_readme(ARGS,cmd):
    readme_lines = []
    if ARGS['language'] == 'Chinese':
        readme_lines.append('# 如何使用Lingo在另一台服务器中训练模型')
        readme_lines.append('*** 欢迎来到Lingo ***,我们能够为你提供各种你想要的训练方式')
        readme_lines.append('### 安装环境')
        readme_lines.append('***具体过程请参考[Github源码](https://github.com/microsoft/Megatron-DeepSpeed)')

if __name__ == '__main__':
    let_lingo()
