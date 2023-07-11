import openai
import pynvml
import time
import json
from lingo.setting import *




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

    print(ARGS)


def get_cmd(ARGS):
    cmd = "deepspeed "
    if ARGS['model'] == 'GLM-130B' and ARGS['GPU Number'] >= 4:
        cmd += '--include localhost:{} '.format(','.join([str(i) for i in list(range(ARGS['GPU Number'] // 4))]))
    else:
        cmd += '--include localhost:{} '.format(','.join([str(i) for i in list(range(ARGS['GPU Number']))]))

    cmd += "main.py --seed 1234 "
    if ARGS['model'] == 'GLM-130B':
        if ARGS['GPU Number'] >= 8:
            cmd += '--model-parallel-size {} '.format(8)
        elif ARGS['GPU Number'] >= 4:
            cmd += '--model-parallel-size {} '.format(4)
        elif ARGS['GPU Number'] >= 2:
            cmd += '--model-parallel-size {} '.format(2)

        cmd += '--num-layers 70 --hidden-size 12288 --inner-hidden-size 32768 --vocab-size 150528 --num-attention-heads 96 '
        log_interval = 1
    else:
        cmd += '--model {} '.format(ARGS['model'])
        log_interval = 10

    if ARGS['method'] == 'LoRA':
        cmd += '--use_lora --lora_rank {} '.format(ARGS['lora_rank'])
    if ARGS['method'] == 'QLoRA':
        cmd += '--use_lora --quantization-bit-width {} --lora_rank {} '.format(ARGS['Quantization'], ARGS['lora_rank'])
    if ARGS['method'] == 'LOMO':
        cmd += '--use_lomo '

    cmd += f'--fp16 --dataset {"WENGSYX/"+ARGS["data"]} --train-data {ARGS["data"]} --valid-data {ARGS["data"]} --max_seq_length {ARGS["max length"]} --no-load-rng --warmup .02 --checkpoint-activations --save-interval {ARGS["save interval"]} --save "{ARGS["save path"]}" --split 1 --eval-interval 10 --eval-batch-size 2 --zero-stage 1 --lr {ARGS["learn rate"]} --num-workers 0 --log-interval {log_interval}'
    return cmd


if __name__ == '__main__':
    let_lingo()
