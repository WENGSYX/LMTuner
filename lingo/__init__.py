import openai
import pynvml
import time
import json
from lingo.setting import *
from lingo.deepspeed_file import get_deepspeed
import subprocess
import os



system = """You are an AI assistant capable of training large language models. Firstly, you need to ask the user questions in order to complete all the information about ARGS and recommend suitable models and method configurations for the user. For each key in ARGS that has a non-None value, you need to ask the user for confirmation. For keys that have default values, you can recommend the default value to the user, but you should also allow the user to modify the value according to their needs. 

ARGS['data']: Default: None; You can either select one from the data list or fill in a local file address. Now you have some existing datasets, including: Common: Lingo-Dataset-v1; Medical: Lingo-Medical-v1; Law: Lingo-law-v1; Chinese: Lingo-Chinese-v1; English: Lingo-English-v1. You have some ready-made datasets that can be added to ARGS ["dataset"], and also support adding user own data. You need to teach users to configure a JSONL file, where each line is a dictionary, and the dictionary keys are "input" and "output", respectively, and please append the absolute path of the dataset in ARGS['dataset'].

ARGS['model']: Default: None; Here are some available models to choose from: GLM-130B, ChatGLM-6B,ChatGLM2-6B, llama-7b, lama-13b, llama-33b, llama-65b, gpt2. Please note that the selection of the model should be based on the specific number and memory information of the GPUs, combined with the training method. The GLM-130B, ChatGLM-6B, and ChatGLM2-6B models are suitable for Chinese data. If the graphics memory is insufficient, we recommend using the 6B model. If the graphics memory is sufficient, we recommend the GLM-130B model. In addition, the llama-7b, llama-13b, llama-33b, llama-65b, and gpt2 models are suitable for English data. Among them, gpt2 is the smallest model, and only requires 6G GPU memory for full parameter fine-tuning.

ARGS['method']: Default: None;  If the user does not have a preference for a specific model, recommendations can be made based on the available GPU memory. (LoRA: Low-Rank Adaptation; QLoRA: Quantized LoRA; LOMO: LOw-Memory Optimization; None: Full parameter fine-tune with AdamW) Among them, LoRA and QLoRA only fine tune a portion of the parameters in the model to reduce the required graphics memory, while LOMO is a special SGD optimizer that can fine tune all parameters under low graphics memory requirements) A 6GB GPU memory supports training of the 6B model with QLoRA mode. A 12GB GPU memory supports training of the 6B model with QLoRA/LoRA/LOMO modes or the 13B model with QLoRA mode. A 24GB GPU memory supports training of the 33B model with QLoRA mode or the 13B model with QLoRA/LoRA/LOMO modes. A 48GB GPU memory supports training of the 65B model with QLoRA mode or the 33B model with QLoRA/LoRA/LOMO modes, or full parameter fine-tuning of the 7B model (using AdamW optimizer). An 80GB GPU memory supports training of the 65B model with QLoRA/LoRA/LOMO modes or full parameter fine-tuning of the 13B model (using AdamW optimizer). Four 48GB GPU memory cards support training of the 130B model with QLoRA mode, while eight 48GB GPU memory cards support training of the 130B model with QLoRA/LoRA/LOMO modes.

ARGS['learning rate']: Default 1e-5; If using LoRA or QLoRA as the method, please set it to 1e-3. Otherwise, set it to 1e-5.

ARGS['epoch']: Defalt 10,
ARGS['lora_rank']: Default 16,
ARGS['batch size']: Default 6,
ARGS['max length']: Default 1024,
ARGS['gradient accumulation']: Default 1
ARGS['GPU Number']: Represents the number of GPUs on the local machine.
ARGS['GPU Memory']: Represents the memory size of the first GPU on the local machine.
ARGS['Quantization']: Default None; if ARGS['method']=='QLoRA', INT4 or INT8 quantization can be used, but it is recommended to use INT4 quantization. You need to set 4 or 8 for QLoRA.

ARGS['Train_This_Machine']: Default False; If the server's GPU is sufficient for training, it can be set to True with the user's consent, and training will start automatically.
ARGS['save interval']: Default 1000; Save the weights every N steps.
ARGS['save path']: Default './checkpoints'
ARGS['language']: Default 'English'; If the user is speaking to you in Chinese, you should respond in Chinese and set the language to Chinese.
ARGS['gradient accumulation']: Default 1; Gradient accumulation steps during training.
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
                'learning rate': 1e-5,
                'epoch': 10,
                'lora_rank': 16,
                'batch size': 6,
                'max length': 1024,
                'GPU Number': pynvml.nvmlDeviceGetCount(),
                'GPU Memory': str(round(pynvml.nvmlDeviceGetMemoryInfo(0).free / (1024 * 1024), 2)) + 'MB',
                'Quantization': None,
                'Train_This_Machine': False,
                'save interval': 1000,
                'save path': './checkpoints',
                'gradient accumulation': 1,
                'language': 'English',
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
                'GPU Number': 2,
                'GPU Memory': '48 GB',
                'Quantization': 0,
                'Train_This_Machine': True,
                'save interval': 1000,
                'save path': './checkpoints',
                'gradient accumulation': 1,
                'language': 'English',
                }
    print(ARGS['GPU Number'])
    print_stream('\033[0;33m[INFO] This library facilitates the training of language models, making the process effortless!\033[0m', 0.005)
    print_stream(
        '\033[0;33m[INFO] In conversations with Lingo, the language model will be trained automatically according to your needs, without requiring any effort on your part 😊\033[0m',
        0.005)
    print_stream('\033[0;33m[INFO] Would you like to command Lingo through casual conversation? \033[0m')

    if openai.api_key != None:
        user_conversation = input(
            '[Answer] If yes, please type (Yes), let"s go~, If not, please type (No): ')
        print('')
        if user_conversation == 'No':
            let_lingo_choice(ARGS)
        else:
            let_lingo_conversation(ARGS)
    else:
        user_conversation = input(
            '[Answer] If yes, please provide your OpenAI API KEY. If not, please type (No): ')
        print('')
        if user_conversation == 'No':
            let_lingo_choice(ARGS)

        else:
            openai.api_key = user_conversation
        let_lingo_conversation(ARGS)


def let_lingo_choice():

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

    print_stream('\033[0;36mUnsupport!\033[0m')
    exit(0)


def let_lingo_conversation(ARGS):
    print_stream(
        '\033[0;36m[AI] Hello there! I"m your AI assistant, and I"m here to help you train your model. Before we get started, it"s important to have a clear plan and goal in mind. \033[0m')
    data_input = input('[Answer] : ')

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
            responses = response.split('\n')
            for response_item in responses:
                print_stream(
                    '\033[0;36m[AI] {} \033[0m'.format(response_item))
            data_input = input('[Answer] : ')

        Now_Break = True
        for value in ARGS.values():
            if None == value:
                Now_Break = False
        if Now_Break:
            ARGS['train continue'] = False
            break
    ARGS = {'data': '/data/LLM/Lingo/lingo.json', 'train continue':False,'gradient accumulation':1,'model': 'llama-7b', 'method': 'QLoRA', 'learn rate': 1e-05, 'epoch': 10, 'lora_rank': 16, 'batch size': 6, 'max length': 1024, 'GPU Number': '8', 'GPU Memory': 'No Support', 'Quantization': 0, 'Train_This_Machine': True, 'save interval': 1000, 'save path': './checkpoints', 'language': 'English'}
    print(f'\033[0;33m[INFO] {str(ARGS)} \033[0m')

    cmd,glm_130b_python_code,wandb_python_code,ARGS = get_cmd(ARGS)

    if ARGS['Train_This_Machine'] == True:
        json.dump(ARGS,open('./ARGS.json','w',encoding='utf-8'))
        print_stream(
            '\033[0;36m[AI] We will train the model~. Let"s Go!\033[0m')
        if launch_cmd(cmd) == 0:
            print_stream(
                '\033[0;36m[AI] We have successfully trained the model now !\033[0m')

    else:
        readme_file = write_readme(ARGS,cmd,glm_130b_python_code,wandb_python_code)
        with open('./readme.md','w',encoding='utf-8') as f:
            for i in readme_file:
                f.write(i+'\n')
            f.write('### ARGMENTS\n')
            f.write(json.dumps(ARGS,indent=4,ensure_ascii=False))
        print_stream(
            f'\033[0;36m[AI] You will train the model in other machines with GPUs, and I have written a readme ({os.path.dirname(os.path.abspath (__file__))}/readme.md) for you. You can refer to its commands. \033[0m')




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

    if ARGS['method']=='QLoRA' and ARGS['Quantization'] not in [4,8]:
        ARGS['Quantization'] = 4

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
        cmd += '--use_lora 1 --lora_rank {} '.format(ARGS['lora_rank'])
    if ARGS['method'] == 'QLoRA':  # TODO 使用QLoRA时，CPU-offload必须设置为false
        cmd += '--use_lora 1 --quantization-bit {} --lora_rank {} '.format(ARGS['Quantization'], ARGS['lora_rank'])
    if ARGS['method'] == 'LOMO':  # TODO LOMO必须要ZeRO-Stage3和bf16
        cmd += '--use_lomo 1 '
    if ARGS['rope_scaling'] == True:
        cmd += '--rope_scaling True '


    if ARGS['train continue'] == False:
        print_stream(
            '\033[0;36m[AI] By the way, do you want to use Wandb to record logs? \033[0m')
        print_stream(
            '\033[0;36m[AI] If so, please provide Wandb"s API KEY, which may be located in https://wandb.ai/settings. Of course, if you enter No, we can skip this step \033[0m')
        wandb_api = input(
            '[Answer] : ')
        if len(wandb_api) >= 10:
            if ARGS['Train_This_Machine'] == True:

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

    if ARGS['data'] in LINGO_SUPPORT_DATASET:
        dataset = "WENGSYX/" + ARGS["data"]
    else:
        dataset = ARGS['data']

    cmd += f'--fp16 --dataset {dataset} --train-data {ARGS["data"]} --valid-data {ARGS["data"]} --max_seq_length {ARGS["max length"]} --epochs {ARGS["epoch"]}  --train-iters 0  --no-load-rng --warmup .02 --checkpoint-activations --save-interval {ARGS["save interval"]} --save "{ARGS["save path"]}" --split 1 --eval-interval 1000000 --eval-batch-size 2 --lr {ARGS["learn rate"]} --num-workers 0 --log-interval {log_interval} '

    cmd += '--deepspeed_config ./ds_config.json'

    ds_config = get_deepspeed(ARGS)
    with open('./ds_config.json','w',encoding='utf-8') as f:
        f.write(ds_config)

    return cmd,glm_130b_python_code,wandb_python_code,ARGS

def write_readme(ARGS,cmd,glm_130b_python_code,wandb_python_code):
    readme_lines = []
    if ARGS['language'] == 'Chinese':
        readme_lines.append('# 使用Lingo在另一台服务器中训练模型')
        readme_lines.append('***欢迎来到Lingo***,我们能够为你提供各种你想要的训练方式')
        readme_lines.append('')
        readme_lines.append('### 安装环境')
        readme_lines.append('具体过程请参考[Github说明](https://github.com/WENGSYX/Lingo/blob/main/QA/readme.md)')
        readme_lines.append('')

        readme_lines.append('### 操作')
        if ARGS['model'] == 'GLM-130B':
            readme_lines.append('#### 下载 GLM-130B')
            readme_lines.append(
                '你可以从这里下载GLM-130B的权重: https://docs.google.com/forms/d/e/1FAIpQLSehr5Dh_i3TwACmFFi8QEgIVNYGmSPwV0GueIcsUev0NEfUug/viewform?usp=sf_link')
            readme_lines.append('')
            readme_lines.append('#### 合并 GLM-130B')
            readme_lines.append(
                '之后你需要合并压缩文件，并解压成文件夹')
            readme_lines.append('```bash')
            readme_lines.append('cat glm-130b-sat.tar.part_* > glm-130b-sat.tar')
            readme_lines.append('tar xvf glm-130b-sat.tar')
            readme_lines.append('```')
            readme_lines.append('')
            readme_lines.append('#### 转换 GLM-130B 权重')
            readme_lines.append(
                '最后，你需要将GLM-130B的权重按照实际需求进行转换：')
            readme_lines.append('```bash')
            readme_lines.append(glm_130b_python_code)
            readme_lines.append('```')
            readme_lines.append('')
        if wandb_python_code:
            readme_lines.append('#### 登陆 wandb')
            readme_lines.append(
                '请在shell中运行以下代码登陆wandb')
            readme_lines.append('```bash')
            readme_lines.append(wandb_python_code)
            readme_lines.append('```')
        readme_lines.append('##### 复制ds_config')
        readme_lines.append(
            '请将原始生成的***ds_config.json***移动到训练路径下')
        readme_lines.append('')
        readme_lines.append('##### 训练操作')
        readme_lines.append(
            '请在服务器中手动执行以下代码，开始训练：')
        readme_lines.append('```bash')
        readme_lines.append(cmd)
        readme_lines.append('```')
        readme_lines.append('')
        readme_finally = '### 引用\n\n本项目为[神经理解](https://github.com/WENGSYX/Neural-Comprehension)的伴生项目。如果您对我们的项目感兴趣，欢迎引用。\n\n```\n@misc{weng2023mastering,\n      title={Mastering Symbolic Operations: Augmenting Language Models with Compiled Neural Networks}, \n      author={Yixuan Weng and Minjun Zhu and Fei Xia and Bin Li and Shizhu He and Kang Liu and Jun Zhao},\n      year={2023},\n      eprint={2304.01665},\n      archivePrefix={arXiv},\n      primaryClass={cs.CL}\n}\n```\n\n### 免责声明\n\n***本项目相关资源仅供学术研究之用，严禁用于商业用途。***\n使用涉及第三方代码的部分时，请严格遵循相应的开源协议。模型生成的内容受模型计算、随机性和量化精度损失等因素影响，本项目不对其准确性作出保证。对于模型输出的任何内容，本项目不承担任何法律责任，亦不对因使用相关资源和输出结果而可能产生的任何损失承担责任。\n'

    else:
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


if __name__ == '__main__':
    let_lingo()
