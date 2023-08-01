from LMTuner.setting import *
from LMTuner.deepspeed_file import get_deepspeed
from LMTuner.dataset import LingoDataset
from LMTuner.utils import *



def Let_Lingo(ARGS_file = ''):
    if ARGS_file and os.path.exists(ARGS_file):
        ARGS = json.load(open(ARGS_file))
        ARGS['train continue'] = True
        cmd, glm_130b_python_code, wandb_python_code, ARGS = get_cmd(ARGS)

        json.dump(ARGS, open('./ARGS.json', 'w', encoding='utf-8'))
        print(
            '\033[0;36m[AI] We will train the model~. Let"s Go!\033[0m')
        if launch_cmd(cmd) == 0:
            print(
                '\033[0;36m[AI] We have successfully trained the model now !\033[0m')

    else:
        try:
            pynvml.nvmlInit()

            ARGS = {'data': None,
                    'model': None,
                    'method': None,
                    'learning rate': 1e-5,
                    'epoch': 10,
                    'lora rank': 16,
                    'batch size': 6,
                    'max length': 1024,
                    'GPU Number': pynvml.nvmlDeviceGetCount(),
                    'GPU Memory': str(round(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).free / (1024 * 1024), 2)) + 'MB',
                    'quantization': None,
                    'train this machine': True,
                    'save interval': 1000,
                    'save path': './checkpoints',
                    'gradient accumulation': 1,
                    'rope scaling': False,
                    }
        except:
            ARGS = {'data': None,
                    'model': None,
                    'method': None,
                    'learning rate': 1e-5,
                    'epoch': 10,
                    'lora rank': 16,
                    'batch size': 6,
                    'max length': 1024,
                    'GPU Number': None,
                    'GPU Memory': '48 GB',
                    'quantization': 0,
                    'train this machine': False,
                    'save interval': 1000,
                    'save path': './checkpoints',
                    'gradient accumulation': 1,
                    'rope scaling': False,
                    }

        print_stream('\033[0;33m[INFO] This library facilitates the training of language models, making the process effortless!\033[0m', 0.005)
        print_stream(
            '\033[0;33m[INFO] In conversations with Lingo, the language model will be trained automatically according to your needs, without requiring any effort on your part \033[0m',
            0.005)
        print_stream('\033[0;33m[INFO] Would you like to command Lingo through casual conversation? \033[0m')

        if openai.api_key != None:
            user_conversation = input(
                '[Answer] If yes, please type (Yes), let"s go~, If not, please type (No): ')
            print('')
            if user_conversation == 'No':
                ARGS = let_lingo_choice(ARGS)
            else:
                ARGS = let_lingo_conversation(ARGS)
        else:
            user_conversation = input(
                '[Answer] If yes, please provide your OpenAI API KEY. If not, please type (No): ')
            print('')
            if user_conversation == 'No':
                ARGS = let_lingo_choice(ARGS)

            else:
                openai.api_key = user_conversation
                ARGS = let_lingo_conversation(ARGS)
        print(ARGS)
        cmd,glm_130b_python_code,wandb_python_code,ARGS = get_cmd(ARGS)

        if ARGS['train this machine'] == True or ARGS['train this machine'] == 'True':
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




if __name__ == '__main__':
    let_lingo()
