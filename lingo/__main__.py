from lingo import let_lingo,get_cmd,launch_cmd
import json
import argparse
import os

def continue_train(ARGS_file):
    ARGS = json.load(open(ARGS_file))
    ARGS['train continue'] = True
    cmd, glm_130b_python_code, wandb_python_code, ARGS = get_cmd(ARGS)

    if ARGS['Train_This_Machine'] == True:
        ARGS['train continue'] = True
        json.dump(ARGS, open('./ARGS1.json', 'w', encoding='utf-8'))
        print(
            '\033[0;36m[AI] We will train the model~. Let"s Go!\033[0m')
        if launch_cmd(cmd) == 0:
            print(
                '\033[0;36m[AI] We have successfully trained the model now !\033[0m')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ARGS', type=str, default='')
    args = parser.parse_args()
    if os.path.exists(args.ARGS):
        continue_train(args.ARGS)
    else:
        let_lingo()
