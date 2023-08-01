from LMTuner import Let_Lingo,get_cmd,launch_cmd
from LMTuner.models import get_model_and_tokenizer
import json
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ARGS', type=str, default='')
    args = parser.parse_args()

    if ARGS_file and os.path.exists(ARGS_file):
        ARGS = json.load(open(ARGS_file))
        ARGS['train continue'] = True
        for k,v in ARGS.items():
            args[k] = v

    args.strategy = 'beam_serach'
    args.top_p = 0.95
    args.top_k = 10
    args.temperature = 0.8

    model,tokenizer = get_model_and_tokenizer(args)

    response = model.generate(["patient: hey there i have had cold \"symptoms\" for over a week and had a low grade fever last week. for the past two days i have been feeling dizzy. should i contact my dr? should i see a dr","patient: just found out i was pregnant. yesterday diagnosed with pneumonia. i am a high risk pregnancy. fertility issues, pcos, weak cervix. delivered first daughter at 29 weeks, miscarried, and gave birth at 38 weeks to second daughter, but was on bedrest for weak cervix beginning at 5 months. i m a wreck. when i miscarried they said my progesterone level is low which caused me to miscarry, and gave me progesterone shots every week. can t see doctor for two days.","patient: can you contact coronavirus or any virus from the air?"],tokenizer)
    for i in response:
        print(i)