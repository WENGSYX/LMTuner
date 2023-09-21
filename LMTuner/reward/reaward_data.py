import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import LlamaTokenizer

def dataReader(args=None, type=0):
    # type=0: train
    # type=1: validation
    # need args : reward_validation_path/reward_validation_path

    if type==1:
        if not hasattr(args,'reward_validation_path'):
            df = pd.read_parquet('/home/wangzhiqi/data/oasst1_rlhf_reward/validation-00000-of-00001-6855d7506403041c.parquet')
        else:
            df = pd.read_parquet(args.reward_validation_path)
    else:
        if not hasattr(args,'reward_train_path'):
            # read parquet file
            df = pd.read_parquet('/home/wangzhiqi/data/oasst1_rlhf_reward/train-00000-of-00001-5466fcbe20f5c0ef.parquet')
        else:
            df = pd.read_parquet(args.reward_train_path)
 
    data = [list(df[x]) for x in list(df)]
    data = list(map(list, zip(*data)))

    return data

class RewardData(Dataset):
    def __init__(self):
        self.data  = dataReader()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx][2].replace('prompter:','User:'), 'Assistent:'+self.data[idx][3], 'Assistent:'+self.data[idx][4])

class DataCollatorReward:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
    def __call__(self, data):
        chosen_dataset = []
        reject_dataset = []

        max_seq_len = 1024

        tokenizer = self.tokenizer
        for i, tmp_data in enumerate(data):
            # tokenize the text
            chosen_sentence = tmp_data[0]+'\n' + tmp_data[1]  # the accept response
            reject_sentence = tmp_data[0]+'\n' + tmp_data[2]  # the accept response


            if chosen_sentence is not None and reject_sentence is not None:
                chosen_sentence += tokenizer.eos_token  # the accept response
                reject_sentence += tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
                chosen_token = tokenizer([chosen_sentence],
                                         max_length=max_seq_len,
                                         padding='max_length',
                                         return_tensors="pt")
                reject_token = tokenizer([reject_sentence],
                                         max_length=max_seq_len,
                                         padding='max_length',
                                         return_tensors="pt")

                chosen_token["input_ids"] = chosen_token["input_ids"]
                chosen_token["attention_mask"] = chosen_token["attention_mask"]

                reject_token["input_ids"] = reject_token["input_ids"]
                reject_token["attention_mask"] = reject_token["attention_mask"]
                if chosen_token["input_ids"].size()[-1] != max_seq_len or reject_token["input_ids"].size()[-1] != max_seq_len:
                    continue
                chosen_dataset.append(chosen_token)
                reject_dataset.append(reject_token)

        batch = {}

        batch["input_ids"] = torch.cat([f["input_ids"] for f in chosen_dataset] + [f["input_ids"] for f in reject_dataset],
                                       dim=0)
        batch["attention_mask"] = torch.cat([f["attention_mask"] for f in chosen_dataset] +
                                            [f["attention_mask"] for f in reject_dataset],
                                            dim=0)
        return batch
if __name__ == '__main__':
    x = RewardData()
    tokenizer = LlamaTokenizer.from_pretrained('/data/LLM/llama/hf')
    collector = DataCollatorReward(tokenizer=tokenizer)
    x_loader = DataLoader(x,batch_size=4,collate_fn=collector)

    for i in x_loader:
        print(i)
        break
