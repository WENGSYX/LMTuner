from sat import mpu, get_args, get_tokenizer
from LMTuner.trainer import training_main
from LMTuner.initialize import initialize
import torch,json
import time
from LMTuner.models import get_model_and_tokenizer
from LMTuner.dataset import LingoDataset
from LMTuner.setting import *

from transformers import DataCollatorForSeq2Seq
from datasets import Dataset


def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['input_ids', 'labels']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()
    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['input_ids'].long()
    labels = data_b['labels'].long()

    return tokens, labels


from torch.nn import CrossEntropyLoss
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu
import numpy as np


def forward_step_eval(data_iterator, model, args, timers):
    # Metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
            result = scores[0]

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred))
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict

    # Get the batch.
    timers('batch generator').start()
    tokens, labels = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()

    gen_kwargs = {"max_length": 512, "num_beams": 1, "do_sample": True, "top_p": 0.7,
                  "temperature": 0.95}
    outputs = model.generate(input_ids=tokens, **gen_kwargs)
    return torch.tensor(0, device=outputs.device), {k: torch.tensor(v, device=outputs.device) for k, v in
                                                    compute_metrics((outputs.cpu(), labels.cpu())).items()}


def forward_step(data_iterator, model, args, timers,lr=None):
    """Forward step."""

    # Get the batch.
    if lr is None:
        lr = 0
    timers('batch generator').start()
    tokens, labels = get_batch(
        data_iterator, args, timers)
    timers('batch generator').stop()


    logits, *_  = model(input_ids=tokens.to(torch.int64))
    dtype = logits.dtype
    lm_logits = logits.to(torch.float32)

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    loss = loss.to(dtype)
    if args.wandb:
        global Total_Tokens
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                Total_Tokens += sum((shift_labels!=-100).view(-1)) * torch.distributed.get_world_size()
                wandb.log({'Tokens': Total_Tokens, 'Loss': loss.item(),'lreaning rate':lr})
        else:
            Total_Tokens += sum((shift_labels!=-100).view(-1))
            wandb.log({'Tokens': Total_Tokens,'Loss': loss.item(),'lreaning rate':lr})
    return loss, {'loss': loss}


def create_dataset_function(path, args):

    if args.dataset in LINGO_SUPPORT_DATASET:
        lingo_dataset = LingoDataset(args.dataset)
        data = lingo_dataset.turn_conversations_to_io()
    else:
        data = [json.loads(i) for i in open(args.dataset, encoding='utf-8').readlines()]
    def _gen():
        for i in data:
            if len(i) == 2:
                if type(i['input']) == str and type(i['output']) == str:
                    yield i
    dataset = Dataset.from_generator(_gen)


    dataset = dataset.map(args.dataset_function, batched=True,remove_columns=['input','output'],
                              load_from_cache_file=True, desc="Running tokenizer on train dataset")

    return dataset


if __name__ == '__main__':
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
        py_parser.add_argument('--lora_save',type=str,default='')
        py_parser.add_argument('--lora_load',type=str,default='')

        py_parser.add_argument('--models', type=str, default="")
        py_parser.add_argument('--dataset', type=str, default="")
        py_parser.add_argument('--finetune', type=bool, default=True)
        py_parser.add_argument('--wandb', type=bool, default=False)
        py_parser.add_argument('--quantization_bit', type=int, default=0)

        py_parser.add_argument('--rope_scaling', type=bool, default=False)

    args = initialize(extra_args_provider=add_generation_specific_args)
    if args.wandb:
        import wandb
        from pynvml import *
        import pynvml

        Total_Tokens = 0
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:

                wandb.init(
                    project=f"{time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))}",
                    config={
                        "model": args.models,
                        "seed": args.seed,
                        "dataset": args.dataset,
                        "time": time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())),
                        "save interval": args.save_interval,
                        "save": args.save,
                        "learn_rate": args.lr,
                        "batch size": args.batch_size,
                        "GPU": pynvml.nvmlDeviceGetName(handle),
                        "GPU Number": pynvml.nvmlDeviceGetCount(),
                        "GPU Memory": str(round(pynvml.nvmlDeviceGetMemoryInfo(handle).total/ (1024 * 1024), 2)) + 'MB',
                        "GPU Driver": nvmlSystemGetDriverVersion()}
                )

        else:
            wandb.init(
                project=f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))}",
                config={
                    "model": args.models,
                    "seed": args.seed,
                    "dataset": args.dataset,
                    "time": time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                    "save interval": args.save_interval,
                    "save": args.save,
                    "learn_rate": args.lr,
                    "batch size": args.batch_size,
                    "GPU": pynvml.nvmlDeviceGetName(handle),
                    "GPU Number": pynvml.nvmlDeviceGetCount(),
                    "GPU Memory": str(round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024 * 1024), 2)) + 'MB',
                    "GPU Driver": nvmlSystemGetDriverVersion()}
            )

    model,tokenizer,args = get_model_and_tokenizer(args)
    get_tokenizer(outer_tokenizer=tokenizer)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False,
    )
    training_main(args, model_cls=model, forward_step_function=forward_step, create_dataset_function=create_dataset_function, collate_fn=data_collator, forward_step_eval=forward_step_eval)
