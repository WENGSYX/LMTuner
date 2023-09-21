#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import sys

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from LMTuner.models import get_model_and_tokenizer
from LMTuner.models.model_io import save_checkpoint
# from LMTuner.utils import get_all_reduce_mean, get_optimizer_grouped_parameters, to_device

from .reaward_data import RewardData
from .reaward_data import DataCollatorReward
from .reward_model import RewardModel
from .reward_util import get_train_ds_config

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

class Args:
    # init it
    def __init__(self) -> None:
        pass



def main():
    args = Args()
    device = torch.device("cuda:1")

    rm_model,tokenizer,args = get_model_and_tokenizer(args)
    rm_model = RewardModel(rm_model, tokenizer, num_padding_at_beginning=0)

    train_dataset = RewardData()
    eval_dataset = RewardData()

    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size)

    def evaluation_reward(model, eval_dataloader):
        model.eval()
        correct_predictions = 0
        total_predictions = 0
        scores = 0
        for step, batch in enumerate(eval_dataloader):
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            chosen = outputs["chosen_mean_scores"]
            rejected = outputs["rejected_mean_scores"]
            correct_predictions += (chosen > rejected).sum()
            total_predictions += chosen.shape[0]
            scores += outputs["chosen_mean_scores"].mean().float()
            if step == 99:  # For faster evaluation and debugging
                break
        acc = correct_predictions / total_predictions
        scores = scores / (step + 1)
        return scores, acc


    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(rm_model.parameters(),
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    batch_size = args.per_device_train_batch_size,
                                    micro_batch_size = args.micro_batch_size)


    rm_model, optimizer, *_ = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        dist_init_required=True)

    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()

    # Train!
    print("***** Running training *****", args.global_rank)

    print(
        f"***** Evaluating reward, Epoch {0}/{args.num_train_epochs} *****",
        args.global_rank)
    reward_score, acc = evaluation_reward(rm_model, eval_dataloader)
    print(
        f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}",
        args.global_rank)

    for epoch in range(args.num_train_epochs):
        print(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        rm_model.train()
        mean_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            outputs = rm_model(**batch)
            loss = outputs["loss"]
            rm_model.backward(loss)
            rm_model.step()
            mean_loss += loss.item()
        print(
            f"Epoch {epoch+1}/{args.num_train_epochs} with loss {mean_loss/(step+1)}")
        # Evaluate reward_loss on the validation set.
        print(
            f"***** Evaluating reward, Epoch {epoch+1}/{args.num_train_epochs} *****")
        reward_score, acc = evaluation_reward(rm_model, eval_dataloader)
        print(
            f"chosen_last_scores (higher is better) : {reward_score}, acc (higher is better) : {acc}")
        rm_model.tput_timer.update_epoch_count()

    if args.output_dir is not None:
        print('saving model ...', args.global_rank)

        if args.global_rank == 0:
            save_checkpoint('latest',rm_model, args)


if __name__ == "__main__":
    main()
