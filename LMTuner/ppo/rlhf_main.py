#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""

engine = DeepSpeedRLHFEngine(actor_model_name_or_path=actor_model_name_or_path,
                             critic_model_name_or_path=critic_model_name_or_path,
                             tokenizer=tokenizer,
                             args=args)
trainer = DeepSpeedPPOTrainer(engine=engine, args=args)

for prompt_batch in prompt_train_dataloader:
    out = trainer.generate_experience(prompt_batch)
    actor_loss, critic_loss = trainer.train_rlhf(out)

"""
import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    default_data_collator,
)

import deepspeed

from ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from rlhf_engine import DeepSpeedRLHFEngine
from LMTuner.models.model_io import save_checkpoint
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from .ppo_data import create_prompt_dataset, MiniDataset, DataCollatorRLHF, get_unsupervised_data
from .ppo_util import print_rank_0, get_all_reduce_mean, set_random_seed, to_device, moving_average
from transformers import LlamaTokenizer



def create_datasets(args, tokenizer, train_phase=3):
    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    prompt_train_dataset, _ = create_prompt_dataset(
        args.local_rank, args.data_path, args.data_split,
        args.data_output_path, train_phase, args.seed, tokenizer,
        args.max_prompt_seq_len)
    if unsupervised_training_enabled:
        unsupervised_train_dataset = get_unsupervised_data(args, tokenizer)
    else:
        unsupervised_train_dataset = None

    # DataLoaders creation:
    data_collator = DataCollatorRLHF(args.max_prompt_seq_len,
                                     args.inference_tp_size)
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = RandomSampler(
                unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if unsupervised_training_enabled:
            unsupervised_train_sampler = DistributedSampler(
                unsupervised_train_dataset)
    prompt_train_dataloader = DataLoader(
        prompt_train_dataset,
        collate_fn=data_collator,
        sampler=prompt_train_sampler,
        batch_size=args.per_device_train_batch_size)
    if unsupervised_training_enabled:
        unsupervised_train_dataloader = DataLoader(
            unsupervised_train_dataset,
            collate_fn=default_data_collator,
            sampler=unsupervised_train_sampler,
            batch_size=args.per_device_train_batch_size)
    else:
        unsupervised_train_dataloader = [None] * len(
            prompt_train_dataloader)  # basically a dummy dataloader

    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (args.per_device_train_batch_size / args.per_device_mini_train_batch_size) * \
        args.ppo_epochs / args.gradient_accumulation_steps
    num_total_iters = int(args.num_train_epochs * num_update_steps_per_epoch)

    return prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters

class Args():
    def __init__(self) -> None:
        pass


def main():
    args = Args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    assert not args.offload, "zero-offload is not currently supported but coming soon!"

    unsupervised_training_enabled = args.unsupervised_dataset_name and args.unsupervised_dataset_config_name
    if unsupervised_training_enabled:
        # if we enable unsupervised training, we need to double the batch size for actor model
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    else:
        args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()

    # create common tokenizer based on actor model
    tokenizer = LlamaTokenizer.from_pretrained('/home/wangzhiqi/deepspeed-chat/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output',
                                  fast_tokenizer=False)
    tokenizer.pad_token = tokenizer.eos_token

    prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(
        args=args, tokenizer=tokenizer, train_phase=3)

    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        tokenizer=tokenizer,
        num_total_iters=num_total_iters,
        args=args)

    args.end_of_conversation_token = "<|endoftext|>"

    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if unsupervised_training_enabled else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)

    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                   args.per_device_mini_train_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batch_numbers,
                                     args.per_device_mini_train_batch_size)

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)

    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}",
            args.global_rank)
        for step, (batch_prompt, batch_unsupervised) in enumerate(
                zip(prompt_train_dataloader, unsupervised_train_dataloader)):
            batch_prompt = to_device(batch_prompt, device)
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add(
                    [[None] * args.per_device_train_batch_size])
            # prompts = batch_prompt['prompt']
            # length = prompts.size(-1)
            # if length > args.max_prompt_seq_len:
            #     prompts = prompts[:, length - args.max_prompt_seq_len:]
            #     raise ValueError("Prompt length is too long")

            out = trainer.generate_experience(batch_prompt['prompt'],
                                              batch_prompt['prompt_att_mask'])
            exp_dataset = exp_mini_dataset.add(out)

            if exp_dataset is not None:
                inner_iter = 0
                actor_loss_sum, critic_loss_sum, unsup_loss_sum = 0, 0, 0
                average_reward = 0

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()

                for ppo_ep in range(args.ppo_epochs):
                    for i, (exp_data, unsup_data) in enumerate(
                            zip(exp_dataset, unsup_dataset)):
                        actor_loss, critic_loss = trainer.train_rlhf(exp_data)
                        actor_loss_sum += actor_loss.item()
                        critic_loss_sum += critic_loss.item()
                        average_reward += exp_data["rewards"].mean()

                        if unsupervised_training_enabled:
                            unsup_loss = trainer.train_unsupervised(
                                unsup_data, args.unsup_coef)
                            unsup_loss_sum += unsup_loss.item()

                        inner_iter += 1
                        if args.enable_ema:
                            moving_average(rlhf_engine.actor,
                                           rlhf_engine.actor_ema,
                                           zero_stage=args.actor_zero_stage)

                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)

                print_rank_0(
                    f'epoch: {epoch}|step: {step}|ppo_ep: {ppo_ep+1}|act_loss: {actor_loss_sum/inner_iter}|cri_loss: {critic_loss_sum/inner_iter}|unsuper_loss: {unsup_loss_sum/inner_iter}',
                    args.global_rank)
                average_reward = get_all_reduce_mean(average_reward).item()
                print_rank_0(
                    f"average reward score: {average_reward/inner_iter}",
                    args.global_rank)
                print_rank_0(
                    "-------------------------------------------------------------------------------------",
                    args.global_rank)

            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable()

    if args.output_dir is not None:
        print_rank_0('saving model ...')

        if torch.distributed.get_rank() == 0:
            save_checkpoint('latest',rlhf_engine.actor,
                           args=args)
            save_checkpoint('latest',rlhf_engine.critic,
                           args=args)


if __name__ == "__main__":
    main()
