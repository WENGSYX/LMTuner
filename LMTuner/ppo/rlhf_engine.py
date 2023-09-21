# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import time
import torch
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam

from LMTuner.models import get_model_and_tokenizer
from LMTuner.models.model_io import load_checkpoint
from LMTuner.reward.reward_model import RewardModel


# from utils.ds_utils import get_train_ds_config, get_eval_ds_config

# from utils.model.model_utils import create_hf_model, create_critic_model
# from utils.utils import get_optimizer_grouped_parameters
"""
TODOs:
  * support HF models for critic (for debugging), must be a previously saved ckpt from step-2
  * determine ds_config/zero_stage based on model size, gpu style, world size, etc
    - get model size by creating simple meta model
    - 1.3b: zero-2 for actor/ref models, zero-0 for others
    - 13b+: zero-3 for all models
"""


class DeepSpeedRLHFEngine():

    def __init__(self, args, num_total_iters):
        self.args = args
        self.num_total_iters = num_total_iters


        actor_model, self.tokenizer, args = get_model_and_tokenizer(args)
        ref_model ,_ , _ = get_model_and_tokenizer(args)

        critic_model = RewardModel(args.critic_model_name_or_path, self.tokenizer, num_padding_at_beginning=0)
        reward_model = RewardModel(args.critic_model_name_or_path, self.tokenizer, args.num_padding_at_beginning)
        load_checkpoint(critic_model, args)
        load_checkpoint(reward_model, args)

        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        #todo:断点续训

        optim_actor = AdamOptimizer(lr=self.args.actor_learning_rate,
                              betas=(0.9, 0.95))

        optim_critic = AdamOptimizer(lr=self.args.critic_learning_rate,
                              betas=(0.9, 0.95))

        # DeepSpeed Engine
        actor_engine, *_ = deepspeed.initialize(model=actor_model,
                                                optimizer=optim_actor,
                                                )
        
        ref_engine, *_ = deepspeed.initialize(model=ref_model)
        
        critic_engine, *_ = deepspeed.initialize(model=critic_model,
                                                 optimizer=optim_critic,
                                                 )
        
        reward_engine, *_ = deepspeed.initialize(model=reward_model)

        self.actor = actor_engine
        self.ref = ref_engine
        self.critic = critic_engine
        self.reward = reward_engine
        # self.actor_optim = actor_optim
        # self.critic_optim = critic_optim




