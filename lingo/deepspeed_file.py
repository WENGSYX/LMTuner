

def get_deepspeed(ARGS):
    default_deepspeed_config = """{
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 1,
        "gradient_clipping": 0.5,
        "zero_allow_untested_optimizer": true,
        "fp16": {
          "enabled": true
        },
        "zero_force_ds_cpu_optimizer":false,
        "zero_optimization": {
          "stage": 2,
          "cpu_offload": true,
          "contiguous_gradients": false,
          "overlap_comm": false,
          "reduce_scatter": false,
          "reduce_bucket_size": 100000000,
          "allgather_bucket_size": 1000000000,
          "load_from_fp32_weights": false
        },
        "activation_checkpointing": {
          "partition_activations": false,
          "contiguous_memory_optimization": false,
          "cpu_checkpointing": false
        },
        "wall_clock_breakdown": false
      }"""
    if ARGS['method'] == 'LOMO':
        default_deepspeed_config = """{
                "train_micro_batch_size_per_gpu": 2,
                "gradient_accumulation_steps": 1,
                "steps_per_print": 1,
                "gradient_clipping": 0.5,
                "zero_allow_untested_optimizer": true,
                "bf16": {
                  "enabled": true
                },
                "zero_force_ds_cpu_optimizer":false,
                "zero_optimization": {
                    "stage": 3,
                    "contiguous_gradients": true,
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "stage3_prefetch_bucket_size": 1e7,
                    "stage3_param_persistence_threshold": 1e5,
                    "stage3_gather_16bit_weights_on_model_save": true,
                    "reduce_bucket_size": 1e7,
                    "reduce_scatter": true,

                    "sub_group_size": 1e9,
                    "offload_optimizer": {
                        "device": "cpu"
                     },
                    "offload_param": {
                        "device": "cpu"
                   }
        },
                "activation_checkpointing": {
                  "partition_activations": false,
                  "contiguous_memory_optimization": false,
                  "cpu_checkpointing": false
                },
                "wall_clock_breakdown": false
        }"""

    if ARGS['method'] == 'QLoRA':
        default_deepspeed_config = default_deepspeed_config.replace('"cpu_offload": true','"cpu_offload": false')

    default_deepspeed_config = default_deepspeed_config.replace('"train_micro_batch_size_per_gpu": 2',f'"train_micro_batch_size_per_gpu": {ARGS["batch size"]}')
    default_deepspeed_config = default_deepspeed_config.replace('"gradient_accumulation_steps": 1',f'"gradient_accumulation_steps": {ARGS["gradient accumulation"]}')
    return default_deepspeed_config

