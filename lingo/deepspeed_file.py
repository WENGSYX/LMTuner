

def get_deepspeed(ARGS):
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

    if ARGS['method'] == 'QLoRA':
        default_deepspeed_config.replace('"cpu_offload": true','"cpu_offload": false')

    default_deepspeed_config.replace('"train_micro_batch_size_per_gpu": 2',f'"train_micro_batch_size_per_gpu": {ARGS["batch size"]}')
    default_deepspeed_config.replace('"gradient_accumulation_steps": 1',f'"gradient_accumulation_steps": {ARGS["gradient accumulation"]}')
    return default_deepspeed_config

