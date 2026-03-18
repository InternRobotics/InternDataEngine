# Training Instructions

Here we provide instructions for pretraining on InternData-A1, finetuning on real-world tasks and finetuning on InternData-A1 tasks for sim2real transfer.

Before training, you need to compute the normalization statistics for the tasks you want to train on. Please refer to [norm_stats.md](norm_stats.md) for more details.

---

## 1. Pretraining on InternData-A1


### Write a training config
We provide a `TrainConfig` example named `pretrain-interndata-a1` in `src/openpi/training/config.py`.
InternData-A1 contains four robot embodiments:
- `split_aloha`
- `lift2`
- `genie1`
- `franka`

Accordingly, we define three `MultiDataConfigFactory` classes:
- `MultiSimSplitAlohaDataConfig` for `split_aloha` and `lift2`
- `MultiSimGenieDataConfig` for `genie1`
- `MultiSimFrankaDataConfig` for `franka`

Please either:
- create a soft link from the InternData-A1 dataset to `data/InternData-A1`, or
- modify the `repo_dir` field in all relevant `MultiDataConfig` entries to point to your local InternData-A1 path.

Set `stats_dir` to your local normalization statistics directory. If you use the default setting, ensure that the normalization statistics for simulation tasks are saved under `stats/sim`.

We initialize the model from PaliGemma-3B using:
```
weight_loader=weight_loaders.PaliGemmaWeightLoader("checkpoints/jax/paligemma/pt_224.npz")
```
Please download the PaliGemma-3b checkpoint by running 
```
python scripts/download_paligemma.py
```

You may adjust other training parameters based on your available GPUs and training budget:
- `num_train_steps`: Total number of training steps
- `num_workers`: Number of data loading workers
- `fsdp_devices`: Number of GPUs used for FSDP per node to distribute model parameters, gradients, and optimizer states across devices for reduced memory usage
- `batch_size`: Batch size per GPU
- `save_interval`: Checkpoint saving interval (in steps)

### Run training
For multi node training, run
```
bash scripts/training_scripts/multi_node.sh
```

For single node multi-GPU training, run
```
config_name=pretrain-interndata-a1
bash scripts/training_scripts/single_node_multi_gpu.sh  ${config_name}
```

The ckpts will be saved to `checkpoints/${config_name}`.

## 2. Finetuning on Real-World Tasks
### Write a training config
We provide a `TrainConfig` example named `finetune-a2d-pen` in `src/openpi/training/config.py`.

Key arguments you may need to modify include:
- `MultiDataConfigFactory` class: 
    - `MultiLeRobotReala2dDataConfig` for `genie1`
    - `MultiLeRobotRealArxLift2DataConfig` for `lift2` and `acone`
- `repo_dir`: Path to the real-world task dataset.
- `robot_name`: the robot name in `repo_dir`, e.g. "genie1".
- `fixed_stats_dir`: Path to the normalization statistics for the real-world task. When this is set, statistics from `stats_dir` will not be used.
- `weight_loader`: Pretrained checkpoint used for initialization.
You may download our pretrained checkpoints from [here]().

### Run training
For training, run
For single node multi-GPU training, run
```
config_name=finetune-a2d-pen
bash scripts/training_scripts/single_node_multi_gpu.sh ${config_name}
```

The ckpts will be saved under `checkpoints/${config_name}`.

## 3. Finetuning on InternData-A1 Tasks for Sim2Real Transfer
### Write a training config
We provide a `TrainConfig` example named `finetune-sim2real-lift2-sort-rubbish` in `src/openpi/training/config.py`.

Key arguments you may need to modify include:
- `MultiDataConfigFactory` class: Currently, sim-to-real transfer is evaluated only on `lift2` tasks:
    - `MultiSim2RealSplitAlohaDataConfig` for `lift2`
- `repo_dir`: Path to the corresponding InternData-A1 task.
- `fixed_stats_dir`: Path to the normalization statistics for the sim-to-real task. When specified, statistics from `stats_dir` will not be used.
- `weight_loader`: Pretrained checkpoint used for initialization.

### Run training
For training, run
For single node multi-GPU training, run
```
config_name=finetune-sim2real-lift2-sort-rubbish
bash scripts/training_scripts/single_node_multi_gpu.sh ${config_name}
```