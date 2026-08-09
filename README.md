
## Installation and Requirements

1. Create a conda environment, activate it and install Packages
```Shell
conda create -n HLLaVA python=3.10 -y
conda activate HLLaVA
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

2. Install additional packages
```Shell
pip install flash-attn==2.5.7 --no-build-isolation
```
#### Upgrade to the latest code base

```Shell
git pull
pip install -e .
```

## Get Started

#### 1. Data Preparation

Please refer to the [Data Preparation](https://tinyllava-factory.readthedocs.io/en/latest/Prepare%20Datasets.html) section in TinyLLaVA's [Documenation](https://tinyllava-factory.readthedocs.io/en/latest/).

#### 2. Train

Here's an example for training a LMM using Phi-2.

- Replace data paths with yours in `scripts/train/train_phi.sh`
- Replace `output_dir` with yours in `scripts/train/pretrain.sh`
- Replace `pretrained_model_path` and `output_dir` with yours in `scripts/train/finetune.sh`
- Adjust your GPU ids (localhost) and `per_device_train_batch_size` in `scripts/train/pretrain.sh` and `scripts/train/finetune.sh`

```bash
bash scripts/train/train_phi.sh
```

#### 3. Evaluation

Please refer to the [Evaluation](https://tinyllava-factory.readthedocs.io/en/latest/Evaluation.html) section in TinyLLaVA's [Documenation](https://tinyllava-factory.readthedocs.io/en/latest/Evaluation.html).

