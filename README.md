This repository contains the unofficial implementation of [SignBERT+: Hand-model-aware Self-supervised Pre-training for Sign Language Understanding](https://ieeexplore.ieee.org/abstract/document/10109128). 

# After cloning

The contents of submodules are not cloned by default. Once cloned, execute the
following:

```bash
git submodule init
git submodule update
```

Or clone using the flag:
```bash
git clone --recurse-submodules <REPO_URL>
```

# Download MANO files

1. Go to the [MANO website](http://mano.is.tue.mpg.de/)
2. Create an account
3. Download Models & Code
4. Extract the `*.zip` inside the `signbert/model/thirdparty/mano_assets` folder
5. Folder structure should look like this:
```bash
mano_assets/
    ├── info.txt
    ├── __init__.py
    ├── LICENSE.txt
    ├── models
    │   ├── info.txt
    │   ├── LICENSE.txt
    │   ├── MANO_LEFT.pkl
    │   ├── MANO_RIGHT.pkl
    │   ├── SMPLH_female.pkl
    │   └── SMPLH_male.pkl
    └── webuser
        └── ...
```

# Create virtual environment

```bash
conda env create -f environment.yml
```

# Install manotorch
```
cd signbert/model/thirdparty/manotorch
pip install .
```

# Run a training session
```bash
python train.py --config configs/pretrain.yml
```

# Run finetuning
```bash
python finetune.py --config finetune/configs/ISLR_MSASL.yml
--ckpt checkpoints/pretrain/ckpts/<CKPT_NAME>.ckpt
```

# Visualize logs with Tensorboard

```bash
tensorboard --logdir <LOGS_DPATH>
```

# Create visualization
First, install [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)

```bash
cd visualization
python create_visualization.py
```

If you find this code useful for your research, please consider citing:

    @article{hu2023signbert+,
        title={SignBERT+: Hand-model-aware Self-supervised Pre-training for Sign Language Understanding},
        author={Hu, Hezhen and Zhao, Weichao and Zhou, Wengang and Li, Houqiang},
        journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
        year={2023},
        publisher={IEEE}
    }
