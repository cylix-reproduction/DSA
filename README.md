# DSA

Reproduction of the paper [*Transferability of White-box Perturbations: Query-Efficient Adversarial Attacks against Commercial DNN Services*](https://www.usenix.org/conference/usenixsecurity24/presentation/shen-meng), forked from the [original repo](https://github.com/lcycode/DSA).

> **NOTE**
>
> This is a *reproduction* repo. If you're working on the basis of the paper, please don't forget to cite the authors. See [citation](#-citation) for detail.
>
> To show respect for the original work, the original content is placed at the end of this README.

- [DSA](#dsa)
  - [Setting up environments](#setting-up-environments)
    - [Installing UV](#installing-uv)
    - [Preparing dataset](#preparing-dataset)
    - [Syncing dependencies](#syncing-dependencies)
  - [Running demo](#running-demo)
- [\<Original README\> Transferability of White-box Perturbations: Query-Efficient Adversarial Attacks against Commercial DNN Services](#original-readme-transferability-of-white-box-perturbations-query-efficient-adversarial-attacks-against-commercial-dnn-services)
- [🔥Design](#design)
- [🚀QuickStart](#quickstart)
  - [Requirements](#requirements)
  - [Config](#config)
  - [Run](#run)
- [📄 Citation](#-citation)
- [🎉Compatibility](#compatibility)
- [💡 Questions?](#-questions)

## Setting up environments
This repo replaced the old-fashioned `requirements.txt` with the [PEP 517](https://peps.python.org/pep-0517/) defined `pyproject.toml`, which is recommended by the official guide. You should use a package manager (UV, PDM, Poetry) to resolve this. This project is organized using [UV](https://docs.astral.sh/uv).

### Installing UV
If you don't have UV installed, use the following command to install it.

**macOS/Linux**
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Cargo**
```shell
cargo install --git https://github.com/astral-sh/uv uv
```

For more installation methods, see the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Preparing dataset

Currently, the code only supports the [ImageNet](https://image-net.org/) dataset. More precisely, the [ILSVRC 2012](https://image-net.org/challenges/LSVRC/2012/index.php) dataset is used. According to [`torchvision.dataset.ImageNet`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html), it is required to download ImageNet 2012 dataset from [here](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and place the following files:

- `ILSVRC2012_devkit_t12.tar.gz`
- `ILSVRC2012_img_train.tar`
- `ILSVRC2012_img_test.tar`
- `ILSVRC2012_img_val.tar`

into a specific directory, and specify the path of this directory in [`config/attack_config.py`](config/attack_config.py).

### Syncing dependencies
This repo configures all the dependencies and the Python interpreter for you. You just need to run

```shell
uv sync --extra cpu
```

to create a virtual environment, install dependencies including a CPU version of PyTorch. For CUDA 11.8, 12.6 and 12.8, use

```shell
uv sync --extra cu118
# or
uv sync --extra cu126
# or
uv sync --extra cu128
```

to install the corresponding PyTorch version respectively.

## Running demo
If the environment is configured according to the instructions above, you can use

```shell
uv run demo.py
```

to run the provided demo. The required pretrained models will be downloaded from the Internet automatically, so make sure your network condition is fine.

# <mark>&lt;Original README&gt;</mark> Transferability of White-box Perturbations: Query-Efficient Adversarial Attacks against Commercial DNN Services

This is a PyTorch implementation of DSA, a powerful black-box adversarial attack, as described in the following:
> Meng Shen, Changyue Li, Qi Li, Hao Lu, Liehuang Zhu, and Ke Xu. Transferability of Whitebox Perturbations: Query-Efficient Adversarial Attacks against Commercial DNN Services. Proceedings of the 33rd USENIX Conference on Security Symposium (USENIX Security), 2024.

This repository aims to provide a simple demo to show how to use DSA to attack neural networks.

*Note that this demo may not consume the full query budget. With this demo, you can obtain a candidate adversarial example, which can be used as the starting point with the remaining query budget for existing query-based attacks, e.g., [SurFree](https://openaccess.thecvf.com/content/CVPR2021/papers/Maho_SurFree_A_Fast_Surrogate-Free_Black-Box_Attack_CVPR_2021_paper.pdf), and [HSJA](https://ieeexplore.ieee.org/document/9152788).*



#  🔥Design

The key components of this repository are as follows:

The `./demo.py` launches the adversarial attack against VGG16 using three substitute models, i.e.,  DenseNet, Inc-v3, IncRes-v2. These models are pre-trained and automatically downloaded on the first load.

The `./attack/dsa.py` is the main implementation of the DSA method.

The `./config/attack_config.py` contains the key parameters of the attack.



# 🚀QuickStart 

## Requirements

DSA is tested with Python >= 3.10 and  PyTorch >= 1.12. You can see the versions we currently use for testing in the [Compatibility section](#Compatibility) below, but newer versions are in general expected to work.

Other dependencies can be installed using the following command:

```bash 
pip install timm==0.9.16 pretrainedmodels==0.7.4 yacs==0.1.8 six==1.16 numpy==1.23.5
```

## Config

**Data**: As this demo uses two images with different labels from ImageNet, it is required to set your ImageNet path in `cfg.dataset.root_dir` . If ImageNet is unavailable, you can manually load two images by changing the function `load_images` in  `demo.py`.

**Model**: You can change `cfg.model.root_path` to indicate the download path of pre-trained models, which is the default torch_home when set to None. You can set different model types by changing `target_model` and `substitute_models`, and see available model types in [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch) and [timm](https://github.com/huggingface/pytorch-image-models).

## Run

You can use the following command to launch the attack.

```bash
python demo.py
```




# 📄 Citation
If you use this repository for your work, please cite our paper using the following BibTeX entries:
>@inproceedings {298048, 
author = {Meng Shen and Changyue Li and Qi Li and Hao Lu and Liehuang Zhu and Ke Xu}, 
title = {Transferability of White-box Perturbations: {Query-Efficient} Adversarial Attacks against Commercial {DNN} Services}, 
booktitle = {33rd USENIX Security Symposium (USENIX Security 24)}, 
year = {2024},
isbn = {978-1-939133-44-1},
address = {Philadelphia, PA},
pages = {2991--3008},
url = {https://www.usenix.org/conference/usenixsecurity24/presentation/shen-meng}, 
publisher = {USENIX Association},
month = aug
}



# 🎉Compatibility

We tested with the following versions:

* Python: 3.10, 3.11
* PyTorch: 1.12, 1.13, 2.0, 2.1
* Numpy:  1.23, 1.24



# 💡 Questions?

If you have a question or need help, please feel free to open an issue on GitHub, or contact me at lichangyue98@163.com.

