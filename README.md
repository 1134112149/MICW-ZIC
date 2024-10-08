# Multi-method integration with Confidence-based weighting for Zero-shot Image classification(MICW-ZIC)

## Overview of MICW-ZIC

<p align="center"> <img src="./overview.png" width="100%"> </p>
## Environment

The code is developed and tested under the following environment:

-   Python 3.9
-   PyTorch 2.0.0
-   CUDA 11.7

You can create the environment via:

```
conda create -n lmc python=3.9
conda activate lmc
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install ftfy regex tqdm scikit-learn scipy pandas six timm
pip install transformers openai
pip install git+https://github.com/openai/CLIP.git
```



## Preparation

### Dataset

Decompress the following files in this directory: cifar10_dalle.zip, cifar100_dalle.zip, pic_DALL-E_all.zip, and pic_test.zip. They are the reference image of the cifar10 dataset, the reference image of the cifar100 dataset, the reference image of the TinyImage dataset and the test image of the TinyImage dataset.

### Pre-trained model

To load CLIP pre-trained weights, you can visit official [CLIP](https://github.com/openai/CLIP/) GitHub Repo and download CLIP "ViT-B/32" to `pretrained_model` using download address in [this page](https://github.com/openai/CLIP/blob/main/clip/clip.py). 

To load DINO pre-trained weights, you can visit official [DINOv2]() and download "ViT-B/14 distilled" to `pretrained_model` using download address in [this page](https://github.com/facebookresearch/dinov2#pretrained-models).



### Learn more

 [1134112149/MICW-ZIC (github.com)](https://github.com/1134112149/MICW-ZIC) 