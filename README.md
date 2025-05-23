# Delving into Out-of-distribution Detection with Vision-Language Representations

This codebase provides a Pytorch implementation for the paper [Delving into Out-Of-Distribution Detection with Vision-Language Representations](https://openreview.net/forum?id=KnCS9390Va) at NeurIPS 2022.

### Abstract

Recognizing out-of-distribution (OOD) samples is critical for machine learning systems deployed in the open world. The vast majority of OOD detection methods are driven by a single modality (e.g., either vision or language), leaving the rich information in multi-modal representations untapped. Inspired by the recent success of vision-language pre-training, this paper enriches the landscape of OOD detection from a single-modal to a multi-modal regime. Particularly, we propose Maximum Concept Matching (MCM), a simple yet effective zero-shot OOD detection method based on aligning visual features with textual concepts. We contribute in-depth analysis and theoretical insights to understand the effectiveness of MCM. Extensive experiments demonstrate that MCM achieves superior performance on a wide variety of real-world tasks. MCM with vision-language features outperforms a common baseline with pure visual features on a hard OOD task with semantically similar classes by 13.1% (AUROC). 

### Illustration

![Arch_figure](readme_figs/Arch_figure.png)


### Update
[Aug 12] 🔥 If you are interested in how **parameter-efficient fine-tuning** impacts OOD detection for vision-language models, feel free to check out our recent work [How Does Fine-Tuning Impact Out-of-Distribution Detection for Vision-Language Models?](https://arxiv.org/abs/2306.06048) to appear on International Journal of Computer Vision (IJCV), 2023. The code will be released in a separate repo. 

# Setup

## Required Packages

Our experiments are conducted on Ubuntu Linux 20.04 with Python 3.8 and Pytorch 1.10. Besides, the following commonly used packages are required to be installed:

- [transformers](https://huggingface.co/docs/transformers/installation), scipy, scikit-learn, matplotlib, seaborn, pandas, tqdm. 

## Checkpoints

We use the publicly available checkpoints from Hugging Face where the ViT model is pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k. For example, the checkpoint for ViT-B is available [here](https://huggingface.co/google/vit-base-patch16-224). 

For CLIP models, our reported results are based on checkpoints provided by Hugging Face for [CLIP-B](https://huggingface.co/openai/clip-vit-base-patch16) and [CLIP-L](https://huggingface.co/openai/clip-vit-large-patch14). Similar results can be obtained with checkpoints in the codebase by [OpenAI](https://github.com/openai/CLIP). 



# Data Preparation

For complete information, refer to Appendix B in the paper. The default dataset location is `./datasets`.

## In-distribution Datasets

We consider the following (in-distribution) datasets:

- [`CUB-200`](http://www.vision.caltech.edu/datasets/cub_200_2011/), [`Standford-Cars`](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [`Food-101`](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), [`Oxford-Pet`](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- `ImageNet-1k`, `ImageNet-10`, `ImageNet-20`, `ImageNet-100`

The ImageNet-1k dataset (ILSVRC-2012) can be downloaded [here](https://image-net.org/challenges/LSVRC/2012/index.php#). ImageNet-10, ImageNet-20, and ImageNet-100 can be generated given the class names and IDs provided in `data/ImageNet10/ImageNet-10-classlist.csv` , `data/ImageNet20/ImageNet-20-classlist.csv`, and `data/ImageNet100/class_list.txt` respectively. The other datasets will be automatically downloaded.

The iNaturalist can be downloaded from: https://github.com/visipedia/inat_comp/tree/master/2021 
Specifically, 'Train Mini Images [42GB]','Train Mini Annotations [45MB]' and 'Validation Images [8.4GB]','Validation Annotations [9.4MB]'. Extract them to the `./datasets` folder.




### Realistic semantically hard ID and OOD datasets 

OOD samples that are semantically similar to ID samples are particularly challenging for OOD detection algorithms. However, it is common for the community to use CIFAR-10 (ID) and CIFAR-100 (OOD) as benchmark ID-OOD pairs, which contain low-resolution images that are less realistic.  

Therefore, to evaluate hard OOD detection tasks in realistic settings, we consider ImageNet-10 (ID) vs. ImageNet-20 (OOD) and vice versa. The pair consists of *high-resolution* images with semantically similar categories.  For example, we construct ImageNet-10 that mimics the class distribution of CIFAR-10. For hard OOD evaluation, we curate ImageNet-20, which consists of 20 classes semantically similar to ImageNet-10 (e.g., dog (ID) vs. wolf (OOD)).

To create ImageNet-10, 20, and 100, the following script can be used:

```python
# ImageNet-10 
python create_imagenet_subset.py --in_dataset ImageNet10 --src-dir datasets/ImageNet --dst-dir datasets
# ImageNet-20
python create_imagenet_subset.py --in_dataset ImageNet20 --src-dir datasets/ImageNet --dst-dir datasets
# ImageNet-100
python create_imagenet_subset.py --in_dataset ImageNet100 --src-dir datasets/ImageNet --dst-dir datasets
```

**Remarks**: As ImageNet-10 is less fine-grained than ImageNet-1k, the class n02422699 is referred to as antelope, originally named 'impala' (medium-sized antelopes) in ImageNet-1k, and class n02389026 as horse, originally named 'sorrel' (lighter-shade red horses) in ImageNet-1k. The original names from ImageNet-1k can be used if preferred.

Subdatasets can be constructed from iNaturalist, including 'Manzanita', 'Wild Rye', 'Wrasse', 'Lichen', and 'Bulrush'. The following script can be used to create these subdatasets:
```python
# Manzanita 
python create_imagenet_subset.py --in_dataset Manzanita --src-dir datasets/ImageNet --dst-dir datasets
# Wild Rye
python create_imagenet_subset.py --in_dataset Wild_rye --src-dir datasets/ImageNet --dst-dir datasets
# Wrasse
python create_imagenet_subset.py --in_dataset Wrasse --src-dir datasets/ImageNet --dst-dir datasets
# Bulrush
python create_imagenet_subset.py --in_dataset Bulrush --src-dir datasets/ImageNet --dst-dir datasets
# Lichen
python create_imagenet_subset.py --in_dataset Lichen --src-dir datasets/ImageNet --dst-dir datasets
```


## Out-of-Distribution Datasets

We use the large-scale OOD datasets [iNaturalist](https://arxiv.org/abs/1707.06642), [SUN](https://vision.princeton.edu/projects/2010/SUN/), [Places](https://arxiv.org/abs/1610.02055), and [Texture](https://arxiv.org/abs/1311.3618) curated by [Huang et al. 2021](https://arxiv.org/abs/2105.01879). Please follow instruction from the this [repository](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) to download the subsampled datasets where semantically overlapped classes with ImageNet-1k are removed.

The overall file structure is as follows:

```
MCM
|-- datasets
    |-- ImageNet
    |-- ImageNet10
    |-- ImageNet20
    |-- ImageNet100
    |-- iNaturalist
    |-- Manzanita
    |-- Wild_rye
    |-- Wrasse
    |-- Bulrush
    |-- Lichen
    |-- ImageNet_OOD_dataset
        |-- iNaturalist
        |-- dtd
        |-- SUN
        |-- Places
    ...
```

# Quick Start

The main script for evaluating OOD detection performance is `eval_ood_detection.py`. Here are the list of arguments:

- `--name`: A unique ID for the experiment, can be any string
- `--score`: The OOD detection score, which accepts any of the following:
  - `MCM`: Maximum Concept Matching score
  - `energy`: The [Energy score](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html)
  - `max-logit`: Max Logit score (i.e., cosine similarity without softmax)
  - `entropy`: Negative entropy of softmax scaled cosine similarities
  - `var`: Variance of cosine similarities
- `--seed`: A random seed for the experiments
- `--gpu`: The index of the GPU to use. For example `--gpu=0`
- `--in_dataset`: The in-distribution dataset
  - Accepts:  `ImageNet`, `ImageNet10`, `ImageNet20`, `ImageNet100`, `bird200`, `car196`, `flower102`, `food101` , `pet37`,`lichen_in`,`manzanita_in`,`bulrush_in`,`wild_rye_in`,`wrasse_in`,`lichen`,`wrasse`,`manzanita`,`bulrush`,`wild_rye`
- `-b`, `--batch_size`: Mini-batch size
- `--CLIP_ckpt`: Specifies the pre-trained CLIP encoder to use
  - Accepts: `ViT-B/32`, `ViT-B/16`, `ViT-L/14`.

The OOD detection results will be generated and stored in  `results/in_dataset/score/CLIP_ckpt/name/`. 

We provide bash scripts to help reproduce the numerical results of our paper and facilitate future research.  For example, to evaluate the performance of MCM score on ImageNet-10, with an experiment name `eval_ood`: 

```sh
sh scripts/eval_mcm.sh eval_ood ImageNet10 MCM
```
For example, to evaluate the performance of MCM score on lichen, with an experiment name `eval_ood`:

```sh
sh scripts/eval_mcm.sh eval_ood lichen MCM
```

For example, to evaluate the performance of MCM score on lichen, with an experiment name `eval_ood` and evaluation with descriptions:

```sh
sh scripts/eval_mcm_with_descriptors.sh eval_ood lichen MCM True
```
Split the lichen class into `lichen_in` and `lichen_out`, and evaluate the OOD detection performance of the MCM score between them using the experiment name `eval_ood`.

Example Usage:

Set `lichen_in` to 3 classes and evaluate with descriptions:

```sh
sh scripts/eval_mcm_subcls.sh eval_ood lichen_in MCM 3 True
```

Set `lichen_in` to 2 classes and evaluate without descriptions:

```sh
sh scripts/eval_mcm_subcls.sh eval_ood lichen_in MCM 2 False
```

### Citation

If you find our work useful, please consider citing our paper:

```
@inproceedings{ming2022delving,
  title={Delving into Out-of-Distribution Detection with Vision-Language Representations},
  author={Ming, Yifei and Cai, Ziyang and Gu, Jiuxiang and Sun, Yiyou and Li, Wei and Li, Yixuan},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

```
@article{ming2023finetune,
  title={How Does Fine-Tuning Impact Out-of-Distribution Detection for Vision-Language Models?},
  author={Ming, Yifei and Li, Yixuan},
  journal={International Journal of Computer Vision},
  year={2023}
}
```
