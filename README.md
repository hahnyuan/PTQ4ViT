# PTQ4ViT
Post-Training Quantization Framework for Vision Transformers.
We use the twin uniform quantization method to reduce the quantization error on these activation values.
And we use a Hessian guided metric to evaluate different scaling factors, which improves the accuracy of calibration with a small cost.
The quantized vision transformers (ViT, DeiT, and Swin) achieve near-lossless prediction accuracy (less than 0.5\% drop at 8-bit quantization) on the ImageNet classification task. Please read the [paper](https://arxiv.org/abs/2111.12293) for details.

## Updates

*15/01/2022*
Add saved quantized models with PTQ4ViT.
| model        |   link   |
|:------------:|:--------:|
| ViT-S/224/32 | [Google](https://drive.google.com/file/d/195JJJKULvaukte6PA9U08oezjd176CTs/view?usp=sharing)   |
| ViT-S/224    | [Google](https://drive.google.com/file/d/14uEDgRmDBYoKoZtpO9IWMfG8Uvkt_OuL/view?usp=sharing)   |
| ViT-B/224    | [Google](https://drive.google.com/file/d/1ou6s9Vd-_iyQ7sj7VYET-pRvJA6WMMLA/view?usp=sharing)   |
| ViT-B/384    | [Google](https://drive.google.com/file/d/1tuU8or8SfQomtoWam7WFTnUxtuw3n7fs/view?usp=sharing)   |
| DeiT-S/224   | [Google](https://drive.google.com/file/d/1673fX-SuiRlHhm7k0Yyyx_3ynwtvUPyf/view?usp=sharing)   |
| DeiT-B/224   | [Google](https://drive.google.com/file/d/1WRAtmPF0kDR9iTLc9gv_63aEkOCZ_zOI/view?usp=sharing)   |
| DeiT-B/384   | [Google](https://drive.google.com/file/d/1mPPlM2ioe4zts_rdKdjZTCUj8KcbquyA/view?usp=sharing)   |
| Swin-T/224   | [Google](https://drive.google.com/file/d/1bSahHgtL3yFaHPlG-SDtu__YY0zJ8lxr/view?usp=sharing)   |
| Swin-S/224   | [Google](https://drive.google.com/file/d/1SxAdDTwQaeJFWnHLFXncVocxMNBIPDOE/view?usp=sharing)   |
| Swin-B/224   | [Google](https://drive.google.com/file/d/19UUUQYJGs5SQaDe27PjY3x1QTBU5hwXm/view?usp=sharing)   |
| Swin-B/384   | [Google](https://drive.google.com/file/d/1SxAdDTwQaeJFWnHLFXncVocxMNBIPDOE/view?usp=sharing)   |

*10/12/2021*
Add `utils/integer.py`, you can now:
1. convert calibrated fp32 model into int8
2. register pre-forward hook in the model, and fetch activation in int8. (We use uint8 to store results
    of twin quantization, please refer to the paper to see the bits' layout).

## Install

### Requirement 
- python>=3.5
- pytorch>=1.5
- matplotlib
- pandas
- timm

### Datasets
To run example testing, you should put your ImageNet2012 dataset in path `/datasets/imagenet`.

We use `ViTImageNetLoaderGenerator` in `utils/datasets.py` to initialize our DataLoader.
If your Imagenet datasets are stored elsewhere, you'll need to manually pass its root as an argument when instantiating a `ViTImageNetLoaderGenerator`.

## Usage

### 1. Run example quantization
To test on all models with BasePTQ/PTQ4ViT, run
```bash
python example/test_all.py
```

To run ablation testing, run
```bash
python example/test_ablation.py
```

You can run the testing scripts with multiple GPUs. For example, calling
```bash
python example/test_all.py --multigpu --n_gpu 6
```
will use 6 gpus to run the test.

### 2. Download quantized model checkpoints
(Coming soon)

## Results
### Results of BasePTQ

| model        | original | w8a8   | w6a6    |
|:------------:|:--------:|:------:|:-------:|
| ViT-S/224/32 | 75.99    | 73.61  | 60.144  |
| ViT-S/224    | 81.39    | 80.468 | 70.244  |
| ViT-B/224    | 84.54    | 83.896 | 75.668  |
| ViT-B/384    | 86.00    | 85.352 | 46.886  |
| DeiT-S/224   | 79.80    | 77.654 | 72.268  |
| DeiT-B/224   | 81.80    | 80.946 | 78.786  |
| DeiT-B/384   | 83.11    | 82.33  | 68.442  |
| Swin-T/224   | 81.39    | 80.962 | 78.456  |
| Swin-S/224   | 83.23    | 82.758 | 81.742  |
| Swin-B/224   | 85.27    | 84.792 | 83.354  |
| Swin-B/384   | 86.44    | 86.168 | 85.226  |

Results of PTQ4ViT

| model        | original | w8a8   | w6a6    |
|:------------:|:--------:|:------:|:-------:|
| ViT-S/224/32 | 75.99    | 75.582 | 71.908  |
| ViT-S/224    | 81.39    | 81.002 | 78.63   |
| ViT-B/224    | 84.54    | 84.25  | 81.65   |
| ViT-B/384    | 86.00    | 85.828 | 83.348  |
| DeiT-S/224   | 79.80    | 79.474 | 76.282  |
| DeiT-B/224   | 81.80    | 81.482 | 80.25   |
| DeiT-B/384   | 83.11    | 82.974 | 81.55   |
| Swin-T/224   | 81.39    | 81.246 | 80.47   |
| Swin-S/224   | 83.23    | 83.106 | 82.38   |
| Swin-B/224   | 85.27    | 85.146 | 84.012  |
| Swin-B/384   | 86.44    | 86.394 | 85.388  |

### Results of Ablation
- ViT-S/224 (original top-1 accuracy 81.39%)

| Hessian Guided | Softmax Twin | GELU Twin | W8A8   | W6A6    |
|:--------------:|:------------:|:---------:|:------:|:-------:|
|                |              |           | 80.47  | 70.24   |
| ✓              |              |           | 80.93  | 77.20   |
| ✓              | ✓            |           | 81.11  | 78.57   |
| ✓              |              | ✓         | 80.84  | 76.93   |
|                | ✓            | ✓         | 79.25  | 74.07   |
| ✓              | ✓            | ✓         | 81.00  | 78.63   |

- ViT-B/224 (original top-1 accuracy 84.54%)

| Hessian Guided | Softmax Twin | GELU Twin | W8A8   | W6A6    |
|:--------------:|:------------:|:---------:|:------:|:-------:|
|                |              |           | 83.90  | 75.67   |
| ✓              |              |           | 83.97  | 79.90   |
| ✓              | ✓            |           | 84.07  | 80.76   |
| ✓              |              | ✓         | 84.10  | 80.82   |
|                | ✓            | ✓         | 83.40  | 78.86   |
| ✓              | ✓            | ✓         | 84.25  | 81.65   |

- ViT-B/384 (original top-1 accuracy 86.00%)

| Hessian Guided | Softmax Twin | GELU Twin | W8A8   | W6A6    |
|:--------------:|:------------:|:---------:|:------:|:-------:|
|                |              |           | 85.35  | 46.89   |
| ✓              |              |           | 85.42  | 79.99   |
| ✓              | ✓            |           | 85.67  | 82.01   |
| ✓              |              | ✓         | 85.60  | 82.21   |
|                | ✓            | ✓         | 84.35  | 80.86   |
| ✓              | ✓            | ✓         | 85.89  | 83.19   |

## Citation
```
@article{PTQ4ViT_arixv2022,
    title={PTQ4ViT: Post-Training Quantization Framework for Vision Transformers},
    author={Zhihang Yuan, Chenhao Xue, Yiqi Chen, Qiang Wu, Guangyu Sun},
    journal={arXiv preprint arXiv:2111.12293},
    year={2022},
}
```
