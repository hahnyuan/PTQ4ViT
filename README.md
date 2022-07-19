# PTQ4ViT
Post-Training Quantization Framework for Vision Transformers.
We use the twin uniform quantization method to reduce the quantization error on these activation values.
And we use a Hessian guided metric to evaluate different scaling factors, which improves the accuracy of calibration with a small cost.
The quantized vision transformers (ViT, DeiT, and Swin) achieve near-lossless prediction accuracy (less than 0.5\% drop at 8-bit quantization) on the ImageNet classification task. Please read the [paper](https://arxiv.org/abs/2111.12293) for details.

## Updates

*19/07/2022*
Add discussion on Base PTQ, and provide more ablation study results.

### Number of Calibration Images

| Model        | W8A8 #ims=32 | W6A6 #ims=32 | W8A8 #ims=128 | W6A6 #ims=128 |
|:------------:|:------------:|:------------:|:-------------:|:-------------:|
| ViT-S/224/32 | 75.58        | 71.91        |  75.54        | 72.29         |
| ViT-S/224    | 81.00        | 78.63        |  80.99        | 78.44         |
| ViT-B/224    | 84.25        | 81.65        |  84.27        | 81.84         |
| ViT-B/384    | 85.83        | 83.35        |  85.81        | 83.84         |
| DeiT-S/224   | 79.47        | 76.28        |  79.41        | 76.51         |
| DeiT-B/224   | 81.48        | 80.25        |  81.54        | 80.30         |
| DeiT-B/384   | 82.97        | 81.55        |  83.01        | 81.67         |
| Swin-T/224   | 81.25        | 80.47        |  81.27        | 80.30         |
| Swin-S/224   | 83.11        | 82.38        |  83.15        | 82.38         |
| Swin-B/224   | 85.15        | 84.01        |  85.17        | 84.15         |
| Swin-B/384   | 86.39        | 85.39        |  86.36        | 85.45         |

| Model        | Time #ims=32 | Time #ims=128 |
|:------------:|:------------:|:-------------:|
| ViT-S/224/32 | 2 min        | 5 min         |
| ViT-S/224    | 3 min        | 7 min         |
| ViT-B/224    | 4 min        | 13 min        |
| ViT-B/384    | 12 min       | 43 min        |
| DeiT-S/224   | 3 min        | 7 min         |
| DeiT-B/224   | 4 min        | 16 min        |
| DeiT-B/384   | 14 min       | 52 min        |
| Swin-T/224   | 3 min        | 9 min         |
| Swin-S/224   | 8 min        | 17 min        |
| Swin-B/224   | 10 min       | 23 min        |
| Swin-B/384   | 25 min       | 69 min        |

One of the targets of PTQ4ViT is to quickly quantize a vision transformer. 
We have proposed to pre-compute the output and gradient of each layer and compute the influence of scaling factor candidates in batches to reduce the quantization time. 
As demonstrated in the second table, PTQ4ViT can quantize most vision transformers in several minutes using 32 calibration images. 
Using 128 calibration images significantly  increases  the  quantization  time.  
We observe the Top-1 accuracy varies slightly in the first table, demonstrating PTQ4ViT is not very sensitive to the number of calibration images.

### Base PTQ
Base PTQ is a simple quantization strategy and serves as a benchmark for our experiments. 
Like PTQ4ViT, we quantize all weights and inputs for fully-connect layers (including the first projection layer and the last prediction layer), as well as all input matrices of matrix multiplication operations. 
For fully-connected layers, we use layerwise scaling factors $\Delta_W$ for weight quantization and $\Delta_X$ for input quantization; while for matrix multiplication operations, we use $\Delta_A$ and $\Delta_B$ for A's quantization and B's quantization respectively. 

To get the best scaling factors, we apply a linear grid search on the search space. 
The same as EasyQuantand Liu et al., we take hyper-parameters $\alpha=0.5$, $\beta = 1.2$, one search round and use cosine distance as the metric. 
Note that in PTQ4ViT, we change the hyper-parameters to $\alpha=0$, $\beta = 1.2$ and three search rounds, which slightly improves the performance.

It should be noticed that Base PTQ adopts a parallel quantization paradigm, which makes it essentially different from sequential quantization paradigms such as EasyQuant. 
In sequential quantization, the input data of the current quantizing layer is generated with all previous layers quantizing weights and activations. 
While in parallel quantization, the input data of the current quantizing layer is simply the raw output of the previous layer. 

In practice, we found sequential quantization on vision transformers suffers from significant accuracy degradation on small calibration datasets. 
While parallel quantization shows robustness on small calibration datasets. 
Therefore, we choose parallel quantization for both Base PTQ and PTQ4ViT.

### More Ablation Study

We supply more ablation studies for the hyper-parameters.
It is enough to set the number of quantization intervals $\ge$ 20 (accuracy change $< 0.3\%$).
It is enough to set the upper bound of m $\ge$ 15 (no accuracy change).
The best settings of alpha and beta vary from different layers. 
It is appropriate to set $\alpha=0$ and $\beta=1/2^{k-1}$, which has little impact on search efficiency.
We observe that search rounds has little impact on the prediction accuracy (accuracy change $<$ 0.05\% when search rounds $>1$).

We randomly take 32 calibration images to quantize different models 20 times and we observe the fluctuation is not significant. 
The mean/std of accuracies are: ViT-S/32 $75.55\%/0.055\%$ , ViT-S $80.96\%/0.046\%$, ViT-B $84.12\%/0.068\%$, DeiT-S $79.45\%/0.094\%$ , and Swin-S $83.11\%/0.035\%$.


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
