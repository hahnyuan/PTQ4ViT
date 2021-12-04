# PTQ4ViT
Post-Training Quantization for Vision transformers.

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
To test on all models with BasePTQ/PTQ4ViT, run:
```bash
python example/test_all.py
```

To run ablation testing, run:
```bash
python example/test_ablation.py
```

### 2. Download quantized model checkpoints
(Coming soon)

## Results
Results of BasePTQ:

| model        | original | w8a8   | w6a6    |
|--------------|----------|--------|---------|
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
|--------------|----------|--------|---------|
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

## Citation
```
@article{PTQ4ViT_cvpr2022,
    title={PTQ4ViT: Post-Training Quantization Framework for Vision Transformers},
    author={Zhihang Yuan, Chenhao Xue, Yiqi Chen, Qiang Wu, Guangyu Sun},
    journal={arXiv preprint arXiv:2111.12293},
    year={2022},
}
```