import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
from example.test_vit import *
import utils.net_wrap as net_wrap
import utils.datasets as datasets
import utils.integer as integer
from utils.quant_calib import HessianQuantCalibrator

from itertools import product

def get_int_weights(name, config_name):
    quant_cfg = init_config(config_name)

    net = get_net(name)

    wrapped_modules=net_wrap.wrap_modules_in_net(net,quant_cfg)
    
    g=datasets.ViTImageNetLoaderGenerator('/datasets/imagenet','imagenet',32,32,16, kwargs={"model":net})
    test_loader=g.test_loader()
    calib_loader=g.calib_loader(num=32)
    
    quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
    quant_calibrator.batching_quant_calib()

    int_weights = integer.get_model_int_weight(wrapped_modules)
    torch.save(int_weights, f"./int_weights/{name}.pth")


if __name__ == "__main__":
    args = parse_args()

    names = [
        # "vit_tiny_patch16_224",
        # "vit_small_patch32_224",
        # "vit_small_patch16_224",
        # "vit_base_patch16_224",
        "vit_base_patch16_384",

        # "deit_tiny_patch16_224",
        # "deit_small_patch16_224",
        # "deit_base_patch16_224",
        # "deit_base_patch16_384",

        # "swin_tiny_patch4_window7_224",
        # "swin_small_patch4_window7_224",
        # "swin_base_patch4_window7_224",
        # "swin_base_patch4_window12_384",
        ]
    config_names = ["PTQ4ViT", "BasePTQ"]

    cfg_list = []
    for name, config in product(names, config_names):
        cfg_list.append({"name":name, "config_name":config})
    
    if args.multiprocess:
        multiprocess(get_int_weights, cfg_list, n_gpu=args.n_gpu)
    else:
        for cfg in cfg_list:
            get_int_weights(**cfg)