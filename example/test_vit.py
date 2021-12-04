import sys
sys.path.insert(0,'..')
sys.path.insert(0,'.')
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from importlib import reload,import_module
import multiprocessing
import os
import time
from itertools import product

import utils.datasets as datasets
import utils.net_wrap as net_wrap
from utils.quant_calib import QuantCalibrator, HessianQuantCalibrator
from utils.models import get_net

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=6)
    parser.add_argument("--multiprocess", action='store_true')
    args = parser.parse_args()
    return args

def test_classification(net,test_loader,max_iteration=None, description=None):
    pos=0
    tot=0
    i = 0
    max_iteration = len(test_loader) if max_iteration is None else max_iteration
    with torch.no_grad():
        q=tqdm(test_loader, desc=description)
        for inp,target in q:
            i+=1
            inp=inp.cuda()
            target=target.cuda()
            out=net(inp)
            pos_num=torch.sum(out.argmax(1)==target).item()
            pos+=pos_num
            tot+=inp.size(0)
            q.set_postfix({"acc":pos/tot})
            if i >= max_iteration:
                break
    print(pos/tot)
    return pos/tot

def process(pid, experiment_process, args_queue, n_gpu):
    """
    worker process. 
    """
    gpu_id=pid%n_gpu
    os.environ['CUDA_VISIBLE_DEVICES']=f'{gpu_id}'

    tot_run=0
    while args_queue.qsize():
        test_args=args_queue.get()
        print(f"Run {test_args} on pid={pid} gpu_id={gpu_id}")
        experiment_process(**test_args)
        time.sleep(0.5)
        tot_run+=1
        # run_experiment(**args)
    print(f"{pid} tot_run {tot_run}")


def multiprocess(experiment_process, cfg_list=None, n_gpu=6):
    """
    run experiment processes on "n_gpu" cards via "n_gpu" worker process.
    "cfg_list" arranges kwargs for each test point, and worker process will fetch kwargs and carry out an experiment.
    """
    args_queue = multiprocessing.Queue()
    for cfg in cfg_list:
        args_queue.put(cfg)

    ps=[]
    for pid in range(n_gpu):
        p=multiprocessing.Process(target=process,args=(pid,experiment_process,args_queue,n_gpu))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()

def init_config(config_name):
    """initialize the config. Use reload to make sure it's fresh one!"""
    _,_,files =  next(os.walk("./configs"))
    if config_name+".py" in files:
        quant_cfg = import_module(f"configs.{config_name}")
    else:
        raise NotImplementedError(f"Invalid config name {config_name}")
    reload(quant_cfg)
    return quant_cfg
        

def experiment_basic(net='vit_base_patch16_384', config="PTQ4ViT"):
    """
    A basic testbench.
    """
    quant_cfg = init_config(config)
    net = get_net(net)
    wrapped_modules = net_wrap.wrap_modules_in_net(net,quant_cfg)
    
    g=datasets.ViTImageNetLoaderGenerator('/datasets/imagenet','imagenet',32,32,16,kwargs={"model":net})
    test_loader=g.test_loader()
    calib_loader=g.calib_loader(num=32)
    
    quant_calibrator = HessianQuantCalibrator(net,wrapped_modules,calib_loader,sequential=False,batch_size=4) # 16 is too big for ViT-L-16
    quant_calibrator.batching_quant_calib()
    
    test_classification(net,test_loader)

if __name__=='__main__':
    args = parse_args()
    cfg_list = []

    nets = ['vit_tiny_patch16_224', "deit_base_patch16_384"]
    configs= ['PTQ4ViT']

    cfg_list = [{
        "net":net,
        "config":config,
        }
        for net, config in product(nets, configs) 
    ]

    if args.multiprocess:
        multiprocess(experiment_basic, cfg_list, n_gpu=args.n_gpu)
    else:
        for cfg in cfg_list:
            experiment_basic(**cfg)
    

