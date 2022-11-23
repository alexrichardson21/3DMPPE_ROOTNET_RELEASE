#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import shutil
from loguru import logger

import tensorrt as trt
import torch
from torch2trt import torch2trt

from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn

from config import cfg
from model import get_pose_net

def make_parser():
    parser = argparse.ArgumentParser("YOLOX ncnn deploy")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "-w", '--workspace', type=int, default=32, help='max workspace size in detect'
    )
    parser.add_argument("-b", '--batch', type=int, default=1, help='max batch size in detect')
    return parser


@logger.catch
@torch.no_grad()
def main():
    args = make_parser().parse_args()
    cudnn.benchmark = True

    # snapshot load
    model = get_pose_net(cfg, False)
    model = DataParallel(model).cuda()
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['network'])
    model.eval()

    model.head.decode_in_inference = False
    x = torch.ones(1, 3, 256, 256).cuda()
    y = torch.ones(1, 1, 1).cuda()
    model_trt = torch2trt(
        model,
        [x, y],
        fp16_mode=True,
        log_level=trt.Logger.INFO,
        max_workspace_size=(1 << args.workspace),
        max_batch_size=args.batch,
    )
    torch.save(model_trt.state_dict(), os.path.join('ckpts', "model_trt.pth"))
    logger.info("Converted TensorRT model done.")
    engine_file = os.path.join('ckpts', "rootnet_model_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())

    logger.info("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
