# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2024(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import os
import torch
import image_animation
import argparse
import todos
import pdb

def test_input_shape():
    import time
    import random
    from tqdm import tqdm

    print("Test input shape ...")

    model, device = image_animation.get_drive_face_generator_model()

    N = 100
    B, C, H, W = 1, 3, 256, 256

    mean_time = 0
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        kp1 = torch.randn(B, 50, 2)
        kp2 = torch.randn(B, 50, 2)

        x = torch.randn(B, C, H, W)

        start_time = time.time()
        with torch.no_grad():
            y = model(kp1.to(device), kp2.to(device), x.to(device))
        torch.cuda.synchronize()
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")


def run_bench_mark():
    print("Run benchmark ...")

    model, device = image_animation.get_drive_face_generator_model()
    N = 100
    B, C, H, W = 1, 3, 256, 256

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as p:
        for ii in range(N):
            image1 = torch.randn(B, C, H, W)
            image2 = torch.randn(B, C, H, W)
            with torch.no_grad():
                y = model(image1.to(device), image2.to(device))
            torch.cuda.synchronize()
        p.step()

    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    os.system("nvidia-smi | grep python")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smoke Test')
    parser.add_argument('-s', '--shape_test', action="store_true", help="test shape")
    parser.add_argument('-b', '--bench_mark', action="store_true", help="test benchmark")
    args = parser.parse_args()

    if args.shape_test:
        test_input_shape()
    if args.bench_mark:
        run_bench_mark()

    if not (args.shape_test or args.bench_mark):
        parser.print_help()
