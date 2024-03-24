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
import video_consistent

import argparse
import pdb

def test_input_shape():
    import time
    import random
    from tqdm import tqdm

    print("Test input shape ...")

    model, device = video_consistent.get_model("output/model.pth")

    N = 100
    B, HW, C = 1, 512*512, 2
    dummy_xy = torch.randn(B, HW, C).to(device)
    dummy_time = torch.randn(B, 1).to(device)

    mean_time = 0
    progress_bar = tqdm(total=N)
    for count in range(N):
        progress_bar.update(1)

        h = random.randint(-16, 16)
        dummy_xy = torch.randn(B, HW + h, C)

        start_time = time.time()
        with torch.no_grad():
            y = model(dummy_xy.to(device), dummy_time.to(device))
        if 'cpu' not in str(device):
            torch.cuda.synchronize()
        mean_time += time.time() - start_time

    mean_time /= N
    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")


def run_bench_mark():
    print("Run benchmark ...")

    model, device = video_consistent.get_model("output/model.pth")
    N = 100

    B, HW, C = 1, 512*512, 2
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as p:
        for ii in range(N):
            dummy_xy = torch.randn(B, HW, C).to(device)
            dummy_time = torch.randn(B, 1).to(device)
            with torch.no_grad():
                y = model(dummy_xy, dummy_time)
            if 'cpu' not in str(device):
                torch.cuda.synchronize()
        p.step()

    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    os.system("nvidia-smi | grep python")


def export_onnx_model():
    # https://github.com/onnx/onnx/blob/main/docs/Operators.md

    import onnx
    import onnxruntime
    from onnxsim import simplify
    import onnxoptimizer
    
    print("Export onnx model ...")

    # 1. Run torch model
    model, device = video_consistent.get_model("output/model.pth")

    B, HW, C = 1, 512*512, 2
    dummy_xy = torch.randn(B, HW, C).to(device)
    dummy_time = torch.randn(B, 1).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_xy, dummy_time)
    torch_outputs = [dummy_output.cpu()]

    # 2. Export onnx model
    input_names = [ "xy", "time" ]
    output_names = [ "output" ]
    dynamic_axes = { 
        'xy' : {1: 'batch'}, 
        'output' : {1: 'batch'} 
    } 
    onnx_filename = "output/video_consistent.onnx"

    torch.onnx.export(model, 
        (dummy_xy, dummy_time), 
        onnx_filename, 
        verbose=False, 
        input_names=input_names, 
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # 3. Check onnx model file
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

    onnx_model, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx_model = onnxoptimizer.optimize(onnx_model)    
    onnx.save(onnx_model, onnx_filename)
    # print(onnx.helper.printable_graph(onnx_model.graph))

    # 4. Run onnx model
    if 'cuda' in device.type:
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CUDAExecutionProvider'])
    else:        
        ort_session = onnxruntime.InferenceSession(onnx_filename, providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    onnx_inputs = {input_names[0]: to_numpy(dummy_xy),
                   input_names[1]: to_numpy(dummy_time) }
    onnx_outputs = ort_session.run(None, onnx_inputs)

    # 5.Compare output results
    assert len(torch_outputs) == len(onnx_outputs)
    for torch_output, onnx_output in zip(torch_outputs, onnx_outputs):
        torch.testing.assert_close(torch_output, torch.tensor(onnx_output), rtol=0.01, atol=0.01)

    print("!!!!!! Torch and ONNX Runtime output matched !!!!!!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Smoke Test')
    parser.add_argument('-s', '--shape_test', action="store_true", help="test shape")
    parser.add_argument('-b', '--bench_mark', action="store_true", help="test benchmark")
    parser.add_argument('-e', '--export_onnx', action="store_true", help="export onnx model")
    args = parser.parse_args()

    if args.shape_test:
        test_input_shape()
    if args.bench_mark:
        run_bench_mark()
    if args.export_onnx:
        export_onnx_model()

    
    if not (args.shape_test or args.bench_mark or args.export_onnx):
        parser.print_help()
