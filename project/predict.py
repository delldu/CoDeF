"""Model trainning."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2024, All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 20 Mar 2024 11:50:36 AM CST
# ***
# ************************************************************************************/
#
import os
import argparse
import video_consistent
import pdb  # For debug


if __name__ == "__main__":
    """Trainning model."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--checkpoint", type=str, default="output/model.pth", help="checkpoint file")
    parser.add_argument("-r", "--reference", type=str, default="output/canonical.png", help="reference image file")
    parser.add_argument("-o", "--output", type=str, default="output/reference", help="output directory")
    args = parser.parse_args()

    assert os.path.exists(args.checkpoint), f"Model file {args.checkpoint} not exit."
    net, device = video_consistent.get_model(args.checkpoint)
    net.eval()

    if len(args.reference) > 0:
        video_consistent.video_sample(args.reference, net, device, args.output)
    else:
        video_consistent.video_restruct(net, device, args.output)

