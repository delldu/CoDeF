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
    parser.add_argument("-a", "--atlas", type=str, default="output/atlas.png", help="atlas image file")
    parser.add_argument("-o", "--output", type=str, default="output/atlas", help="output directory")
    args = parser.parse_args()

    assert os.path.exists(args.checkpoint), f"Model file {args.checkpoint} not exit."
    net, device = video_consistent.get_model(args.checkpoint)
    net.eval()

    if len(args.atlas) > 0:
        video_consistent.video_sample(args.atlas, net, device, args.output)
    else:
        video_consistent.video_restruct(net, device, args.output)

