"""Setup."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2024, All Rights Reserved.
# ***
# ***    File Author: Dell, Wed 20 Mar 2024 10:20:29 AM CST
# ***
# ************************************************************************************/
#

from setuptools import setup

with open("README.md", "r") as file:
    long_description = file.read()

setup(
    name="video_consistent",
    version="1.0.0",
    author="Dell Du",
    author_email="18588220928@163.com",
    description="video consistent package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/delldu/CoDeF.git",
    packages=["video_consistent"],
    package_data={"video_consistent": ["models/video_consistent.pth"]},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch >= 1.5.0",
        "torchvision >= 0.6.0",
        "Pillow >= 7.2.0",
        "numpy >= 1.19.5",
    ],
)
