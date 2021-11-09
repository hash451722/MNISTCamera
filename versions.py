import platform

import cv2 
import torch
import torchvision


def print_version():
    print("Python", platform.python_version())
    print("OpenCV", cv2.__version__)
    print("PyTorch", torch.__version__)
    print("torchvision", torchvision.__version__)

if __name__ == '__main__':
    print_version()