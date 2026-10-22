# VideoFake / Face Swap & Face Enhance Video Colab (English) based on Roop without NSFW filter

UPD Oct 2026: Notebook and deps updated for **current Google Colab** (Python 3.12+/3.13, CUDA 12/13). Removed broken `nvidia-cudnn` apt install and old `numpy==1.24.3` / CUDA 11 pins. Use `roop/requirements-colab.txt` in Colab.

UPD feb 13 2025: [NVIDIA CUDA dependency for GPU and a new model host](https://github.com/neuromodern/VideoFake/releases/tag/1.0.1).

UPD: Current models url is updated (inswapper_128.onnx & GFPGANv1.4.pth). Pipeline was optimized for faster work. NSFW filter deleted.

This repository is a comprehensive collection of code and resources that allow you to perform face swapping and face enhancement on videos using Google Colab. With the power of deep learning and computer vision algorithms, this project enables you to seamlessly swap faces between individuals in a video and enhance facial features for a more visually appealing result.

The repository provides a user-friendly and interactive Colab notebook that guides you through the process step-by-step. It offers pre-trained models, datasets, and various image processing techniques to achieve accurate and realistic face swaps and enhancements. Whether you're interested in creating hilarious face-swapped videos or improving the appearance of facial attributes, this repository has got you covered.

[Open in Colab](https://colab.research.google.com/github/ixelizm/VideoFake/blob/main/VideoFake_colab.ipynb)

## Colab quick start

1. Open the notebook with a **GPU** runtime (`Runtime → Change runtime type → GPU`).
2. Run cells in order.
3. Cell **4** asks you to upload the face image and target video.

Do **not** install `nvidia-cudnn` via apt — Colab already ships CUDA/cuDNN; the old package conflicts with `libcudnn9` / CUDA 13.
