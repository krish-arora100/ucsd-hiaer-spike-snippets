# HiAER-Spike FPGA Deployment

Code for training, quantizing, and converting spiking and analog neural networks to run on **HiAER-Spike**, an FPGA-based neuromorphic computing platform built at UCSD's Integrated Systems Neuroengineering Lab (Cauwenberghs Lab) and published in *npj Unconventional Computing* (2026). This picked up after the lab's earlier EIC/PIC chip work ([hardware-constrained-nas](https://github.com/krish-arora100/hardware-constrained-nas)) once the team moved to HiAER-Spike's larger, less hardware-constrained architecture.

This is a general summary of the key files, not a full record — most of the day-to-day iteration variants aren't included, just the versions that mattered.

## What this project is

HiAER-Spike runs trained networks on FPGA hardware for real-time, event-driven inference. The lab's toolchain quantizes a trained PyTorch model, converts it into the chip's connectivity format, and runs it on the FPGA or a software simulator to check hardware accuracy against the original model.

## Repo structure

- **`converters/`** — model-to-hardware converters. `converter_krish_avgpool.py` extends the base converter (which uses Keli's original quantization approach) with AvgPool2d support. `Converter_LeNet5_Stride2_2Channels_IFNeuron.py` is the converter built together with Christopher for the 2-channel stride LeNet5/IFNeuron pipeline. `modular_converters/` holds the final modular set — MLP, LeNet5, LeNet5 with maxpool, DVS Gesture — each swappable independently. `other_converters/` has earlier iterations of the same pipeline.
- **`demo/`** — `icrcdemo_krish_hardware_CIFAR10-DVS.py` runs the full train/quantize/convert/test pipeline for a DVS Gesture model and compares hardware vs. software accuracy; `utils_krish.py` has the training, testing, and plotting helpers it uses.
- **`models/`** — network architectures (`models_krish.py`) and custom IF/LIF neuron implementations (`Krish_custom_neurons.py`) used across the converters and demos.
- **`plots/`** — accuracy and membrane-potential plotting utilities.
- **`quantizer.py`** — the weight quantization module the converters use.
- **`CIFAR10_ANN/`** — bit-slicing based CIFAR-10 ANN-to-SNN conversion work.

## Tech stack

PyTorch, SpikingJelly, snnTorch, quantization-aware training, MNIST / DVS128 Gesture / CIFAR-10.
