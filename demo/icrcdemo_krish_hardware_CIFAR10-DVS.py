#to activate venv: eval $(poetry env activate)
#to avoid breakpoints: PYTHONBREAKPOINT=0 python icrcdemo_krish.py
#to run in background and log output(make sure to change paths): PYTHONBREAKPOINT=0 nohup python icrcdemo_krish.py -out-dir /home/k7arora/output/data_10T/1 > /home/k7arora/output/data_10T/1/log.txt 2>&1 &(but change path)
#to run default spikingjelly model with val_split: PYTHONBREAKPOINT=0 nohup python -u /home/k7arora/hs_api/examples/CRI_Mapping/icrcdemo_krish.py -b 16 -channels 128 -epochs 256 -out-dir /home/k7arora/output/data_10T/spikingjellytutorial/val_split > /home/k7arora/output/data_10T/spikingjellytutorial/val_split/log.txt 2>&1 &(but change path)
#PYTHONBREAKPOINT=0 nohup python -u icrcdemo_krish_hardware_09-02-25.py -b 64 -channels 4 -epochs 50 -out-dir /home/k7arora/hs_api/examples/CRI_Mapping/chris_code/converter_testing/IFNeuron/output > /home/k7arora/hs_api/examples/CRI_Mapping/chris_code/converter_testing/IFNeuron/output/log.txt 2>&1 &

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda import amp
import torchvision.transforms as transforms
import numpy as np
import random
from spikingjelly.datasets import pad_sequence_collate
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from examples.CRI_Mapping.chris_code.converter_testing.IFNeuron.utils_krish import train_DVS_Time_with_plot_autotrain, test_DVS_Time
from hs_api import CRI_network
#from hs_api.converter import CRI_Converter, Quantize_Network, BN_Folder #initially just hs_api.converter
from hs_api.quantizer import Quantize_Network #initially just hs_api.converter
import os
import functools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.ao.quantization as tq
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os
from hs_api.custom_neurons import Custom_LIFNode, Custom_IFNode

from spikingjelly.activation_based import neuron, functional, surrogate, layer
from copy import deepcopy

os.environ["PYTHONBREAKPOINT"] = "0"


#USER ARGUMENTS

# (channels, num_conv_blocks)
model_variants = (
    (45, 3),
)

input_size = (90, 90) # for resizing
kernel_size = 3
spiking_neuron = Custom_LIFNode #neuron.LIFNode #Custom_IFNode


parser = argparse.ArgumentParser()
parser.add_argument("-b", default=64, type=int, help="batch size")
parser.add_argument(
    "-data-dir",
    default="/home/k7arora/DVS128Gesture",
    type=str,
    help="path to dataset",
)
parser.add_argument(
    "-out-dir",
    default="/home/k7arora/hs_api/examples/CRI_Mapping/chris_code/converter_testing/10-26-25_CIFAR10-DVS/1",
    type=str,
    help="dir path that stores the trained model checkpoint",
)
parser.add_argument("-resume_path", default="", type=str, help="checkpoint file to resume training if desired(leave empty in this case)")
parser.add_argument("-epochs", default=250, type=int, help="max number of epochs to train for")
parser.add_argument("-lr", default=1e-5, type=float)
parser.add_argument(
    "-weight_decay", default=0.001, type=float, help="weight decay for Adam"
)
parser.add_argument(
    "-j",
    default=8,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 8)",
)
parser.add_argument(
    "-opt", default="adam", type=str, help="use which optimizer. SDG or Adam"
)
parser.add_argument("-patience", default=20, type=int, help="patience for early stopping")
parser.add_argument("-T_max", default=250, type=int, help="T_max for CosineAnnealingLR")
parser.add_argument("-targets", default=10, type=int, help="target label size (CIFAR10-DVS has 10)")


#END OF USER ARGUMENTS




args = parser.parse_args()
#output_dir = "/home/k7arora/output/data_10T/1"
output_dir = args.out_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

quantized_model_path = os.path.join(output_dir, "quantized_model.pth")


class CIFAR10DVSNetChrisModeled(nn.Module):
    def __init__(self, input_shape, channels=2, conv_blocks=2, spiking_neuron: callable = None, kernel_size=5, **kwargs):
        super(CIFAR10DVSNetChrisModeled, self).__init__()
        B, C, H, W = input_shape
        conv_layers = []
        for i in range(conv_blocks):
            if i == 0:
                conv_layers.append(layer.Conv2d(C, channels, kernel_size=kernel_size, stride=2, padding=0, bias=False))
            else:
                conv_layers.append(layer.Conv2d(channels, channels, kernel_size=kernel_size, stride=2, padding=0, bias=False))
            conv_layers.append(spiking_neuron(**deepcopy(kwargs)))

        conv_seq = nn.Sequential(*conv_layers)
        #print(conv_seq)
        B, C, H, W = input_shape
        dummy_input = torch.zeros((B, C, H, W))


        with torch.no_grad():
            x_out = conv_seq(dummy_input)
            in_features = x_out.flatten(start_dim=1).shape[1]

        print("Input features to first linear layer:", in_features)

        self.conv_fc = nn.Sequential(
            *conv_layers,
            layer.Flatten(),
            layer.Linear(in_features, 120, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(120, 84, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
            layer.Linear(84, args.targets, bias=False),
            spiking_neuron(**deepcopy(kwargs)),
        )


    def forward(self, x):
        return self.conv_fc(x)        
    


def main():
    scaler = amp.GradScaler()

    # resize transform that iterates over the temporal dimension, binarizes
    class DVSResizeAndBinarize:
        def __init__(self, size):
            self.size = size

        def __call__(self, data):
            frames, label = data if isinstance(data, tuple) else (data, None)
            if isinstance(frames, np.ndarray):
                frames = torch.from_numpy(frames)
            T, C, H, W = frames.shape

            resized = torch.zeros((T, C, self.size[0], self.size[1]), dtype=frames.dtype, device=frames.device)
            for t in range(T):
                frame = frames[t]  # [C, H, W]
                resized_frame = torch.nn.functional.interpolate(
                    frame.unsqueeze(0), size=self.size, mode='bilinear', align_corners=False
                ).squeeze(0)
                binarized_frame = (resized_frame > 0).float()
                resized[t] = binarized_frame
            return (resized, label) if label is not None else resized


    # Use resize transform for all datasets
    resize_transform = DVSResizeAndBinarize(size=input_size)  # resize from 128x128 to 90x90

    # collate wrapper: call existing pad_sequence_collate then ensure frames are resized
    # and binarized to the expected input_size (T, B, C, H, W -> resize each frame)
    def pad_collate_resize(batch, size):
        imgs, labels, others = pad_sequence_collate(batch)
        # imgs shape: (T, B, C, H, W)
        T, B, C, H0, W0 = imgs.shape
        imgs = imgs.view(T * B, C, H0, W0)
        imgs = F.interpolate(imgs, size=size, mode="bilinear", align_corners=False)
        imgs = (imgs > 0).float()
        imgs = imgs.view(T, B, C, size[0], size[1])
        return imgs, labels, others


    full_dataset = CIFAR10DVS(
        root=args.data_dir,
        frames_number=10,
        data_type="frame",
        split_by="number",
        transform=None,
    )

    # Determine sizes for train/val/test. 90% train, 5% val = 5% test.
    N = len(full_dataset)
    test_size = int(0.05 * N)
    val_size = int(0.05 * N)
    train_size = N - val_size - test_size

    # Reproducible random split
    torch.manual_seed(42)
    indices = torch.randperm(N)
    train_indices = indices[:train_size]
    val_indices = indices[train_size: train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create dataset instances that include transforms (resize + binarize) and then Subset them
    train_set_aug = CIFAR10DVS(
        root=args.data_dir,
        frames_number=10,
        data_type="frame",
        split_by="number",
        transform=resize_transform,
    )

    # Create subsets
    train_set = Subset(train_set_aug, train_indices)
    val_set = Subset(full_dataset, val_indices)  # No augmentation
    test_set = Subset(full_dataset, test_indices)  # No augmentation

    print(f"Training samples: {len(train_set)} ({len(train_set)/len(full_dataset)*100:.1f}%)")
    print(f"Validation samples: {len(val_set)} ({len(val_set)/len(full_dataset)*100:.1f}%)")
    print(f"Test samples: {len(test_set)} ({len(test_set)/len(full_dataset)*100:.1f}%)")

    # Create DataLoaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        collate_fn=functools.partial(pad_collate_resize, size=input_size),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=functools.partial(pad_collate_resize, size=input_size),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        collate_fn=functools.partial(pad_collate_resize, size=input_size),
    )
    
    T, C, H, W = train_set[0][0].shape
    print(f"Input shape: {(T, C, H, W)}")
    print(f"Number of training samples: {len(train_set)}")
    print(f"Number of validation samples: {len(val_set)}")
    print(f"Number of testing samples: {len(test_set)}")

    #normally remove
    print("using val set for all test accuracies")
    test_loader = val_loader


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(device)
    #get list of channels and conv_blocks from model_variants
    for channels, conv_blocks in model_variants:
        print(f"\n\nTraining model with {channels} channels and {conv_blocks} conv blocks")
        train((T, C, H, W), channels, conv_blocks, spiking_neuron, train_loader, val_loader, test_loader, device, scaler, kernel_size)




def train(input_shape, channels, conv_blocks, spiking_neuron, train_loader, val_loader, test_loader, device, scaler, kernel_size):
    T, C, H, W = input_shape
    #create subfolder inside args.out-dir to save this model, named after channels and conv_blocks
    model_dir = os.path.join(output_dir, f"ChrisModeled_C_{channels}_CB_{conv_blocks}")
    os.makedirs(model_dir, exist_ok=True)

    net = CIFAR10DVSNetChrisModeled(
        input_shape=(args.b,C,H,W),
        channels=channels,
        conv_blocks=conv_blocks,
        spiking_neuron=spiking_neuron,
        kernel_size=kernel_size,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        decay_input=False,
        tau=float(2**63),
    )
    print(net)

    net.to(device)

    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    
    print("Start Training")
    net.train()

    # train
    val_acc, min_val_loss, best_epoch = train_DVS_Time_with_plot_autotrain(args, net, train_loader, val_loader, device, scaler, channels, output_dir=model_dir, save_every=25)
    print("Training Finished")
    print("Best epoch: ", best_epoch+1)
    print("Minimum validation loss at best epoch: ", min_val_loss)
    print("Validation accuracy at best epoch: ", val_acc)

    # load the best model after training
    checkpoint_path = os.path.join(model_dir, f"checkpoint_max_T_{T}_C_{channels}_lr_{args.lr}.pth")
    #checkpoint_path = "/home/k7arora/output/data_10T/spikingjellytutorial/val_split/checkpoint_max_T_10_C_128_lr_0.001.pth"
    #checkpoint_path = "/home/k7arora/output/data_10T/spikingjellytutorial/val_split/avg_pool/checkpoint_max_T_10_C_128_lr_0.001.pth"
    print("checkpoint_path: ", checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location=torch.device(device),
    )
    net.load_state_dict(checkpoint["net"])
    
    net.eval()


    # Test original model accuracy
    print("\n" + "="*50)
    print("TESTING ORIGINAL MODEL ACCURACY")
    print("="*50)
    original_accuracy, original_loss = test_DVS_Time(args.targets, net, test_loader, device)
    print(f"Spikingjelly model accuracy: {original_accuracy}%")
    
if __name__ == "__main__":
    main()
