from hs_api.api import CRI_network
import hs_bridge
import CIFAR10_bitslicing
import torch
import torchvision
import torchvision.transforms as transforms
from hs_api.neuron_models import LIF_neuron
import torch.nn.functional as F
import numpy as np
from spikingjelly.activation_based import functional
import os



'''
Adapted from /LeNet5/LeNet5_Converter.py
Implements clock cycle and hbmaccesses recording
Implements converter for Padding=1 conv layers
All axons, neurons, and feature map indices are indexed from 1
'''

#CHANGE HERE FOR DIFFERENT MODELS
kernel_size = 3     #kernel size of convolutional layers
stride1 = 1           #stride of convolutional layers 1-2
stride2 = 2           #stride of convolutional layers 3-4
input_res = 32     #resolution of input
conv1_output_res = 32      #resolution of output feature maps from conv1
conv2_output_res = 32      #resolution of output feature maps from conv2
conv3_output_res = 16
conv4_output_res = 8
num_layers = 6 #(4 conv, 2 fc)
num_outputs = 10 #CIFAR has 10 classes
extra_timesteps = 11  #number of extra timesteps to run after last input frame to allow spikes to propogate through network

#parameters
batch_size = 64

#weights path
PATH = "/home/k7arora/hs_api/examples/CRI_Mapping/chris_code/converter_testing/11-24-25_CIFAR10_ANNToSNN/final_LIF_SNN_weights_QAT_nobias"
T=30                     #number of timesteps to run SNN

# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# deterministic per-channel binarization: ToTensor -> threshold
class Binarize(object):
    def __init__(self, threshold=0.5):
        self.th = threshold
    def __call__(self, x):
        # x is a tensor in [C,H,W] with values in [0,1]
        return (x > self.th).float()
    
#convert conv layer to dictionary entries
def convert_conv_layer(connections: dict, weights: dict, scales: dict, weight_key, output_res: int, layer_idx: int, kernel_size: int, stride: int, padding: int): 
        #creating C1Map to identify which C1 neurons connect to which pixel/neuron of the feature map in conv2
        cMap = torch.arange(output_res ** 2, dtype=torch.float32).reshape(1, 1, output_res, output_res)
        cMap = cMap + 1          
        patchTensor = F.unfold(input=cMap, kernel_size=kernel_size, stride=stride, padding=padding)   # dilation=1 by default

        #patch_rows is a tensor where #rows = resolution of feature maps.
        #Each row contains the indices of the axons corresponding to each pixel in the feature map
        patch_rows = patchTensor.transpose(1, 2).squeeze(0)
        patch_rows = patch_rows.to(torch.int16)   #convert patch_rows from FP32 tensor to INT16 tensor

        #connecting C1 neurons in conv1 to C2 neurons in conv2
        #outer loop: iterate over output channels in conv2
        for output_idx, output_channel in enumerate(weights[weight_key], start=1): 

            #inner loop: iterate over input-channel kernels with index for this output channel
            for feature_map, kernel in enumerate(output_channel, start=1):  
                flat_kernel = kernel.flatten() #flatten kernel is 1D tensor of the weights for C1 -> C2
                
                #inner loop 2: iterate through each patch row. #rows = resolution of output feature map 
                for j, row in enumerate(patch_rows, start=1):
                    neuronName = f"C{layer_idx+1}.{output_idx}.{j}"  #each row corresponds to one pixel in feature map. Create neuron entry C2.{feature map#}.{index}
                    neuronType = LIF_neuron(threshold=int(scales[weight_key]), shift=0, leak=63)
                    connections[neuronName] = ([], neuronType)
            
                    #inner loop 3: iterate through each elem in row. Each elem is index of C1 -> C2 
                    for i, elem in enumerate(row):
                        index = int(elem.item())
                        if index != 0:
                            key = f"C{layer_idx}.{feature_map}.{index}"
                            weight = flat_kernel[i].item()
                            connections[key][0].append((neuronName, weight))

#convert fp32 weights in model into int16
def fp32_to_int16_state_dict(weights: dict):
    """Return two dicts:
       1. int16 weights   
       2. per‑tensor scale factors (float32)

       Biases are scaled using their corresponding weight scale
    """
    int16_sd, scales = {}, {}

    #First pass: compute and store scales for all weight tensors
    for name, tensor in weights.items():
        if name.endswith(".weight"):
            max_val = tensor.abs().max()
            if max_val == 0:
                max_val = 1 #avoid divide-by-zero
            scale   = (2**15 - 1) / max_val
            int16_sd[name] = torch.round(tensor * scale).to(torch.int16)
            scales[name]   = scale.item()

    #Second pass: convert biases using the scale from corresponding weights
    for name, tensor in weights.items():
        if name.endswith(".bias"):
            weight_name = name.replace(".bias", ".weight")
            if weight_name not in scales:
                raise ValueError(f"Missing corresponding weight tensor for bias: {name}")
            scale = scales[weight_name]
            int16_sd[name] = torch.round(tensor * scale).to(torch.int16)
            scales[name] = scale  # Store bias scale under its own name
    return int16_sd, scales


#reads specific MPs from a list of specified neurons
def membrane_potential_reader(results: tuple, neurons: list):
    membrane_potentials = []
    membrane_potentials_dict = dict(results[0]) # convert to dict for easy look up
    for label in neurons:       #iterate through all output neurons
        membrane_potentials.append(membrane_potentials_dict[label])
    
    return membrane_potentials  #return output neuron with greatest membrane potential

#determines the max membrane potential from the output neurons
def max_membrane_potential(object: tuple, outputs: list):
    output_membrane_potentials = []
    max = float('-inf') # start lower than any real value
    max_label = None
    membrane_potentials_dict = dict(object[0]) # convert to dict for easy look up
    for label in outputs:       #iterate through all output neurons
        output_membrane_potentials.append(membrane_potentials_dict[label])
        if membrane_potentials_dict[label] > max:
            max = membrane_potentials_dict[label]
            max_label = label
    
    return max_label, output_membrane_potentials  #return output neuron with greatest membrane potential

#determines the max membrane potential from the output neurons
def max_membrane_potential2(outputs: list):
    max = float('-inf') # start lower than any real value
    max_label = None
    membrane_potentials_dict = dict(outputs) # convert to dict for easy look up
    for key in membrane_potentials_dict:       #iterate through all output neurons
        if membrane_potentials_dict[key] > max:
            max = membrane_potentials_dict[key]
            max_label = key
    
    return max_label  #return output neuron with greatest membrane potential

def main():
    #load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  
        transforms.PILToTensor(),
        CIFAR10_bitslicing.cifar10_to_15channel_binary,  
        Binarize(threshold=0.5),
    ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Pads the image by 4 pixels and then takes a random 32x32 crop.
        transforms.RandomHorizontalFlip(),    # Flips the image horizontally with a default probability of 0.5.
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),            
        CIFAR10_bitslicing.Cutout(n_holes=1, length=16),
        transforms.ConvertImageDtype(torch.uint8),
        CIFAR10_bitslicing.cifar10_to_15channel_binary,
        Binarize(threshold=0.5)
    ])

    # Loading the dataset and preprocessing
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                    train=True,
                                                    transform=transform_train,
                                                    download=True)

    full_val_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                    train=True,
                                                    transform=transform,
                                                    download=True)

    #split full_train_dataset into training set and validation set
    train_size = int(0.8 * len(full_train_dataset)) #50k for training, 10k for validation
        
    # Get random permutation of indices and split
    perm = torch.randperm(len(full_train_dataset))
    train_idx = perm[:train_size].tolist()
    val_idx   = perm[train_size:].tolist()

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_idx)
    val_dataset   = torch.utils.data.Subset(full_val_dataset, val_idx)
        
    test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                                train = False,
                                                transform = transform,
                                                download=True)
        
        
    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)
        
    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                                batch_size = batch_size,
                                                shuffle = True)

    C, H, W = train_dataset[0][0].shape
    print(f"Input shape: {(C, H, W)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    
    #load model architecture weights
    #model = CIFAR10_SNN_krishquant_QAT.make_SNN_model().to(device)
    weights = torch.load(PATH, map_location=device)  
    '''
    #load full precision weights into model
    model.load_state_dict(weights)

    print("using test set for testing")

    print("\n" + "="*50)
    print("TESTING ORIGINAL MODEL ACCURACY")
    print("="*50)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)  #moves image tensor with shape [batch, channels, height, width] to computation device (CPU, GPU)
            labels = labels.to(device)  #moves label tensor with shape [batch] to computational device 

            # Reset SNN state
            functional.reset_net(model) 
            out_spike_sum = 0

            # Forward pass through SNN with time steps
            for t in range(T):
                out_spike_sum += model(images)

            #add 11 extra timesteps after last input frame to allow it to propogate through network
            #need 0s for inputs in same size as input frame
            zeros = torch.zeros_like(images)
            for i in range(extra_timesteps):
                out_spike_sum += model(zeros)

            out_spike_rate = out_spike_sum / T
            prediction = out_spike_rate.argmax(dim=1)    #neuron with highest firing rate is prediction 
            correct += (prediction == labels).sum().item()
            total += labels.numel()

    SNN_accuracy = 100 * correct / total
    print(f"NoQuant SNN accuracy (T={T}): {SNN_accuracy:.2f} %")
    '''
    # Conversion to run on HiAER Spike
    print("\n" + "="*50)
    print("QUANTIZING MODEL")
    print("="*50)

    int16_sd, scales = fp32_to_int16_state_dict(weights=weights) #int16_sd is now the state dict of the quantized model
    print("Model weights quantized to int16")

    print(int16_sd.keys())

    #defining dictionaries and input/output lists
    axons = {}
    connections = {}
    outputs = []

    #create axon for every pixel in image
    for i in range(1, (3 *input_res * input_res) + 1):
        key = f"A{i}"
        axons[key] = []

    #add extra axon to avoid multiple of 256 axons issue
    key = f"Adummy"
    axons[key] = []

    #creating axonMap to identify which axons connect to which pixel/neuron of the feature map in conv1
    axonMap = torch.arange(3 * (input_res ** 2), dtype=torch.float32).reshape(1, 3, input_res, input_res)
    axonMap = axonMap + 1 #each entry labeled 1 to input_res**2 to ignore zeros from padding          
    patchTensor = F.unfold(input=axonMap, kernel_size=kernel_size, stride=stride1, padding=1)   #conv1, padding=0, dilation=1 by default

    #patch_rows is a tensor where #rows = resolution of feature maps.
    #Each row contains the indices of the axons corresponding to each pixel in the feature map
    patch_rows = patchTensor.transpose(1, 2).squeeze(0) 
    patch_rows = patch_rows.to(torch.int16)   #convert patch_rows from FP32 tensor to INT16 tensor

    # iterate through every weight kernel in first convolutional layer and map axons → (neuron, weight)
    for feature_map, kernel in enumerate(int16_sd["0.weight"], start=1):
        flat_kernel = kernel.flatten()

        for index , row in enumerate(patch_rows, start=1):
            neuronName = f"C1.{feature_map}.{index}"  #each row corresponds to one pixel in feature map. Create neuron entry C1.{feature map#}.{index}
            neuronType = LIF_neuron(threshold=int(scales["0.weight"]), shift=0, leak=63)
            connections[neuronName] = ([], neuronType)
        
            for i, elem in enumerate(row):   #each elem in row is the axon index
                axon_id = int(elem.item())
                if axon_id != 0:     #avoid all zeros from padding 
                    key = f"A{axon_id}"
                    weight = flat_kernel[i].item()
                    axons[key].append((neuronName, weight))

    
    #converting subsequent conv layers
    convert_conv_layer(connections, int16_sd, scales, "3.weight", conv1_output_res, 1, kernel_size, stride1, padding=1)  #conv2
    convert_conv_layer(connections, int16_sd, scales, "6.weight", conv2_output_res, 2, kernel_size, stride2, padding=1)  #conv3
    convert_conv_layer(connections, int16_sd, scales, "9.weight", conv3_output_res, 3, kernel_size, stride2, padding=1)  #conv4

    #connecting conv4 to fc1
    feature_map = 1
    for col in range(int16_sd["13.weight"].shape[1]):  #x.shape[1] == number of col
        if col % (conv4_output_res ** 2) == 0 and col != 0:  #determines the feature_map of the C4 neuron for C4 --> FC1
            feature_map += 1
        for i, elem in enumerate(int16_sd["13.weight"][:, col], start=1):     #iterate over element in a col
            connectingNeuron = (f"FC1.{i}", elem.item())
            connections[f"C4.{feature_map}.{(col % (conv4_output_res ** 2)) + 1}"][0].append(connectingNeuron)
    #connecting fc1 to fc2 (output layer)
    for col in range(int16_sd["16.weight"].shape[1]):  #x.shape[1] == number of col
        allConnections = []
        for i, elem in enumerate(int16_sd["16.weight"][:, col]):     #iterate over element in a col
            connectingNeuron = (i, elem.item())
            allConnections.append(connectingNeuron)
        neuronType = LIF_neuron(threshold=int(scales["13.weight"]), shift=0, leak=63)
        connections[f"FC1.{col+1}"] = (allConnections, neuronType)

    #creating output neurons
    outputs = []
    for x in range(10):     #output neurons are named 0-9
        neuronType = LIF_neuron(threshold=int(scales["16.weight"]), shift=0, leak=63)
        connections[x] = ([], neuronType)
        outputs.append(x)

    #counting synapses of network
    number_synapses = 0
    for key in connections:
        number_synapses += len(connections[key][0])

    for key in axons:
        number_synapses += len(axons[key])

    print(f"Number of LIF neurons: {len(connections)}")
    print(f"Number of axons: {len(axons)}")
    print(f"Number of synapses: {number_synapses}")
    
    #create network
    network = CRI_network(axons=axons,connections=connections,outputs=outputs,target="CRI")
    print("Network Loaded onto HiAER Spike")
    #used to save clock cycles and hbm accesses
    data = []

    #run testing
    correct = 0
    total = 0
    images = 0
    for img, labels in test_dataset:

        #reset membrane potnetials before each image
        hs_bridge.FPGA_Execution.fpga_controller.clear(len(connections), False, 0) 

        img = img.to(device) #shape [C, H, W]
        img = img.unsqueeze(0)  #add batch dimension -> shape [1, C, H, W]
        img = img.flatten(start_dim=1)  #flatten to shape [1, 3*32*32]
        spike_counts = torch.zeros(len(outputs))  #to count spikes over all frames
        for t in range(T):
            inputs = [] #list of input spikes for current frame

            for i, elem in enumerate(img[0, :]):
                if elem.item() > 0:  
                    inputs.append(f"A{i}")

            # Forward pass through SNN with time steps
            hardwareSpikes, _, _ = network.step(inputs)
            #print(f"Output spikes: {hardwareSpikes}")

            for spike in hardwareSpikes:
                if spike in outputs:
                    spike_counts[spike] += 1
                else:
                    print(f"Error: invalid output spike {spike}")

        #add 11 extra timesteps after last input frame to allow it to propogate through network
        for i in range(extra_timesteps):
            hardwareSpikes, clock_cycles, hbm_accesses = network.step([])
            #print(f"Output spikes: {hardwareSpikes}")

            for spike in hardwareSpikes:
                if spike in outputs:
                    spike_counts[spike] += 1
                else:
                    print(f"Error: invalid output spike {spike}")

        #record clock cycles and hbm accesses
        data.append((clock_cycles, hbm_accesses))

        spike_counts = spike_counts / T  #average spike counts(spike rate)
        print(f"Spike counts: {spike_counts}")

        predicted = torch.argmax(spike_counts).item()
        print(f"Predicted: {predicted}, Label: {labels}")

        total += 1
        if predicted == labels:
            correct += 1
        
        running_accuracy = 100 * correct / total
        print(f"Running accuracy : {running_accuracy:.2f} %")

        images += 1
        if images == 100:   #limit to 100 images for testing
            break

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

    #record (clockcyles, hbmaccess) as ordered pairs in numpy arr
    arr = np.asarray(data)              

    #Save converter FPGA accuracy to txt file and clock cycles to npy file
    parent_directory = os.path.dirname(PATH)

    #if accuracies.txt file already exists, just append converted accuracy to it, otherwise, create new file
    if os.path.exists(os.path.join(parent_directory, "accuracies.txt")):
        mode = "a"
    else:
        mode = "w"

    with open(os.path.join(parent_directory, "accuracies.txt"), mode) as f:
        f.write(f"FPGA Converted Accuracy: {accuracy:.2f}%\n")

    np.save(os.path.join(parent_directory, "clock_cycles_SNN_CIFAR10.npy"), arr)

if __name__ == "__main__":
    main()
