import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.ao.quantization as tq
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import CIFAR10_bitslicing

'''
3x90x90 input resolution

ReLU activation function
Best Model determined based on lowest validation loss
Epoch patience resets to 0 if loss improves

Convolutional and Linear layers have bias=False
BatchNorm2d removed
'''
    
# Define relevant variables for the ML task
model_name = "CIFAR10_ANN_bitslicing_AutoAugment_Cutout_NOQUANT_Nov24_2"
batch_size = 64
num_classes = 10  #number of output classes (10 for CIFAR)
learning_rate = 0.001
num_epochs = 10000
early_stop_threshold = 50
    
# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Mean and STD used for normalization
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2616)

#defining the quantized convolutional layer
class QuantConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_bits):
      super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias=False)  #initializes the parent object, Conv2d
      self.num_bits = num_bits
  
    def forward(self, x):
      q_w = QuantizedWeightSTE.apply(self.weight, self.num_bits) #q_w is the quantized weights of the layer
      return F.conv2d(x, q_w, self.bias, self.stride,
                      self.padding, self.dilation, self.groups) #applies a 2d conv over the input with quantized weights
   
#defining the quantized linear layer
class QuantLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_bits): 
      super().__init__(in_features, out_features, bias=False) #initializes the parent object, nn.Linear
      self.num_bits = num_bits
  
    def forward(self, x):
      q_w = QuantizedWeightSTE.apply(self.weight, self.num_bits) #q_w is the quantized weights of the layer
      return F.linear(x, q_w, self.bias) #applies a linear transform on the input with quantized weights

class CustomCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # conv1-3: 21 channels, stride=1, padding=1
            QuantConv(3, 21, kernel_size=3, stride=1, padding=1, num_bits=16), nn.ReLU(),
            QuantConv(21, 21, kernel_size=3, stride=1, padding=1, num_bits=16), nn.ReLU(),
            QuantConv(21, 21, kernel_size=3, stride=1, padding=1, num_bits=16), nn.ReLU(),

            # conv4: 42 channels, stride=2, padding=1
            QuantConv(21, 42, kernel_size=3, stride=2, padding=1, num_bits=16), nn.ReLU(),

            # conv5-6: 42 channels, stride=1, padding=1
            QuantConv(42, 42, kernel_size=3, stride=1, padding=1, num_bits=16), nn.ReLU(),
            QuantConv(42, 42, kernel_size=3, stride=1, padding=1, num_bits=16), nn.ReLU(),

            # conv7: 84 channels, stride=2, padding=1
            QuantConv(42, 84, kernel_size=3, stride=2, padding=1, num_bits=16), nn.ReLU(),

            # conv8-9: 84 channels, stride=1, padding=1
            QuantConv(84, 84, kernel_size=3, stride=1, padding=1, num_bits=16), nn.ReLU(),
            QuantConv(84, 84, kernel_size=3, stride=1, padding=1, num_bits=16), nn.ReLU(),
        )

        self.classifer = nn.Sequential(
            nn.Flatten(),
            QuantLinear(84 * 8 * 8, 512, num_bits=16),
            nn.ReLU(inplace=True),
            QuantLinear(512, num_classes, num_bits=16),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifer(x)
        return x
        
class CustomCNN_NOQUANT(nn.Module):
    def __init__(self, num_classes: int, input_shape=(64, 15, 32, 32)):
        super().__init__()

        self.features = nn.Sequential(
            # conv1-3: 21 channels, stride=1, padding=1
            nn.Conv2d(15, 21, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(21, 21, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(21, 21, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(),

            # conv4: 42 channels, stride=2, padding=1
            nn.Conv2d(21, 42, kernel_size=3, stride=2, padding=1, bias=False), nn.ReLU(),

            # conv5-6: 42 channels, stride=1, padding=1
            nn.Conv2d(42, 42, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(42, 42, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(),

            # conv7: 84 channels, stride=2, padding=1
            nn.Conv2d(42, 84, kernel_size=3, stride=2, padding=1, bias=False), nn.ReLU(),

            # conv8-9: 84 channels, stride=1, padding=1
            nn.Conv2d(84, 84, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(84, 84, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(),
        )

        dummy_input = torch.zeros(input_shape)
        with torch.no_grad():
            x_out = self.features(dummy_input)
            in_features = x_out.flatten(start_dim=1).shape[1]

        print("Input features to first linear layer:", in_features)

        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifer(x)
        return x
    
    
class CustomCNN_NOQUANT_NOV24_1(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(15, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 50, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 100, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(100, 200, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # compute flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 15, 32, 32)
            out = self.features(dummy)
            in_features = out.flatten(start_dim=1).shape[1]

        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifer(x)
        return x

class CustomCNN_NOQUANT_NOV24_2(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(15, 96, kernel_size=3, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=0, bias=False),
            nn.ReLU(inplace=True),
        )

        # compute flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, 15, 32, 32)
            out = self.features(dummy)
            in_features = out.flatten(start_dim=1).shape[1]

        self.classifer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifer(x)
        return x


#applies quantization with a straight-through estimator in the backward pass to ensure gradients flow on the standard input for weights (symmetric)
class QuantizedWeightSTE(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, num_bits):
    #quantize to number of bits, keeping scale the same
    max = torch.max(torch.abs(x)).to(device) #max abs val
    max = torch.where(max == 0, torch.tensor(1.0).to(device), max) #avoid dividing by 0 if max is 0

    levels = 2**(num_bits-1)-1 #number of quantization levels
    x_q = torch.round(x / max * levels) #quantize to range [-(2**(num_bits-1)-1),  2**(num_bits-1)-1], integers, max becomes levels

    x_quantized_scaled = x_q * max / levels #scale back to the range the input was in (all positive floating point, max is preserved)

    ctx.save_for_backward(x) #save input for backward

    return x_quantized_scaled

  @staticmethod
  def backward(ctx, grad_output):
    #backward pass: Use STE (pass gradients as if quantization didn't exist)
    grad_input = grad_output.clone()  #pass gradient straight through
    return grad_input, None


def main():
  # deterministic per-channel binarization: ToTensor -> threshold
  class Binarize(object):
    def __init__(self, threshold=0.5):
      self.th = threshold
    def __call__(self, x):
      # x is a tensor in [C,H,W] with values in [0,1]
      return (x > self.th).float()

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

      
  model = CustomCNN_NOQUANT_NOV24_2(num_classes).to(device)
  #model = CustomCNN_NOQUANT_NOV24_1(num_classes, input_shape=(batch_size,C,H,W)).to(device)

  print(model)
  
  #Setting the loss function
  cost = nn.CrossEntropyLoss()
      
  #Setting the optimizer with the model parameters and learning rate
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0001)
      
  #this is defined to print how many steps are remaining when training
  total_step = len(train_loader)

  #lists to store loss and accuracy values
  loss_history = []
  val_losses = []
  val_accuracies = []

  #Keeping track of best val loss and accuracy for early stop
  best_val_accuracy = 0.0
  epoch_best_val_accuracy = 0
  best_val_loss = 100.0 #absurd loss
  epochs_without_improvement = 0

  #Model Training and validation
  for epoch in range(num_epochs):
      running_loss = 0.0

      for i, (images, labels) in enumerate(train_loader):  #iterates over mini batches produced by DataLoader
          images = images.to(device)  #moves image tensor with shape [batch, channels, height, width] to computation device (CPU, GPU)
          labels = labels.to(device)  #moves label tensor with shape [batch] to computational device 
              
          #Forward pass
          outputs = model(images)  #produce logits of shape [batch, num_classes]
          loss = cost(outputs, labels) #computes cross entropy loss
          #Backward and optimize
          optimizer.zero_grad()  #clear existing gradients because Pytorch accumulates gradients by default
          loss.backward()        #fills param.grad with gradient of loss with respect to that parameter
          optimizer.step()       #updates parameter using Adam Optimizer
          running_loss += loss.item()
          if (i+1) % 400 == 0:
              print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

      print("lr: " + str(scheduler.get_last_lr()))
      scheduler.step()

      epoch_loss = running_loss / len(train_loader)
      loss_history.append(epoch_loss)

      #validation phase
      model.eval()  # Set the model to evaluation mode
      val_loss = 0.0
      val_correct = 0
      val_total = 0

      with torch.no_grad():
        for inputs, labels in val_loader:
          inputs, labels = inputs.to(device), labels.to(device)

          #forward pass
          outputs = model(inputs)
          loss = cost(outputs, labels)

          #statistics
          val_loss += loss.item()
          _, predicted = torch.max(outputs, 1)
          val_correct += (predicted == labels).sum().item()
          val_total += labels.size(0)

      val_loss /= len(val_loader)
      val_accuracy = 100 * val_correct / val_total

      val_losses.append(val_loss)
      val_accuracies.append(val_accuracy)

      #print val statistics after each epoch
      print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

      #check for early stopping based on validation loss
      #save model with best val_loss
      if val_losses[-1] < best_val_loss:   #index -1 returns element at end of val_losses
        print(f"Best Val Loss: {best_val_loss:.4f}, New Best Val Loss: {val_losses[-1]:.4f}")
        torch.save(model.state_dict(), './' + model_name)  #save the model
        print(f"Model updated at epoch {epoch + 1}")
        best_val_loss = val_losses[-1]  #update best loss
        epochs_without_improvement = 0  #reset epochs without improvement to 0
      else:
        epochs_without_improvement += 1

      # If the validation loss starts increasing or is plateauing, stop training
      if epochs_without_improvement >= early_stop_threshold:
        print(f"Early stopping at epoch {epoch + 1}")
        break

      #update best val_accuracy
      if val_accuracy >= best_val_accuracy:
        best_val_accuracy = val_accuracy
        epoch_best_val_accuracy = epoch + 1


  #print best training and val loss. print best val accuracy
  print(f"Best Train Loss {min(loss_history):.4f}")
  print(f"Best Val Loss {min(val_losses):.4f}")
  print(f"Best Val accuracy {best_val_accuracy:.2f}% at epoch {epoch_best_val_accuracy}")

  #create loss graph
  plt.figure(figsize=(6,4))
  plt.plot(range(1, epoch + 2), loss_history, marker="o", label="Training")
  plt.plot(range(1, epoch + 2), val_losses, marker="o", linestyle="--", label="Validation")
  plt.title("Training and Validation Loss per epoch")
  plt.xlabel("Epoch")
  plt.ylabel("Cross-entropy loss")
  plt.grid(True)
  plt.legend()
  plt.tight_layout()
  plt.savefig("loss_graph_" + model_name + ".png", dpi=300)  # Save as an image
  
  #test phase
  PATH = "./" + model_name
  model.load_state_dict(torch.load(PATH)) #load weights/parametes of best trained model into model 
  model.eval()  # Set the model to evaluation mode

  #test model on test set 
  with torch.no_grad():
      correct = 0
      total = 0

      for images, labels in test_loader:
          images = images.to(device)
          labels = labels.to(device)
          
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

      accuracy = 100 * correct / total
      print(f'Accuracy of the network on the 10000 test images: {accuracy:.2f} %')

if __name__ == "__main__":
    main()