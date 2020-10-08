import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from PIL import Image
from matplotlib.ticker import FormatStrFormatter

from train_utils import load_data
from train_functions import build_classifier, validation, train_model, test_model, save_model

parser = argparse.ArgumentParser(description='Train Model')

parser.add_argument('data_directory', action = 'store',
                    default = '../aipnd-project/flowers',
                    help = 'Enter path to training data.')

parser.add_argument('--arch', action='store',
                    dest = 'pretrained_model', default = 'vgg19',
                    help= 'Enter pretrained model to use. The default is VGG-19.')

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Enter location to save checkpoint in.')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=int, default = 0.0007,
                    help = 'Enter learning rate for training the model, default is 0.0007.')

parser.add_argument('--dropout', action = 'store',
                    dest='drpt', type=int, default = 0.07,
                    help = 'Enter dropout for training the model, default is 0.07.')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'units', type=int, default = 600,
                    help = 'Enter number of hidden units in classifier, default is 600.')

parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 2,
                    help = 'Enter number of epochs to use during training, default is 2')

parser.add_argument('--gpu', action = "store_true", default = False,
                    help = 'Turn GPU mode on or off, default is off.')

results = parser.parse_args()

data_dir = results.data_directory
save_dir = results.save_directory
learning_rate = results.lr
dropout = results.drpt
hidden_units = results.units
epochs = results.num_epochs
gpu_mode = results.gpu

# Load and preprocess data 
trainloader, testloader, validloader, train_data, test_data, valid_data = load_data(data_dir)

# Load pretrained model
pre_tr_model = results.pretrained_model
print(pre_tr_model)
model = getattr(models,pre_tr_model)(pretrained=True)

# Build and attach new classifier
input_units = model.classifier[0].in_features
model = build_classifier(model, input_units, hidden_units, dropout)

# Recommended to use NLLLoss when using Softmax
criterion = nn.NLLLoss()
# Using Adam optimiser which makes use of momentum to avoid local minima
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# Train model
model, optimizer = train_model(model, epochs, trainloader, validloader, criterion, optimizer, gpu_mode)
# Test model
test_model(model, testloader, gpu_mode)
# Save model
model.class_to_idx = train_data.class_to_idx
save_model(model, train_data, optimizer, save_dir, epochs)
