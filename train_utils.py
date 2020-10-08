
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image

from torchvision import datasets, transforms, models

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    mean_norms = [0.485, 0.456, 0.406]
    std_devs = [0.229, 0.224, 0.225]
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean_norms, std_devs)
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_norms, std_devs)
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean_norms, std_devs)
    ]),
    }   
    image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
    }

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=32),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=True)
    }
    # trainloader, validloader and testloader for training, validation and testing
    trainloader = dataloaders['train']
    testloader = dataloaders['test']
    validloader = dataloaders['valid']
    
    return trainloader, testloader, validloader, image_datasets['train'], image_datasets['test'], image_datasets['valid']
