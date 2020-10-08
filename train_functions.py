
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image


# Function to build new classifier
def build_classifier(model, input_units, hidden_units, dropout):
    # Weights of pretrained model are frozen so we don't backprop through/update them
    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    # Replacing the pretrained classifier with the one above
    model.classifier = classifier
    return model



def validation(model, validloader, criterion, gpu_mode):
    valid_loss = 0
    accuracy = 0
    
    if gpu_mode == True:
    # change model to work with cuda
        model.to('cuda')
    else:
        pass
    # Iterate over data from validloader
    for ii, (images, labels) in enumerate(validloader):
        
        if gpu_mode == True:
        # Change images and labels to work with cuda
            images, labels = images.to('cuda'), labels.to('cuda')
        else:
            pass
        # Forward pass image though model for prediction
        output = model.forward(images)
        # Calculate loss
        valid_loss += criterion(output, labels).item()
        # Calculate probability
        ps = torch.exp(output)
        # Calculate accuracy
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy



def train_model(model, epochs, trainloader, validloader, criterion, optimizer, gpu_mode):
    print_every = 40 # Prints every 40 images out of batch of 50 images
    steps = 0
    if gpu_mode == True:
        model.to('cuda')
    else:
        pass
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}... "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

        return model, optimizer



def test_model(model, testloader, gpu_mode):
    correct = 0
    total = 0
    
    if gpu_mode == True:
        model.to('cuda')
    else:
        pass

    with torch.no_grad():
        for ii, (images, labels) in enumerate(testloader):
            
            if gpu_mode == True:
                images, labels = images.to('cuda'), labels.to('cuda')
            else:
                pass
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"Test accuracy of model for {total} images: {round(100 * correct / total,3)}%")


    
def save_model(model, train_data, optimizer, save_dir, epochs):
    # Saving: feature weights, new classifier, index-to-class mapping, optimiser state, and No. of epochs
    checkpoint = {'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'opt_state': optimizer.state_dict,
                  'num_epochs': epochs}

    return torch.save(checkpoint, save_dir)


   
def predict(filepath, image_path, model, topk=5):
    
    # Loading model
    load_model = load_checkpoint(filepath, model).cpu()
    # Pre-processing image
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    image_torch = image_tensor.unsqueeze_(0)

    # Setting model to evaluation mode and turning off gradients
    load_model.eval()
    with torch.no_grad():
        # Running image through network
        output = load_model.forward(image_torch)

    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)
