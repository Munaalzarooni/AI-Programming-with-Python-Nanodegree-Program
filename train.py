import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse
from collections import OrderedDict
import json
import PIL
from torchvision import transforms
import seaborn as sns
from PIL import Image
import time
import os

      
        
def process_data():
    folder = 'flowers'

    transformations = { 
        'train': transforms.Compose([
            transforms.RandomResizedCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
    }

    #datasets
    im_datasets = {
                'train' : datasets.ImageFolder(folder + '/train', transform=transformations['train']),
                'test' : datasets.ImageFolder(folder + '/test', transform=transformations['test']),
                'val' : datasets.ImageFolder(folder + '/valid', transform=transformations['valid'])
                }
    c_t_idx= im_datasets["train"].class_to_idx;

    # define data loaders
    dataloaders = { 
                    'train' : torch.utils.data.DataLoader(im_datasets['train'], batch_size=64, shuffle=True),
                    'test' : torch.utils.data.DataLoader(im_datasets['test'], batch_size=64),
                    'val' : torch.utils.data.DataLoader(im_datasets['val'], batch_size=64)
                  }

    # print data set examples
    print("No. of training examples: ",len(im_datasets['train']))
    print("No. of test examples: ",len(im_datasets['test']))
    print("No. of Validation examples: ",len(im_datasets['val']))

    return dataloaders,c_t_idx
###############################################################################
def create_model():
    
    arch_type = 'vgg' if (args.arch is None) else args.arch
    model = models.vgg19(pretrained=True) if(arch_type == 'vgg')  else models.densenet121(pretrained=True) 
    input_node = 512*7*7 if(arch_type == 'vgg') else 1024
    hidden_units = 4096 if (args.hidden_units is None)  else args.hidden_units 
    
    for param in model.parameters():
        param.requires_grad = False
    hidden_units = int(hidden_units)
    classifier = nn.Sequential(
                 nn.Linear(input_node,hidden_units),
                 nn.ReLU(inplace = True),
                 nn.Dropout(0.5),
                 nn.Linear(hidden_units, 102),
                 nn.LogSoftmax(dim=1))
    model.classifier = classifier
    return model,hidden_units,arch_type
##########################################################################################
def train_model(model,dataloaders):
    learn_rate =  0.001 if (args.learning_rate is None) else args.learning_rate
    num_epochs =  3 if (args.epochs is None) else args.epochs
    device =  'cuda' if (args.gpu) else 'cpu'
    learn_rate = float(learn_rate)
    num_epochs = int(num_epochs)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
    model.to(device)
    running_loss=0
    for epoch in range(num_epochs):
        for inputs, targets in dataloaders['train']:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            probabilities = model.forward(inputs)
            loss = criterion(probabilities, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        else:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in dataloaders['val']:
                    inputs, labels = inputs.to(device), labels.to(device)                               
                    probabilities = model.forward(inputs)
                    batch_loss = criterion(probabilities, labels)
                    valid_loss += batch_loss.item()
                    # calaculate Accuracy of the model
                    ps = torch.exp(probabilities)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print("Epoch", (epoch+1),"/",num_epochs)
            print("Train loss:",running_loss/len(dataloaders['train']))
            print("Validation loss: ",valid_loss/len(dataloaders['val']))
            print("Validation accuracy: ",100*accuracy/len(dataloaders['val'])) 
            running_loss = 0
            model.train()
    return model,num_epochs,learn_rate,optimizer



###########################################################################################

def evaluate_model(model,dataloaders):    
    true_count = 0
    total = 0
    model.eval()
    device = 'cuda' if (args.gpu) else 'cpu'
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            probabilites = model.forward(inputs)
            _, Y = torch.max(probabilites, 1)
            total += labels.size(0)
            true_count += (Y == labels).sum().item()
    print('Accuracy on Test Set: ', (100 * true_count / total))

################################################################
def save_model(model,hidden_units,learn_rate,epochs,arch_type,optimizer):
    print("saving model")
    save_directory = 'check.pth' if (args.save_dir is None)  else args.save_dir
    checkpoint = {
                'model': model.cpu(),
                'features': model.features,
                'classifier': model.classifier,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'hidden_units':hidden_units,
                'learning_rate':learn_rate,
                'no_of_epochs':epochs,
                'structure': arch_type,
                'class_to_idx':model.class_to_idx
    }
    torch.save(checkpoint, save_directory)
    return 0

#########################################################################
def main():
    global args
    parser = argparse.ArgumentParser(description='Image Classifier training module')
    parser.add_argument('data_directory', help='path of the folder where dataset is stored')
    parser.add_argument('--save_dir', help='directory to save a neural network.')
    parser.add_argument('--arch', help='Two Options [vgg,densenet]')
    parser.add_argument('--learning_rate', help='learning rate for the model')
    parser.add_argument('--hidden_units', help='hidden units for the classifier')
    parser.add_argument('--epochs', help='number of epochs for training')
    parser.add_argument('--gpu',action='store_true', help='set gpu for training')
    args = parser.parse_args()
    if(not os.path.isdir(args.data_directory)):
        raise Exception(args.data_directory, 'direcory does not exist')
    data_dir = os.listdir(args.data_directory)
    if args.arch not in ('vgg','densenet',None):
        raise Exception('Please choose one otraf: vgg or densenet')      
    args = parser.parse_args()
    dataloaders,c_t_idx = process_data()
    model,hidden_units,arch_type = create_model()
    trained_model,epochs,learn_rate ,optimizer= train_model(model,dataloaders)
    evaluate_model(trained_model,dataloaders)
    model.class_to_idx = c_t_idx
    save_model(trained_model,hidden_units,learn_rate,epochs,arch_type,optimizer)
main()