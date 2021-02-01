import argparse
import os
import time
import torch
import matplotlib.pyplot as plt 
import numpy as np
import json
from torch import nn, optim
from torchvision import datasets, models, transforms
from collections import OrderedDict
from PIL import Image





  
#####################################################################################

def load_model(filepath):
    checkpoint = torch.load(filepath)
    hidden_units = checkpoint['hidden_units']
    learning_rate=checkpoint['learning_rate']
    epochs = checkpoint['no_of_epochs']
    structure = checkpoint['structure']
    structure='densenet'
    device =  'cuda' if (args.gpu) else 'cpu'
    model = models.vgg19(pretrained=True) if(structure == 'vgg')  else models.densenet121(pretrained=True) 
    input_node = 512*7*7 if(structure == 'vgg') else 1024  
    model.class_to_idx = checkpoint['class_to_idx']
    print(structure)
    for param in model.parameters():
        param.requires_grad = False   
    model.classifier =nn.Sequential(
                     nn.Linear(input_node,hidden_units),
                     nn.ReLU(inplace = True),
                     nn.Dropout(0.5),
                     nn.Linear(hidden_units, 102),
                     nn.LogSoftmax(dim=1))
    model = model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if torch.cuda.is_available() and device == 'gpu':
        model.cuda()
    return model

########################################################################################

def process_image(image):
    pil_image = Image.open(image)
    process_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])   
    processed_image = process_image(pil_image)
    return processed_image
###############################################################################
def predict(image_path, model, topk=5):

    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    model.to(device)
    
    processed_image = process_image(image_path).to(device)
    new_image = processed_image.unsqueeze_(0)
    
    model.eval()
    with torch.no_grad():
        logps = model.forward(new_image)
    
    probs = torch.exp(logps)
    top_k, top_classes_idx = probs.topk(topk, dim=1)
    top_k, top_classes_idx = np.array(top_k.to(device)[0]), np.array(top_classes_idx.to(device)[0])
    
    # Inverting dictionary
    idx_to_class = {a: b for b, a in model.class_to_idx.items()}
    
    top_classes = []
    for index in top_classes_idx:
        top_classes.append(idx_to_class[index]) 

    return list(top_k), list(top_classes)

###################################################################################

def main():
    global args
    parser = argparse.ArgumentParser(description='Prediction Modeule for Image Classifier')
    parser.add_argument('imagepath', action="store", type=str , help='path of the image to be predicted')
    parser.add_argument('checkpoint',default='check.pth', help='Path of saved checkpoint')
    parser.add_argument('--top_k',  type=int, default= 5, help='number of top classes to be predicted')
    parser.add_argument('--category_names', action= 'store_true', help='category_names cat_to_name.json')
    parser.add_argument('--gpu',action='store_true', help='enable gpu')
    args = parser.parse_args()   
    model = load_model(args.checkpoint)
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file) 
    probabilities, classes=predict(args.imagepath,model,args.top_k)
    classes_names = [cat_to_name[number] for number in classes]
    print('predicted class',classes_names[0])
    print("top k classes",classes_names)
    print("probabilities of top class",probabilities)

 
    
main()