# Import Libraries #
import argparse
import json
import PIL
import torch
import numpy as np

from math import ceil
from train import check_device
from torchvision import models

# Define function that parses arguments to run from the command line
def arg_parser():
    # Define a parser
    parser = argparse.ArgumentParser(description="Image Classifier Settings")

    parser.add_argument('--image', 
                        type=str, 
                        required=True)

    parser.add_argument('--checkpoint', 
                        type=str)
    
    parser.add_argument('--top_k', 
                        type=int)
    
    parser.add_argument('--category_names', 
                        type=str)

    parser.add_argument('--gpu', 
                        action="store_true")

    args = parser.parse_args()
    
    return args

# define function to load saved checkpoint from train.py 
def load_savedcheckpoint(checkpoint_path):
    
    checkpoint = torch.load('savedcheckpoint.pth')
    
    model = models.alexnet(pretrained=True)
    model.name = "alexnet"
      
    for param in model.parameters(): param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
        
    return model

# define function to process images to crop and resize images
def processimage(image_path):
    testimage = PIL.Image.open(image_path)

    orig_width, orig_height = testimage.size

    if orig_width < orig_height: 
        resize_size=[256, 256**2]
    else: 
        resize_size=[256**2, 256]
        
    testimage.thumbnail(size=resize_size)

    # Crop image from center to create 224x224 image
    width, height = testimage.size 
    left = (width - 224)/2 
    top = (height - 224)/2
    right = left + 224 
    bottom = top + 224
    testimage = testimage.crop((left, top, right, bottom))

    # Convert to numpy 244 x 244 np_image
    np_image = np.array(testimage)/255

    # Normalize image
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
        
    np_image = np_image.transpose(2, 0, 1)
    
    return np_image

# define function that will predict image based on arguments provided
def predict(imagetensor, model, device, cat_to_name, top_k):
    
    # check top_k, if not provided set to 5
    if type(top_k) == type(None):
        top_k = 5
    
    model.eval();

    torchimage = torch.from_numpy(np.expand_dims(imagetensor, 
                                                  axis=0)).type(torch.FloatTensor)

    model=model.cpu()

    logprob = model.forward(torchimage)

    linearprob = torch.exp(logprob)

    topprob, toplabel = linearprob.topk(top_k)
    
    topprob = np.array(topprob.detach())[0] 
    
    toplabel = np.array(toplabel.detach())[0]
    
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    toplabel = [idx_to_class[lab] for lab in toplabel]
    topflower = [cat_to_name[lab] for lab in toplabel]
    
    return topprob, toplabel, topflower


def print_prob(prob, flower):
    
    for i, j in enumerate(zip(flower, prob)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, probability: {}%".format(j[1], ceil(j[0]*100)))
    
def main():
    
    # get argument to parse for prediction
    args = arg_parser()
    
    # Load category names from json file
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    model = load_savedcheckpoint(args.checkpoint)
    
    # Process Image
    imagetensor = processimage(args.image)
    
    # Check for GPU
    device = check_device(gpucpu=args.gpu);
    
    # Use processed_image to predict
    topprob, toplabel, topflower = predict(imagetensor, model, 
                                                 device, cat_to_name,
                                                 args.top_k)
    
    # Print out probabilities
    print_prob(topflower, topprob)

# Run Program
if __name__ == '__main__': 
    main()