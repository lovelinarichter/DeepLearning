# Import Libraries #
import argparse
import torch
from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# Define function that parses arguments to run from the command line, all arguments are optional
def arg_parser():
    parser = argparse.ArgumentParser(description="Image Classifier Settings")
    
    # define architecture from the model
    parser.add_argument('--arch', 
                        type=str)
    
    # define checkpoint directory 
    parser.add_argument('--save_dir', 
                        type=str)
    
    parser.add_argument('--learningrate', 
                        type=float)
    
    parser.add_argument('--hidden_units', 
                        type=int)
    
    parser.add_argument('--epochs', 
                        type=int)
    
    parser.add_argument('--gpu', 
                        action="store_true")
    
    # Parse arguments
    args = parser.parse_args()
    return args

# define function training data transformer 
def traindata_transformer(train_dir):
   # Define transformation
   train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    # Load the Data
   train_image_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
   return train_image_datasets

#define function valid data transformer
def validdata_transformer(valid_dir):
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    return valid_image_datasets 

# define fucntion test data transformer
def testdata_transformer(test_dir):
   test_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], 
                                                              [0.229, 0.224, 0.225])])
    # Load the Data
   test_image_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
   return test_image_datasets

# define function to load data
def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=64)
    return loader

# Define function to check available device
def check_device(gpucpu):
    if not gpucpu:
        return torch.device("cpu")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    return device

# define pretrained model alexnet
def pretrained_model(architecture = 'alexnet'):

    if type(architecture) == type(None): 
        model = models.alexnet (pretrained = True)
        model.name = 'alexnet'
    else: 
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    
    for param in model.parameters():
        param.requires_grad = False 
    return model

# define function model classifer
def model_classifier(model, hidden_units):
    
    if type(hidden_units) == type(None): 
        hidden_units = 4096
   
    # Define the model classifier uwing sequential with OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('linear1', nn.Linear(9216, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.5)),
                          ('linear2', nn.Linear(hidden_units, 2048)),
                          ('output1', nn.LogSoftmax(dim=1))
                          ]))
      
    return classifier

# define validation of model function 
def modelvalidation(model, valid_loader, criterion, device):
    model.to(device)

    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:

        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

# define training of model
def modeltrainer(model, train_loader, valid_loader, device, criterion, optimizer, epochs, print_every, steps):
    
    if type(epochs) == type(None):
        epochs = 5
 
    print("Training process started")

    for e in range(epochs):
        running_loss = 0
        model.train() 
        accuracy_train = 0
    
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1
        
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            # Forward and backward pass
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            ps_train = torch.exp(outputs).data
            equality_train = (labels.data == ps_train.max(1)[1])
            accuracy_train += equality_train.type_as(torch.FloatTensor()).mean()
        
            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = modelvalidation(model, valid_loader, criterion, device)
            
                print("Epoch: {}/{} |".format(e+1, epochs),
                  "Training Loss: {:.4f}..".format(running_loss/print_every),
                  "Validation Loss: {:.4f}..".format(valid_loss/len(valid_loader)),
                  "Training Accuracy: {:.4f}".format(accuracy_train/len(train_loader)),
                  "Validation Accuracy: {:.4f}".format(accuracy/len(valid_loader)))
            
                running_loss = 0
                model.train()
            
    return model

# define validate model accuracy rate
def validatemodel(model, test_loader, device):
    
    # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy: %d%%' % (100 * correct/total))

# define function to save checkpoint
def dircheckpoint(model, train_data):    
    
    # Create `class_to_idx` attribute in model
    model.class_to_idx = train_data.class_to_idx
    
    # Create checkpoint dictionary
    checkpoint = {'classifier': model.classifier,
    'class_to_idx': model.class_to_idx,
    'state_dict': model.state_dict()}

    # Save checkpoint
    torch.save(checkpoint, 'savedcheckpoint.pth')

def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
  
    # define directory
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #transform data
    train_data = traindata_transformer(train_dir)
    valid_data = validdata_transformer(valid_dir)
    test_data = testdata_transformer(test_dir)
    
    #load data
    train_loader = data_loader(train_data)
    valid_loader = data_loader(valid_data, train=False)
    test_loader = data_loader(test_data, train=False)
    
    # call pretrained model function
    model = pretrained_model(architecture=args.arch)
    
    # Call classifier function
    model.classifier = model_classifier(model,
                                        hidden_units=args.hidden_units)
   
    # Check for device
    device = check_device(gpucpu=args.gpu)
    
    # Send model to device
    model.to(device)
    
    # Assign learning rate
    if type(args.learningrate) == type(None):
        learningrate = 0.001
    else: learningrate = args.learningrate
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learningrate)
    
    # Define number of passes and print loss every batch of 50 images
    print_every = 50
    steps = 0
      
    trained_model = modeltrainer(model, train_loader, valid_loader, device, criterion, optimizer, args.epochs, print_every, steps)
    
    print("Training process completed")
    
    # call validate model function
    validatemodel(trained_model, test_loader, device)
    
    # call save checkpoint function
    dircheckpoint(trained_model, train_data)

# Run Program
if __name__ == '__main__': main()