

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
import os
from PIL import Image
import csv
import pandas as pd
import ssl


#ssl._create_default_https_context = ssl._create_unverified_context

resnet = models.resnet18(pretrained=True)
backbone = nn.Sequential(*list(resnet.children())[:-1])

# Freeze the weights 
for param in backbone.parameters():
    param.requires_grad = False


model1 = nn.Sequential(backbone, nn.Flatten())

# Set the device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model1 = model1.to(device)


# Data transformations for preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet mean and std
    transforms.Lambda(lambda x: x.unsqueeze(0)) # add dimension
])


   
def main():
    
    print("Starting ...")
    
    # Acess to data
    PATH_TRAIN = "/tsi/data_education/IMA206/2022-2023/VisioMel/data/train"
    PATH_TEST = "/tsi/data_education/IMA206/2022-2023/VisioMel/data/valid"
    #PATH_TRAIN = "../data/train"
    #PATH_TEST = "../data/valid"

    train_slide_folders = os.listdir(PATH_TRAIN)
    test_slide_folders = os.listdir(PATH_TEST)
    
    i = 0
    # Iterate over the training dataset
    for category in train_slide_folders:
        for slide_folder in os.listdir(os.path.join(PATH_TRAIN, category)):

            # Create a new CSV file for each slide folder
            csv_path = f"./embeddings_resnet/train_csv/{category}/{slide_folder}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Load and process patches for the current slide folder
                for patch_file in os.listdir(os.path.join(PATH_TRAIN, category, slide_folder)):
                    patch_path = os.path.join(os.path.join(PATH_TRAIN, category, slide_folder), patch_file)
                    try:
                        patch = Image.open(patch_path).convert("RGB")
                        patch = transform(patch)  # Apply transformations
                        embedding = model1(patch.to(device))  # Forward pass through the backbone
                        embedding = embedding.cpu().detach().numpy()[0]
                        
                        # Write the embedding to the CSV file immediately
                        writer.writerow(embedding)
                    except FileNotFoundError:
                        # Handle the FileNotFoundError
                        print(f"File {patch_path} not found. Skipping this iteration.")
                        continue

            i += 1
            print("We extracted the features of", i, "training slides")

    for category in test_slide_folders:
        for slide_folder in os.listdir(os.path.join(PATH_TEST, category)):
            i = 0

            # Create a new CSV file for each slide folder
            csv_path = f"./embeddings_resnet/valid_csv/{category}/{slide_folder}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)

                # Load and process patches for the current slide folder
                for patch_file in os.listdir(os.path.join(PATH_TEST, category, slide_folder)):
                    patch_path = os.path.join(os.path.join(PATH_TEST, category, slide_folder), patch_file)
                    try : 
                        patch = Image.open(patch_path).convert("RGB")
                        patch = transform(patch)  # Apply transformations
                        embedding = model1(patch)  # Forward pass through the backbone
                        embedding = embedding.cpu().detach().numpy()[0]
                        
                        # Write the embedding to the CSV file immediately
                        writer.writerow(embedding)
                    except FileNotFoundError:
                        # Handle the FileNotFoundError
                        print(f"File {patch_path} not found. Skipping this iteration.")
                        continue

            i += 1
            print("We extracted the features of", i, "validation slides")
    print("Done!")

  
main();

    
    





