

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
import os
import numpy as np
from PIL import Image
import csv
import pandas as pd
import ssl
from pickle import dump
import random


#ssl._create_default_https_context = ssl._create_unverified_context

"""
#There are 2 different models implemented in the AbMIL paper but i chose the gatedattention because it seems more sophisticated
#what we do is we define two seperate attention models, one nonlinear and one with a sigmoid activation function 
#we obtain the weights needed by doing elemnt wise multiplication on the results from these 2 models
#We need the weights later to visualize the most important patches that we took into consideration when making our decision

"""

class GatedAttention(nn.Module):
    def __init__(self , input_dim, hidden_dim, output_dim):
        super(GatedAttention, self).__init__()

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(hidden_dim, output_dim)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim*output_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        A_V = self.attention_V(x)  # N x hidden_dim
        A_U = self.attention_U(x)  # N x hidden_dim
        A = self.attention_weights(A_V * A_U) # element wise multiplication # N x output_dim
        A = torch.transpose(A, 1, 0)  # output_dim x N
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, x)  # Koutput_dim x input_dim
        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A #Y_prob is the probabilities of each bag of having class = 1 , Y_hat is the binary result of each bag (slide/ slice idk) and A is the attention weights.

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        
        Y = float(Y)
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
    
    def calculate_accuracy(self, X, Y):
        Y = float(Y)
        _, Y_hat, _ = self.forward(X)
        accuracy = Y_hat.eq(Y).cpu().float().mean().item()
        return accuracy
    
        
    
   
def main():
    
    # Define the dimensions of the input and output for the MIL network
    input_dim = 512    
    hidden_dim = 256
    output_dim = 1

    # Create the MIL network
    mil_network = GatedAttention(input_dim, hidden_dim, output_dim)


    # Define the loss function
    # Calculate class weights
    total_samples = 1017 + 190
    weights = [total_samples / 1017, total_samples / 190]
    class_weights = torch.tensor(weights, dtype=torch.float)

    # Define the criterion with class weights
    criterion = nn.BCELoss(weight=class_weights)
    #criterion = nn.BCELoss() #Balanced / weighted

    # Define the optimizer
    optimizer = optim.Adam(mil_network.parameters(), lr=0.001) # ~ 1e-5

    # Set the device to GPU if available
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    mil_network = mil_network.to(device)

    # Define the number of epochs
    num_epochs = 100

    # Acess to data
    PATH_TRAIN = "./embeddings_resnet/train_csv"
    PATH_TEST = "./embeddings_resnet/valid_csv"

    train_slides = os.listdir(PATH_TRAIN)
    test_slides = os.listdir(PATH_TEST)
    df = pd.read_csv('./train_labels.csv', index_col=0)
    
    # Training loop
    train_loss_dict = {}
    val_loss_dict = {}
    accuracy_val_dict = {}
    
    
    for epoch in range(1,num_epochs+1):
        print("epoch n° ",epoch)
        # Set the model to train mode
        mil_network.train()
        train_loss = 0.0 
        
        random.shuffle(train_slides) 
        random.shuffle(test_slides) 
        dropIndex=random.randint(19, 29)  # droupout to induce more randomness
        
        for slide_folder in train_slides:
            embeddings = []  # Store embeddings for the slide

            # Load embeddings from the CSV file
            with open(os.path.join(PATH_TRAIN, slide_folder), 'r') as f:
                reader = csv.reader(f)
                rowIndex = 0
                for row in reader:
                    rowIndex +=1
                    if rowIndex % dropIndex ==0:
                        pass
                    else:
                        embedding = torch.tensor(np.array(row, dtype=np.float32)).to(device) 
                        embeddings.append(embedding)
                
            # Convert embeddings to a tensor
            features = torch.stack(embeddings).to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Reshape the features
            batch_size, num_embeddings= features.size()
                        
            slide_label = df['relapse'][slide_folder[:-4] + str(".tif")]
            
            loss, weight = mil_network.calculate_objective(features, slide_label)
            loss = loss.mean()
            #train_running_loss += loss  
            
            # Backward pass and optimization
            loss.backward(retain_graph=True)  #loss.backward()
            optimizer.step()
            train_loss+=loss.item()
                  
        
        # Calculate average training loss
        train_average_loss = train_loss / len(train_slides)
        print("train average loss = ", float(train_average_loss))     
        train_loss_dict[epoch] = float(train_average_loss)
        #train_weights.append(weight)

        # Set the model to evaluation mode
        # mil_network.eval()
        
        accuracy_scores = []
        test_loss = 0.0

        for slide_folder in test_slides:
            embeddings = []  # Store embeddings for the slide

                # Load embeddings from the CSV file
            with open(os.path.join(PATH_TEST, slide_folder), 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    embedding = torch.tensor(np.array(row, dtype=np.float32)).to(device)
                    embeddings.append(embedding)
                
             # Convert embeddings to a tensor
            features = torch.stack(embeddings).to(device)
        
        
            # Reshape the features
            batch_size, num_embeddings = features.size()
        

            slide_label = df['relapse'][slide_folder[:-4] + str(".tif")]

            loss, weight = mil_network.calculate_objective(features, slide_label)
            loss = loss.mean()
            
            accuracy_scores.append(mil_network.calculate_accuracy(features, slide_label))
       
            #test_running_loss += loss
            test_loss+=loss.item()

        # Calculate average validation loss
        average_loss = test_loss / len(test_slides)
        accuracy = np.mean(accuracy_scores)
        #test_weights.append(weight)
        print("validation average loss = ", float(average_loss))
        val_loss_dict[epoch] = float(average_loss)
        print("validation accuracy = ", float(accuracy))
        accuracy_val_dict[epoch] = float(accuracy)
        
        

    torch.save(mil_network.state_dict(), './mil_network_weights_V2.pth')
    
    # Save the training loss values
    with open('./train_loss.pkl_V2', 'wb') as file:
        dump(train_loss_dict, file)
    
    # Save the validation loss values
    with open('./val_loss.pkl_V2', 'wb') as file:
        dump(val_loss_dict, file)
    
    # Save the accuracy values  
    with open('./accuracy_val.pkl_V2', 'wb') as file:
        dump(accuracy_val_dict, file)
    print("Done! Thank you for being patient")
    
    
    #loaded_model = GatedAttention(input_dim, hidden_dim, output_dim)
    #loaded_model.load_state_dict(torch.load('./mil_network_weights.pth'))
    #loaded_model.eval()

main();

    
    





