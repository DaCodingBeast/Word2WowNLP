import torch.optim as optim
from torch.utils.data import DataLoader
from Model import NounAdjectiveModel
from PosDataset import PosDataset,collate_fn
from io import StringIO
import pandas as pd
import torch.nn as nn
import torch
import csv

df = pd.read_csv("encoded_dataset_50.csv")

# Prepare a mock CSV file using StringIO
        # Initialize model parameters
# pos_size = 18            # Example size for POS tag vocabulary
# detailed_pos_size = 51   # Example size for detailed POS tag vocabulary
# dep_size = 46            # Example size for dependency tags
# ent_type_size = 19       # Example size for entity type tags
embed_dim = 64           # Dimensionality of embeddings
kernel_size = 3          # Kernel size for convolution layers


pos_size = df['pos'].unique().max()
dep_size = df['dep'].unique().max()
detailed_pos_size = df['detailed_pos'].unique().max()
ent_type_size = df['ent_type'].unique().max()
unique_indexes = df['index'].unique().max()
unique_sentences = df['sent'].unique().max()

# Create an instance of the model
model = NounAdjectiveModel(unique_indexes+1,unique_sentences+1,pos_size+1, detailed_pos_size+1, dep_size+1, ent_type_size+1, embed_dim, kernel_size)
# for param in model.parameters():
#     print(param.requires_grad)      

# Create the dataset and dataloader using the CSV file path
dataset = PosDataset("encoded_dataset_50.csv")
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)    

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 20

from customLoss import CustomLoss
from torch.nn import BCELoss

criterion = BCELoss()  # Binary Cross-Entropy Loss

with open('encoded_dataset_50.csv', mode='r', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    # Get the header row (first row)
    headers = next(reader)
    # print("Column Names:", headers)

for param in model.parameters():
    param.requires_grad = True
# Training loop
for epoch in range(num_epochs):

    model.train()  # Set model to training mode
    running_loss = 0.0

    for batch in dataloader:
        index = batch["index"]
        pos = batch["pos"]
        detailed_pos = batch["detailed_pos"]
        dep = batch["dep"]
        ent_type = batch["ent_type"]
        sent = batch["sent"]
        # print(batch["targets"],"Targetsddddddddddddddd")

        targets = torch.clamp(batch["targets"], min=0.0, max=1.0)

        index = index
        pos = pos
        detailed_pos = detailed_pos
        dep = dep
        ent_type = ent_type
        sent = sent

        outputs = model(index,pos,detailed_pos,dep,ent_type,sent) 
        
        # print(pos)
        # print(dep)

        mask = (pos == 7)
        targets = targets[mask]
        print("Target "+str(targets.squeeze()))
        print("Ouput "+str(outputs.squeeze()))

        # for param in model.parameters():
        #     if param.grad is not None:
        #         print(param.grad)  # Should show gradients if calculated
        #     else:
        #         print("No gradient for this parameter.")
        # Calculate loss
        loss = criterion(outputs.squeeze(), targets.squeeze())
        optimizer.zero_grad()  # Zero out old gradients
        loss.backward()        # Backpropagate to compute gradients
        optimizer.step()       # Update model parameters

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')