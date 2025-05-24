import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder

class PosDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.groups = self.data.groupby("group_id")
        self.group_ids = list(self.groups.groups.keys())
        
        # Initialize label encoders for 'dep' and 'ent_type'
        self.dep_encoder = LabelEncoder()
        self.ent_type_encoder = LabelEncoder()
        
        # Fit the encoders on the entire dataset
        self.dep_encoder.fit(self.data["dep"].unique())
        self.ent_type_encoder.fit(self.data["ent_type"].unique())

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        group_id = self.group_ids[idx]
        group_data = self.groups.get_group(group_id)
        
        # Extract data for each column (index, pos, detailed_pos, dep, ent_type, sent)
        indexes = torch.tensor(group_data["index"].values, dtype=torch.long)
        pos = torch.tensor(group_data["pos"].values, dtype=torch.long)
        detailed_pos = torch.tensor(group_data["detailed_pos"].values, dtype=torch.long)
        
        # Encode 'dep' and 'ent_type' using the fitted label encoders
        dep = torch.tensor(self.dep_encoder.transform(group_data["dep"].values), dtype=torch.long)
        ent_type = torch.tensor(self.ent_type_encoder.transform(group_data["ent_type"].values), dtype=torch.long)
        
        # Sentences are text, so we return them as raw text
        sent = group_data["sent"].values.tolist()  # List of sentences
        
        # Assuming 'results' column contains the target values as a list (e.g., "[0, 1, 2]")
        targets = torch.tensor([int(x) for x in group_data["results"].iloc[-1][1:-1].split(",")], dtype=torch.long)
        
        return indexes, pos, detailed_pos, dep, ent_type, sent, targets
from torch.nn.utils.rnn import pad_sequence

# Custom Collate Function to Pad Sequences
def collate_fn(batch):
    indexes = [item[0] for item in batch]
    pos = [item[1] for item in batch]
    detailed_pos = [item[2] for item in batch]
    dep = [item[3] for item in batch]
    ent_type = [item[4] for item in batch]
    sent = [item[5] for item in batch]
    targets = [item[6] for item in batch]

    # Pad sequences (for numeric data)
    indexes_padded = pad_sequence(indexes, batch_first=True, padding_value=0)
    pos_padded = pad_sequence(pos, batch_first=True, padding_value=0)
    detailed_pos_padded = pad_sequence(detailed_pos, batch_first=True, padding_value=0)
    dep_padded = pad_sequence(dep, batch_first=True, padding_value=0)
    ent_type_padded = pad_sequence(ent_type, batch_first=True, padding_value=0)
    
    # Targets do not need padding, since they are indexes
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1)  # Use -1 to represent padding in targets
    
    # Return padded sequences and raw sentences (sentences can be processed later if needed)
    return indexes_padded, pos_padded, detailed_pos_padded, dep_padded, ent_type_padded, sent, targets_padded

class PosCNN(nn.Module):
    def __init__(self, pos_size, detailed_pos_size, entity_size, dep_size, embed_dim):
        super(PosCNN, self).__init__()

        # Embeddings for various features
        self.pos_embed = nn.Embedding(pos_size, embed_dim)  
        self.detailed_pos_embed = nn.Embedding(detailed_pos_size, embed_dim)
        self.entity_embed = nn.Embedding(entity_size, embed_dim)  
        self.dep_embed = nn.Embedding(dep_size, embed_dim)

        # Convolutional layer
        self.conv = nn.Conv1d(embed_dim * 5, 128, kernel_size=3, padding=1)  
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Fully connected layer to predict relationships between each noun-adjective pair
        # The output here will be the relationships between nouns and adjectives (as a matrix)
        self.fc = nn.Linear(128, 1)  # Each word output a single value (relationship with other words)

    def forward(self, indexes, pos, detailed_pos, entity, dep, num_nouns, num_adjectives):
        # Get embeddings for each input
        pos_embeds = self.pos_embed(pos)
        detailed_pos_embeds = self.detailed_pos_embed(detailed_pos)
        entity_embeds = self.entity_embed(entity)
        dep_embeds = self.dep_embed(dep)

        # Concatenate all embeddings into a single tensor
        x = torch.cat([pos_embeds, detailed_pos_embeds, entity_embeds, dep_embeds], dim=-1)

        # Permute for the convolutional layer (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)

        # Apply convolution, ReLU activation, and pooling
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)

        # Global average pooling
        x = x.mean(dim=-1)

        # Pass through the fully connected layer to get relationships between nouns and adjectives
        # Since output_size is dynamic (depending on num_nouns and num_adjectives)
        relationships = self.fc(x)

        # Reshape to get the relationship matrix (num_nouns x num_adjectives)
        relationships = relationships.view(-1, num_nouns, num_adjectives)  # Reshape output

        return relationships


def masked_loss(outputs, targets, padding_idx=-1):
    print(targets)
    print(outputs)
    print("_"*100)

    # Create a mask to ignore padding
    mask = targets != padding_idx
    
    # Apply the mask to the targets and outputs
    targets = targets[mask]  # Flatten and remove padded values
    outputs = outputs.view(-1, outputs.size(-1))  # Flatten the outputs
    outputs = outputs[mask.view(-1)]  # Apply the mask to the flattened outputs
    
    # Calculate loss for the masked targets and outputs
    return criterion(outputs, targets)

import torch.nn.functional as F

def print_probabilities(outputs, targets):
    # Apply softmax to the outputs
    probabilities = F.softmax(outputs, dim=-1)
    
    # Print the probabilities and the target index
    for i, output in enumerate(probabilities):
        print(f"Output probabilities for word {i}: {output.tolist()}")
        print(f"Target index for word {i}: {targets[i].item()}")
        print(f"Predicted class: {torch.argmax(output).item()}")
        print("-" * 40)

def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for idx, (indexes, pos, detailed_pos, dep, ent_type, sent, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(indexes, pos, detailed_pos, dep, ent_type)

            # Print model inputs and outputs to inspect them
            if idx == 0:  # Only print for the first batch to avoid excessive output
                print(f"Epoch {epoch+1}, Batch {idx+1}:")
                print(f"Indexes (padded): {indexes}")
                print(f"POS (padded): {pos}")
                print(f"Detailed POS (padded): {detailed_pos}")
                print(f"Dependency (padded): {dep}")
                print(f"Entity Type (padded): {ent_type}")
                print(f"Sentences: {sent}")  # Sentences are raw text
                print(f"Targets (padded): {targets}")
                print(f"Model Outputs: {outputs}")
                print_probabilities(outputs, targets)

            # Apply the masked loss function
            loss = masked_loss(outputs, targets)

            # Print loss for each batch
            print(f"Batch {idx+1} Loss: {loss.item():.4f}")

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Total Loss: {total_loss:.4f}")



# Parameters (Adjust based on your dataset)
pos_size = 17
detailed_pos_size = 50
entity_size = 18
dep_size = 45
embed_dim = 64
output_size = 10  # Placeholder (adjust based on the max number of results)

batch_size = 16
epochs = 10
lr = 0.001

# Dataset and DataLoader
dataset = PosDataset("encoded_dataset.csv")
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Model, Loss, Optimizer
model = PosCNN(pos_size= pos_size, detailed_pos_size = detailed_pos_size, entity_size = entity_size, embed_dim=embed_dim, dep_size= dep_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs)
