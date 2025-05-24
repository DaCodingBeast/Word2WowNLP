import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import ast

# Define the CNN-based model
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, kernel_size=1, num_filters=64):
        super(CNNModel, self).__init__()
        
        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 1D Convolution layer with kernel_size=1
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size)
        
        # Fully connected layer for binary classification
        self.fc = nn.Linear(num_filters, 1)
        
        # Sigmoid activation for binary output
        self.sigmoid = nn.Sigmoid()

    def forward(self, word_pair):
        # Get embeddings for the two words
        word1, word2 = word_pair[:, 0], word_pair[:, 1]
        word1_emb = self.embeddings(word1).unsqueeze(2)  # Shape: [batch_size, embedding_dim, 1]
        word2_emb = self.embeddings(word2).unsqueeze(2)  # Shape: [batch_size, embedding_dim, 1]
        
        # Concatenate the embeddings of both words along the second axis (embedding_dim)
        x = torch.cat([word1_emb, word2_emb], dim=2)  # Shape: [batch_size, embedding_dim, 2]
        
        # Apply 1D convolution with the adjusted kernel size
        x = self.conv1(x)  # Shape: [batch_size, num_filters, 2 - kernel_size + 1]
        
        # Global max pooling (pooling over the last dimension)
        x = torch.max(x, dim=2)[0]  # Shape: [batch_size, num_filters]
        
        # Fully connected layer to output a single score (0 or 1)
        x = self.fc(x)  # Shape: [batch_size, 1]
        
        # Sigmoid activation for binary classification
        output = self.sigmoid(x).squeeze()  # Shape: [batch_size]
        
        return output

# Load the data
df = pd.read_csv('encoded_dataset.csv')

# Convert the 'results' column from string representation of lists to actual lists
df['results'] = df['results'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)



# Define function to generate word pairs with their labels
def generate_word_pairs(group):
    word_pairs = []
    labels = []
    for _, row in group.iterrows():
        word = row['word']
        pos = row['pos']
        results = row['results']
        
        # Create word pairs (for each word, we pair it with every other word in the sentence)
        for index in results:
            if index != -1:
                noun_row = group[group['index'] == index]
                if not noun_row.empty and noun_row['pos'].iloc[0] in ['NN', 'NNS', 'NNP', 'NNPS']:
                    # Relationship exists between adjective and noun
                    word_pairs.append((word, noun_row['word'].iloc[0]))
                    labels.append(1)  # Positive label (relationship exists)
                else:
                    labels.append(0)  # Negative label (no relationship)
    
    return word_pairs, labels












# Group data by 'group_id'
grouped = df.groupby('group_id')

# Generate word pairs and labels
all_word_pairs = []
all_labels = []

for group_id, group in grouped:
    word_pairs, labels = generate_word_pairs(group)
    all_word_pairs.extend(word_pairs)
    all_labels.extend(labels)
















# Convert word pairs and labels into a DataFrame for training
word_pair_df = pd.DataFrame(all_word_pairs, columns=["word1", "word2"])
word_pair_df['label'] = all_labels
print(all_word_pairs)
# Show the generated word pairs and labels
print(word_pair_df.head())

# Encode words (convert words to indices or embeddings)
le = LabelEncoder()
word_pair_df['word1_enc'] = le.fit_transform(word_pair_df['word1'])
word_pair_df['word2_enc'] = le.transform(word_pair_df['word2'])
















# Define the dataset class
class WordPairDataset(Dataset):
    def __init__(self, word_pairs, labels):
        self.word_pairs = word_pairs
        self.labels = labels

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx):
        word1, word2 = self.word_pairs[idx]
        label = self.labels[idx]
        return torch.tensor([word1, word2]), torch.tensor(label)

# Create dataset and dataloaders
dataset = WordPairDataset(word_pair_df[['word1_enc', 'word2_enc']].values, word_pair_df['label'].values)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model, loss function, and optimizer
vocab_size = len(le.classes_)
model = CNNModel(vocab_size)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):  # Example: 10 epochs
    model.train()
    for batch in train_loader:
        word_pairs, labels = batch
        optimizer.zero_grad()

        # Forward pass
        outputs = model(word_pairs).squeeze()

        # Calculate loss
        loss = loss_fn(outputs, labels.float())

        # Backward pass
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
