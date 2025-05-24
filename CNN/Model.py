import torch
import torch.nn as nn

class NounAdjectiveModel(nn.Module):
    def __init__(self, index_size, sent_size, pos_size, detailed_pos_size, dep_size, ent_type_size, embed_dim, kernel_size):
        super(NounAdjectiveModel, self).__init__()
        
        # Embedding layers for POS, detailed POS, dependency, and entity type
        self.index_embedding = nn.Embedding(index_size, embed_dim)
        self.pos_embedding = nn.Embedding(pos_size, embed_dim)
        self.detailed_pos_embedding = nn.Embedding(detailed_pos_size, embed_dim)
        self.dep_embedding = nn.Embedding(dep_size, embed_dim)
        self.ent_type_embedding = nn.Embedding(ent_type_size, embed_dim)
        self.sent_embedding = nn.Embedding(sent_size, embed_dim)

        self.maxIndex = index_size
        # Convolutional layers to process word-level features (token-level)
        self.conv1 = nn.Conv1d(in_channels=embed_dim * 6,  # POS + detailed_pos + dep + ent_type
                               out_channels=64,
                               kernel_size=kernel_size,
                               padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=1)
        
        # Fully connected layers for token-level output
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, index_size)  # Binary output indicating if a word is an adjective modifying a noun
    def forward(self, index, pos, detailed_pos, dep, ent_type, sent):
    # Embedding lookup for all features
        pos_embed = self.pos_embedding(pos.int())
        detailed_pos_embed = self.detailed_pos_embedding(detailed_pos.int())
        dep_embed = self.dep_embedding(dep.int())
        ent_type_embed = self.ent_type_embedding(ent_type.int())
        index_embed = self.index_embedding(index.int())
        sent_type = self.sent_embedding(sent.int())
        
        # Combine all feature embeddings (concatenate)
        x = torch.cat([pos_embed, detailed_pos_embed, dep_embed, ent_type_embed, index_embed, sent_type], dim=-1)
        
        # Transpose for convolution (batch_size, channels, length)
        x = x.permute(0, 2, 1)
        # Apply convolutional layers
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        
        # Transpose back to (batch_size, sentence_length, num_channels)
        x = x.permute(0, 2, 1)
        # print(x)
        
        # Fully connected layers for token-level classification
        x = self.fc1(x)
        x = torch.relu(x)
        # print(x,"dfdfdf")
        x = self.fc2(x)  # Output shape: (batch_size, sentence_length, 1)
    
        # print("Before fc2:", x.shape)
        # Apply sigmoid to get probabilities between 0 and 1
        x = torch.sigmoid(x).squeeze(-1)  # Shape: (batch_size, sentence_length)

        trimmed_outputs = []
        lengths = index.max().int() +1
        # print(len(index[0].int()))
        for i in range(len(index[0].int())):
            outpots =[]
            # print(lengths)

            # print(x[0][i], lengths)
            # print(str(pos[0][i]) + "posss")
            if(pos[0][i] == 7):
                for a in range(lengths):
                    # print(x[0][i][a])
                    outpots.append(x[0][i][a].round())
                b = torch.tensor(outpots)
                print(b)
                trimmed_outputs.append(outpots)
        x = torch.tensor(trimmed_outputs, dtype=torch.float32, requires_grad= True)  # Combine all trimmed tensors

        # print("After fc2:", x.shape)
        # Mask outputs to retain only predictions for `pos == 0`
        # mask = (pos == 7)
        # x = x[mask]  # Shape: (num_pos_0_words,)

        return x

