import torch
import torch.nn as nn
import torch.optim as optim

# Assuming a simple binary classification model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)  # Example: input of size 10, output of size 1

    def forward(self, x):
        return torch.sigmoid(self.fc(x))  # Sigmoid to output probability between 0 and 1

# Model instance
model = SimpleModel()

# Loss function
criterion = nn.BCELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy data
inputs = torch.randn(3, 10)  # Batch of 3, each with 10 features
targets = torch.tensor([[1.0], [0.0], [1.0]])  # Binary targets (0 or 1)

# Ensure outputs require gradients
outputs = model(inputs)

# Check that the outputs require gradients
print(outputs.requires_grad)  # Should be True

# Calculate loss
loss = criterion(outputs, targets)

# Backward pass
loss.backward()  # This should work as long as outputs require gradients

# Optimizer step
optimizer.step()
