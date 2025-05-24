import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, base_loss=nn.BCELoss(), penalty=10.0):
        super(CustomLoss, self).__init__()
        self.base_loss = base_loss
        self.penalty = penalty

    def forward(self, outputs, targets):
        # Compute the base loss (e.g., BCELoss)
        base_loss = self.base_loss(outputs, targets)

        # Find mismatches where target has 1 but output does not
        mismatches = (targets == 1) & (outputs < 0.5)  # Using a threshold of 0.5 for binary classification
        penalty_loss = mismatches.sum() * self.penalty  # Add penalty for each mismatch

        # Combine the base loss with the penalty loss
        total_loss = base_loss + penalty_loss
        return total_loss
