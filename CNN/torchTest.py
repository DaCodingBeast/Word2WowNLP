import torch
import ast

# Example: the targets section from your CSV data as a string
target_str = "[0, 0, 0, 1, 0, 0, 1, 0]"

# Convert the string to a Python list
target_list = ast.literal_eval(target_str)

# Convert the list to a tensor
batch_targets = []

for target_str in target_str:
    # If the target is a tensor (already in tensor form), convert it to float tensor
    if isinstance(target_str, torch.Tensor):
        target_tensor = target_str.float()
    else:
                # Otherwise, parse it as a string (e.g., '[0, 0, 0, 1]')
        target_tensor = torch.tensor(eval(target_str), dtype=torch.float32)
            
    batch_targets.append(target_tensor)

        # Stack the list of tensors into a single tensor
targets = torch.stack(batch_targets)

        # Clamp the values (between 0.0 and 1.0) if necessary
targets = torch.clamp(targets, min=0.0, max=1.0)
# Print the tensor
print("Targets Tensor:")
print(targets)
