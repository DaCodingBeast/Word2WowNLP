import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class PosDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.groups = self.data.groupby("group_id")
        self.group_ids = list(self.groups.groups.keys())

    def __len__(self):
        return len(self.group_ids)

    def __getitem__(self, idx):
        group_id = self.group_ids[idx]
        group_data = self.groups.get_group(group_id)

        # Extract data for each column (ensure conversion from pandas Series)
        index = torch.tensor(group_data["index"].values, dtype=torch.int)
        pos = torch.tensor(group_data["pos"].values, dtype=torch.int)
        detailed_pos = torch.tensor(group_data["detailed_pos"].values, dtype=torch.int)
        dep = torch.tensor(group_data["dep"].values, dtype=torch.int)
        ent_type = torch.tensor(group_data["ent_type"].values, dtype=torch.int)
        sent = torch.tensor(group_data["sent"].values, dtype=torch.int)
       
        # Get the 'results' column as a string
        # Get the 'results' column as a string
        results_np = group_data["results"].values  # This is a numpy array of lists

        # Initialize an empty list to store the final targets
        targets_list :list[list] = []

        # Loop through each element in the results_np array (each element is a list)
        for result in results_np:
            # Append the current result (which is already a list) to the targets_list
            x = result.replace("'", "")            
            targets_list.append(eval(result))
            # print(result)

        # print(targets_list)

        # Convert the list to a tensor
        targets = torch.tensor(targets_list, dtype=torch.float32)
        # Return as a dictionary
        return {
            "index": index,
            "pos": pos,
            "detailed_pos": detailed_pos,
            "dep": dep,
            "ent_type": ent_type,
            "sent": sent,
            "targets": targets,
        }

    

def collate_fn(batch):
    # Unpack batch dictionaries into individual lists of tensors
    index = [item["index"] for item in batch]
    pos = [item["pos"] for item in batch]
    detailed_pos = [item["detailed_pos"] for item in batch]
    dep = [item["dep"] for item in batch]
    ent_type = [item["ent_type"] for item in batch]
    sent = [item["sent"] for item in batch]
    targets = [item["targets"] for item in batch]

    # print(index)
    # Pad sequences
    indexes_padded = pad_sequence(index, batch_first=True, padding_value=0)
    pos_padded = pad_sequence(pos, batch_first=True, padding_value=0)
    detailed_pos_padded = pad_sequence(detailed_pos, batch_first=True, padding_value=0)
    dep_padded = pad_sequence(dep, batch_first=True, padding_value=0)
    ent_type_padded = pad_sequence(ent_type, batch_first=True, padding_value=0)
    sent_padded = pad_sequence(sent, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=-1)

    return {
        "index": indexes_padded,
        "pos": pos_padded,
        "detailed_pos": detailed_pos_padded,
        "dep": dep_padded,
        "ent_type": ent_type_padded,
        "sent": sent_padded,
        "targets": targets_padded,
    }

# dataset = PosDataset("encoded_dataset_50.csv")
# for i in range(3):  # Check the first 3 examples
#     sample = dataset[i]
#     # print(f"Sample {i}:")
#     print(f"Sent: {sample['sent']}")
#     print(f"Targets: {sample['targets']}")
#     print(f"Targets Length: {len(sample['targets'])}, Sent Length: {len(sample['sent'])}")
#     print("-" * 30)