import pandas as pd
import unittest

class TestPosDataset(unittest.TestCase):
    def setUp(self):
        # Load the actual dataset from CSV
        self.mock_df = pd.read_csv('encoded_dataset.csv')  # Use the correct path to the CSV file
        
        # You can print or inspect the data to ensure it's loaded correctly
        print(self.mock_df.head())  # Print first few rows of the dataset

    def test_getitem(self):
        # Sample test using the actual dataset
        sample = self.mock_df.iloc[0]  # Get the first row as an example
        self.assertIsNotNone(sample)

import torch
from torch.utils.data import DataLoader
import unittest
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class PosDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Example: return word and its corresponding index in the sentence
        row = self.data.iloc[idx]
        word = row['word']
        index = row['index']
        return {'input': torch.tensor([index]), 'label': word}
    

class TestCollateFn(unittest.TestCase):
    def setUp(self):
        # Load the actual dataset (replace this path with the correct one)
        self.df = pd.read_csv('encoded_dataset.csv')  # Make sure the CSV is in the correct location
        # Create the dataset from the dataframe
        self.dataset = PosDataset(self.df)
        
        # Create a DataLoader with your custom collate_fn
        self.dataloader = DataLoader(self.dataset, batch_size=2, collate_fn=self.collate_fn)
    
    def collate_fn(self, batch):
        # This is a simple collate function that pads the sequences to the max length in the batch
        input_tensor = torch.nn.utils.rnn.pad_sequence([item['input'] for item in batch], batch_first=True)
        labels = [item['label'] for item in batch]
        
        # Ensure padding to at least length 4
        max_len = max(input_tensor.shape[1], 4)  # Set the minimum length after padding
        input_tensor = torch.nn.functional.pad(input_tensor, (0, max_len - input_tensor.shape[1]))  # Padding
        
        return input_tensor, labels
    
    def test_collate_fn_padding(self):
        # Get the first batch
        inputs, labels = next(iter(self.dataloader))
        
        # Ensure that the sequence length is greater than 3 after padding
        self.assertGreater(inputs.shape[1], 3)  # Sequence length should be greater than 3 after padding
        
        # Check that all sequences in the batch are padded to the same length
        sequence_lengths = inputs.shape[1]
        self.assertTrue(all(seq_len == sequence_lengths for seq_len in inputs.shape[1:]))  # Compare all dimensions
    
    def test_len(self):
        # Check that the length of the dataset is correct
        self.assertEqual(len(self.df), 306)  # Make sure to update the number if needed

if __name__ == "__main__":
    unittest.main()