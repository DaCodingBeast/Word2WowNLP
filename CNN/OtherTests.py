import torch
import unittest
import pandas as pd
from io import StringIO
from torch.utils.data import DataLoader
from Model import NounAdjectiveModel
from PosDataset import PosDataset,collate_fn  # Assuming the dataset class is named 'POSDataset'

class TestNounAdjectiveModelWithDataset(unittest.TestCase):
    
    def setUp(self):
        # Prepare a mock CSV file using StringIO
        csv_data = """group_id,index,word,pos,detailed_pos,dep,ent_type,sent,results
0,0,The,5,11,20,18,0,"[0, 0, 0, 0, 0, 0, 0]"
0,1,small,0,16,6,18,0,"[0, 0, 0, 0, 0, 0, 0]"
0,2,brown,0,16,6,18,0,"[0, 0, 0, 0, 0, 0, 0]"
0,3,dog,7,22,29,18,0,"[0, 1, 1, 0, 0, 0, 0]"
0,4,runs,15,42,0,18,0,"[0, 0, 0, 0, 0, 0, 0]"
0,5,quickly,2,30,4,18,0,"[0, 0, 0, 0, 0, 0, 0]"
0,6,.,12,5,41,18,0,"[0, 0, 0, 0, 0, 0, 0]"
1,0,A,5,11,20,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
1,1,tall,0,16,6,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
1,2,tree,7,22,29,18,0,"[0, 1, 0, 0, 0, 0, 0, 0]"
1,3,stood,15,38,0,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
1,4,in,1,15,39,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
1,5,the,5,11,20,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
1,6,park,7,22,35,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
1,7,.,12,5,41,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
2,0,The,5,11,20,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
2,1,beautiful,0,16,6,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
2,2,blue,0,16,6,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
2,3,sky,7,22,29,18,0,"[0, 1, 1, 0, 0, 0, 0, 0]"
2,4,is,3,42,0,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
2,5,clear,0,16,2,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
2,6,today,7,22,28,1,0,"[0, 0, 0, 0, 0, 0, 0, 0]"
2,7,.,12,5,41,18,0,"[0, 0, 0, 0, 0, 0, 0, 0]" """
        # Use StringIO to simulate a file
        self.csv_file = StringIO(csv_data)

        # Convert CSV data to DataFrame
        self.df = pd.read_csv(self.csv_file)

        # Write the DataFrame to a temporary CSV file for the dataset
        self.csv_file_path = 'mock_pos_data.csv'
        self.df.to_csv(self.csv_file_path, index=False)

        # Initialize model parameters
        pos_size = 18            # Example size for POS tag vocabulary
        detailed_pos_size = 51   # Example size for detailed POS tag vocabulary
        dep_size = 46            # Example size for dependency tags
        ent_type_size = 19       # Example size for entity type tags
        embed_dim = 16           # Dimensionality of embeddings
        kernel_size = 3          # Kernel size for convolution layers
        
        # Create an instance of the model
        self.model = NounAdjectiveModel(pos_size, detailed_pos_size, dep_size, ent_type_size, embed_dim, kernel_size)
        
        # Create the dataset and dataloader using the CSV file path
        self.dataset = PosDataset(self.csv_file_path)
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)


    def test_pos_dataset(self):
        # Test the POSDataset class by fetching a sample batch
        batch = next(iter(self.dataloader))
        
        # Check the shape of the batch
        self.assertEqual(batch['pos'].shape, (2, 8))  # Batch size 2, sequence length 8 (based on the longest sentence)
        self.assertEqual(batch['detailed_pos'].shape, (2, 8))
        self.assertEqual(batch['dep'].shape, (2, 8))
        self.assertEqual(batch['ent_type'].shape, (2, 8))
    
    def test_model_with_dataset(self):
        # Test that the model works with the dataset
        batch = next(iter(self.dataloader))
        
        # Extract input features
        x_pos = batch['pos']
        x_detailed_pos = batch['detailed_pos']
        x_dep = batch['dep']
        x_ent_type = batch['ent_type']
        
        # Pass the data through the model
        output = self.model(x_pos, x_detailed_pos, x_dep, x_ent_type)
        
        # Check that the output has the correct shape (batch_size, sentence_length, 1)
        self.assertEqual(output.shape, (2, 8, 1), f"Expected output shape (2, 8, 1), but got {output.shape}")
        
        # Check that the output is in the range [0, 1]
        self.assertTrue((output >= 0).all() and (output <= 1).all(), f"Output values should be between 0 and 1, but found {output}")
    
    def test_single_sentence_forward(self):
        # Test that the model works for a single sentence from the dataset
        batch = next(iter(self.dataloader))
        
        # Extract input features for a single sentence
        x_pos = batch['pos'][0:1]  # Get a single sentence
        x_detailed_pos = batch['detailed_pos'][0:1]
        x_dep = batch['dep'][0:1]
        x_ent_type = batch['ent_type'][0:1]
        
        # Pass the data through the model
        output = self.model(x_pos, x_detailed_pos, x_dep, x_ent_type)
        
        # Check that the output shape is correct for a single sentence
        self.assertEqual(output.shape, (1, 8, 1), f"Expected output shape (1, 8, 1), but got {output.shape}")
        
        # Check that the output is in the range [0, 1]
        self.assertTrue((output >= 0).all() and (output <= 1).all(), f"Output values should be between 0 and 1, but found {output}")
    
    def test_invalid_input(self):
        # Check if the model handles invalid input gracefully (e.g., inputs with mismatched dimensions)
        batch = next(iter(self.dataloader))
        
        # Alter one of the inputs to have an incorrect shape
        x_invalid = torch.randint(0, 10, (2, 9))  # Invalid length (length 9 instead of 8)
        
        with self.assertRaises(Exception):  # Expecting an exception due to shape mismatch
            self.model(batch['pos'], batch['detailed_pos'], batch['dep'], x_invalid)

if __name__ == "__main__":
    unittest.main()
