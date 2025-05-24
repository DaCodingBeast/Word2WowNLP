import torch
import unittest
from Model import NounAdjectiveModel  # Assuming the model is saved in a file named 'model.py'

class TestNounAdjectiveModel(unittest.TestCase):
    
    def setUp(self):
        # Initialize model parameters
        pos_size = 10            # Example size for POS tag vocabulary
        detailed_pos_size = 10   # Example size for detailed POS tag vocabulary
        dep_size = 10            # Example size for dependency tags
        ent_type_size = 10       # Example size for entity type tags
        embed_dim = 16           # Dimensionality of embeddings
        kernel_size = 3          # Kernel size for convolution layers
        
        # Create an instance of the model
        self.model = NounAdjectiveModel(pos_size, detailed_pos_size, dep_size, ent_type_size, embed_dim, kernel_size)
    
    def test_forward_pass(self):
        # Create a batch of input tokens with random values
        batch_size = 2
        sentence_length = 5
        
        x_pos = torch.randint(0, 10, (batch_size, sentence_length))
        x_detailed_pos = torch.randint(0, 10, (batch_size, sentence_length))
        x_dep = torch.randint(0, 10, (batch_size, sentence_length))
        x_ent_type = torch.randint(0, 10, (batch_size, sentence_length))
        
        # Pass the inputs through the model
        output = self.model(x_pos, x_detailed_pos, x_dep, x_ent_type)
        
        # Check that the output has the correct shape (batch_size, sentence_length, 1)
        self.assertEqual(output.shape, (batch_size, sentence_length, 1), f"Expected output shape (batch_size, sentence_length, 1), but got {output.shape}")
    
    def test_output_range(self):
        # Create a batch of input tokens with random values
        batch_size = 2
        sentence_length = 5
        
        x_pos = torch.randint(0, 10, (batch_size, sentence_length))
        x_detailed_pos = torch.randint(0, 10, (batch_size, sentence_length))
        x_dep = torch.randint(0, 10, (batch_size, sentence_length))
        x_ent_type = torch.randint(0, 10, (batch_size, sentence_length))
        
        # Pass the inputs through the model
        output = self.model(x_pos, x_detailed_pos, x_dep, x_ent_type)
        
        # Check that the output values are between 0 and 1 (valid probability)
        self.assertTrue((output >= 0).all() and (output <= 1).all(), f"Output values should be between 0 and 1, but found {output}")
    
    def test_embedding_layer_shapes(self):
        # Create a batch of input tokens with random values
        batch_size = 2
        sentence_length = 5
        
        x_pos = torch.randint(0, 10, (batch_size, sentence_length))
        x_detailed_pos = torch.randint(0, 10, (batch_size, sentence_length))
        x_dep = torch.randint(0, 10, (batch_size, sentence_length))
        x_ent_type = torch.randint(0, 10, (batch_size, sentence_length))
        
        # Test embedding shapes
        pos_embed = self.model.pos_embedding(x_pos)
        detailed_pos_embed = self.model.detailed_pos_embedding(x_detailed_pos)
        dep_embed = self.model.dep_embedding(x_dep)
        ent_type_embed = self.model.ent_type_embedding(x_ent_type)
        
        # Check that the embeddings have the correct shape
        self.assertEqual(pos_embed.shape, (batch_size, sentence_length, 16), f"Expected pos embedding shape (batch_size, sentence_length, 16), but got {pos_embed.shape}")
        self.assertEqual(detailed_pos_embed.shape, (batch_size, sentence_length, 16), f"Expected detailed_pos embedding shape (batch_size, sentence_length, 16), but got {detailed_pos_embed.shape}")
        self.assertEqual(dep_embed.shape, (batch_size, sentence_length, 16), f"Expected dep embedding shape (batch_size, sentence_length, 16), but got {dep_embed.shape}")
        self.assertEqual(ent_type_embed.shape, (batch_size, sentence_length, 16), f"Expected ent_type embedding shape (batch_size, sentence_length, 16), but got {ent_type_embed.shape}")
    
    def test_invalid_input(self):
        # Check if the model handles invalid input gracefully (e.g., inputs with mismatched dimensions)
        batch_size = 2
        sentence_length = 5
        
        x_pos = torch.randint(0, 10, (batch_size, sentence_length))
        x_detailed_pos = torch.randint(0, 10, (batch_size, sentence_length))
        x_dep = torch.randint(0, 10, (batch_size, sentence_length))
        x_ent_type = torch.randint(0, 10, (batch_size, sentence_length))
        
        # Alter one of the inputs to have an incorrect shape
        x_invalid = torch.randint(0, 10, (batch_size, sentence_length + 1))  # Invalid length
        
        with self.assertRaises(Exception):  # Expecting an exception due to shape mismatch
            self.model(x_pos, x_detailed_pos, x_dep, x_invalid)
    
    def test_single_token_output(self):
        # Check the output when processing a single token
        batch_size = 1
        sentence_length = 1
        
        x_pos = torch.randint(0, 10, (batch_size, sentence_length))
        x_detailed_pos = torch.randint(0, 10, (batch_size, sentence_length))
        x_dep = torch.randint(0, 10, (batch_size, sentence_length))
        x_ent_type = torch.randint(0, 10, (batch_size, sentence_length))
        
        # Pass the single token through the model
        output = self.model(x_pos, x_detailed_pos, x_dep, x_ent_type)
        
        # Check that the output is a single value between 0 and 1
        self.assertEqual(output.shape, (batch_size, sentence_length, 1), f"Expected output shape (1, 1, 1), but got {output.shape}")
        self.assertTrue((output >= 0).all() and (output <= 1).all(), f"Output values should be between 0 and 1, but found {output}")

if __name__ == "__main__":
    unittest.main()
