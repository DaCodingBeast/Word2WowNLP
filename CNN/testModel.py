from DatasetCreator import create_dataset, export_to_csv

# ChatGPT determines results
labels = {
    0: {  # Sentence 0
        4: [3],  # "cat" (index 4) is affected by "blue" (index 3)
    },
    1: {  # Sentence 1
        3: [1, 2],  # "dog" (index 3) is affected by "happy" (index 1) and "brown" (index 2)
    },
    2: {  # Sentence 2
        2: [1],  # "tree" (index 2) is affected by "tall" (index 1)
    },
    3: {  # Sentence 3
        5: [1, 4],  # "house" (index 5) is affected by "red" (index 1) and "beautiful" (index 4)
    },
    4: {  # Sentence 4
        3: [1, 2],  # "car" (index 3) is affected by "fast" (index 1) and "sleek" (index 2)
    },
}

# Create dataset from sentences
dataset = create_dataset([
   "There is a blue cat.",
   "A happy brown dog plays.",
   "The tall tree sways.",
   "A red roofed beautiful house stands.",
   "The fast sleek car zooms past.",
], labels)

# Export dataset to CSV
export_to_csv(dataset, "encoded_dataset_50.csv")
