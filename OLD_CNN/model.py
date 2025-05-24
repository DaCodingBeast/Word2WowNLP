import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from api.NLP_Model.FitnessFunction import findFitness
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score

# Load the dataset
df = pd.read_csv("encoded_dataset.csv")

df['results'] = df['results'].apply(lambda x: ast.literal_eval(str(x)) if isinstance(x, str) else x)
print("First few rows of 'results' column:")
print(df['results'].head())  # Check the first few values to see the structure

# Define targets
X = df[["group_id","index","word","pos", "detailed_pos", "dep", "ent_type", "sent"]]  # Features
y = df['results']  # Target variable


# Convert 'y' to binary
mlb = MultiLabelBinarizer()
y_binarized = mlb.fit_transform(y)

# Split into group - that is randomized
gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
train_idx, test_idx = next(gss.split(X, y_binarized, groups=X["group_id"]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y_binarized[train_idx], y_binarized[test_idx]


from FitnessFunction import fitnessScore
fitness_scorer = make_scorer(fitnessScore, needs_proba=False, mlb=mlb)

model = RandomForestClassifier()
scores = cross_val_score(
    model,
    X_train.drop(columns=["group_id", "word"]),  # Drop non-feature columns
    y_train,
    scoring=fitness_scorer,
    cv=5
)
print("Model training completed.")

from api.NLP_Model.OLD.DataProcessor import DataProcessor

processor = DataProcessor(X_test= X_test, Y_test= y_test)
processor.processData()
processor.printData()