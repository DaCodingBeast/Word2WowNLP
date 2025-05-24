from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss
import numpy as np
def findFitness(filtered_predictions, filtered_actuals, mlb: MultiLabelBinarizer):

    y_true = [labels for _, labels in filtered_actuals]
    y_pred = [labels for _, labels in filtered_predictions]

    print("Original y_true:", y_true)
    print("Original y_pred:", y_pred)

    # Replace individual empty lists with [-1]
    y_true_filled = [labels if isinstance(labels, list) and len(labels) > 0 else [-1] for labels in y_true]
    y_pred_filled = [labels if isinstance(labels, list) and len(labels) > 0 else [-1] for labels in y_pred]

    print("Filled y_true:", y_true_filled)
    print("Filled y_pred:", y_pred_filled)

    # Fit to all possible labels
    mlb.fit(y_true_filled + y_pred_filled)

    # Transform to binary
    y_true_binarized = mlb.transform(y_true_filled)
    y_pred_binarized = mlb.transform(y_pred_filled)

    # Check for -1 in the predictions and ground truth
    x = any(-1 in sublist for sublist in y_pred_filled)
    y = any(-1 in sublist for sublist in y_true_filled)
    boolean = (x or y) and not (x and y)

    # Apply penalty by increasing the Hamming loss where -1 is present
    hamming = hamming_loss(y_true_binarized, y_pred_binarized)

    # Adding extra penalty for -1 values
    additional_penalty = int(boolean) * 3  # `boolean` should be a scalar boolean, convert to int for penalty

    anotherPenalty = 0
    if not x and not y:
        anotherPenalty -= .1
        hamming *= .8

    # Total hamming loss with the custom penalty
    return hamming + additional_penalty + anotherPenalty


# Define the custom fitness scoring function
def fitnessScore(y_true, y_pred, mlb: MultiLabelBinarizer):
    """
    Calculate the fitness score based on the model's predictions and the actual labels.
    The `findFitness` function will be used for computing the score.
    
    Parameters:
    - y_true: Ground truth (binarized format).
    - y_pred: Predictions from the model (binarized format).
    - mlb: MultiLabelBinarizer instance to transform predictions back to class labels.

    Returns:
    - fitness_score: The fitness score computed using `findFitness`.
    """
    # Convert y_true and y_pred back to class labels
    y_true_labels = [mlb.classes_[row.nonzero()[0]].tolist() for row in y_true]
    y_pred_labels = [mlb.classes_[row.nonzero()[0]].tolist() for row in y_pred]

    # Compute the fitness score
    predictions = [(None, pred) for pred in y_pred_labels]  # Placeholder for "word"
    actuals = [(None, true) for true in y_true_labels]      # Placeholder for "word"

    return findFitness(predictions, actuals, mlb)