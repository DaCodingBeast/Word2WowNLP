from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

class DataProcessor:

    predictions = []
    answers = []

    def __init__(self, X_test, Y_test, model: RandomForestClassifier):
        self.X_test = X_test
        self.Y_test = Y_test
        self.model = model
    
    def processData(self):
        mlb = MultiLabelBinarizer()

        X_test_groups = self.X_test.groupby("group_id")

        for group_id, group_data in X_test_groups:

            startIndex = group_data[group_data["index"] == 0].index[0]
            # print(startIndex)

            X_test_nouns = group_data[group_data["pos"] == 7]  # Filter for nouns (POS tag 7)
            # print(X_test_nouns)
            # print(group_data)
            # print(y_test)
            if not X_test_nouns.empty:
                # Get the indices of the rows in X_test_nouns and map them to y_test
                test_noun_indices = X_test_nouns.index
                # print(test_noun_indices)
                y_test_group = self.y_test[test_noun_indices-startIndex]  # Use the correct indices to access y_test
                # print(y_test_group)
                y_pred_group = self.model.predict(X_test_nouns.drop(columns=["group_id", "word"]))
                
                # Convert predictions and actual results to lists
                y_pred_lists = [mlb.classes_[row.nonzero()[0]].tolist() for row in y_pred_group]
                y_test_lists = [mlb.classes_[row.nonzero()[0]].tolist() for row in y_test_group]

                # Store predictions and actuals
                self.predictions.extend(zip(X_test_nouns["word"], y_pred_lists))
                self.answers.extend(zip(X_test_nouns["word"], y_test_lists))
    
    def printData(self):
        # Print word relationships for each group
        print("\nWord Relationships (Real vs Predicted):")
        for (word, prediction), (_, actual) in zip(self.redictions, self.actuals):
            print(f"Word: {word}")
            print(f"  Actual Relationships: {actual}")
            print(f"  Predicted Relationships: {prediction}")
            print("-" * 40)

    