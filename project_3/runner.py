# Import necessary modules
import os  # Provides functions to interact with the operating system (e.g., change directories, run system commands)
import time  # Provides time-related functions (e.g., track elapsed time)
from tqdm import tqdm

import mymain  # A module for showing progress bars in loops

import pandas as pd
import os
from sklearn.metrics import roc_auc_score

def calculate_auc(split_dir):
    test_path = os.path.join(split_dir, "test_y.csv")
    pred_path = os.path.join(split_dir, "mysubmission.csv")

    # Load submission data
    predict_df = pd.read_csv(pred_path)

    # Load true labels data
    true_labels_df = pd.read_csv(test_path)
    # print(true_labels_df.head())

    # Extract true sentiment and probabilities
    true_sentiment = true_labels_df['sentiment']
    predicted_probabilities = predict_df['prob']

    # Calculate AUC score
    return roc_auc_score(true_sentiment, predicted_probabilities)


def main():
    folds = [f"split_{i}" for i in range(1, 6)]  # ['fold_1', 'fold_2', ..., 'fold_5']

    cwd = os.getcwd()

    project_data = os.path.join(cwd, 'F24_Proj3_data')

    for fold in folds:
        fold_data = os.path.join(project_data, fold)
        os.chdir(fold_data)

        start_time = time.time()

        mymain.main()

        # Record the ending time after the script has finished executing
        end_time = time.time()

        # Calculate the total execution time by subtracting the start time from the end time
        execution_time = end_time - start_time

        auc = calculate_auc(fold_data)
        print(f"{fold} auc: {auc:.4f} time: {execution_time:.2f} s")

    os.chdir(cwd)


if __name__ == '__main__':
    main()
