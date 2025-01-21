# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, confusion_matrix
import mlflow
import mlflow.sklearn
from matplotlib import pyplot as plt

# Define target column
TARGET_COL = "class"

# Define feature columns
FEATURE_COLS = [
    "preg", "plas", "pres", "skin", "test", "mass", "pedi", "age"
]

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")
    parser.add_argument('--criterion', type=str, default='gini',
                        help='The function to measure the quality of a split')
    parser.add_argument('--max_depth', type=int, default=None,
                        help='The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than min_samples_split samples.')

    args = parser.parse_args()

    return args

def main(args):
    '''Read train dataset, train model, save trained model'''

    # Read train data
    train_data = pd.read_parquet(Path(args.train_data))

    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data[FEATURE_COLS]

    # Initialize and train a Decision Tree Classifier
    model = DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth)
    model.fit(X_train, y_train)

    # Log model hyperparameters
    mlflow.log_param("model", "DecisionTreeClassifier")
    mlflow.log_param("criterion", args.criterion)
    mlflow.log_param("max_depth", args.max_depth)

    # Predict using the Decision Tree Model
    yhat_train = model.predict(X_train)

    # Compute and log recall score
    recall = recall_score(y_train, yhat_train)
    print('Recall of Decision Tree classifier on training set: {:.2f}'.format(recall))
    mlflow.log_metric("Recall", float(recall))

    # Create and display a confusion matrix
    cm = confusion_matrix(y_train, yhat_train)
    print(cm)

    # Visualize results
    plt.scatter(y_train, yhat_train, color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.savefig("classification_results.png")
    mlflow.log_artifact("classification_results.png")
    
    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

if __name__ == "__main__":
    
    mlflow.start_run()

    # Parse Arguments
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Model output path: {args.model_output}",
        f"Criterion: {args.criterion}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
