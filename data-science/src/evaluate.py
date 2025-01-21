# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Evaluates trained ML model using test dataset.
Saves predictions, evaluation results and deploy flag.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import recall_score, confusion_matrix

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Define target column
TARGET_COL = "class"

# Define feature columns
FEATURE_COLS = [
    "preg", "plas", "pres", "skin", "test", "mass", "pedi", "age"
]

def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("predict")
    parser.add_argument("--model_name", type=str, help="Name of registered model")
    parser.add_argument("--model_input", type=str, help="Path of input model")
    parser.add_argument("--test_data", type=str, help="Path to test dataset")
    parser.add_argument("--evaluation_output", type=str, help="Path of eval results")
    parser.add_argument("--runner", type=str, help="Local or Cloud Runner", default="CloudRunner")

    args = parser.parse_args()

    return args

def main(args):
    '''Read trained model and test dataset, evaluate model and save result'''

    # Load the test data
    test_data = pd.read_parquet(Path(args.test_data))

    # Split the data into inputs and outputs
    y_test = test_data[TARGET_COL]
    X_test = test_data[FEATURE_COLS]

    # Load the model from input port
    model = mlflow.sklearn.load_model(args.model_input) 

    # ---------------- Model Evaluation ---------------- #
    yhat_test, score = model_evaluation(X_test, y_test, model, args.evaluation_output)

    # ----------------- Model Promotion ---------------- #
    if args.runner == "CloudRunner":
        predictions, deploy_flag = model_promotion(args.model_name, args.evaluation_output, X_test, y_test, yhat_test, score)


def model_evaluation(X_test, y_test, model, evaluation_output):

    # Get predictions
    yhat_test = model.predict(X_test)

    # Save the output data with feature columns, predicted label, and actual label in csv file
    output_data = X_test.copy()
    output_data["real_label"] = y_test
    output_data["predicted_label"] = yhat_test
    output_data.to_csv((Path(evaluation_output) / "predictions.csv"))

    # Evaluate model performance with the test set
    recall = recall_score(y_test, yhat_test)
    print(f'Recall score: {recall:.2f}')
    
    # Print score report to a text file
    (Path(evaluation_output) / "score.txt").write_text(
        f"Scored with the following model:\n{format(model)}"
    )
    with open((Path(evaluation_output) / "score.txt"), "a") as outfile:
        outfile.write(f"Recall score: {recall:.2f}\n")

    mlflow.log_metric("test recall", recall)

    # Visualize results
    plt.scatter(y_test, yhat_test,  color='black')
    plt.plot(y_test, y_test, color='blue', linewidth=3)
    plt.xlabel("Actual value")
    plt.ylabel("Predicted value")
    plt.title("Comparing Model Predictions to Actual values - Test Data")
    plt.savefig("classification_predictions.png")
    mlflow.log_artifact("classification_predictions.png")

    return yhat_test, recall

def model_promotion(model_name, evaluation_output, X_test, y_test, yhat_test, score):
    scores = {}
    predictions = {}

    client = MlflowClient()

    for model_run in client.search_model_versions(f"name='{model_name}'"):
        model_version = model_run.version
        mdl = mlflow.pyfunc.load_model(
            model_uri=f"models:/{model_name}/{model_version}")
        predictions[f"{model_name}:{model_version}"] = mdl.predict(X_test)
        scores[f"{model_name}:{model_version}"] = recall_score(
            y_test, predictions[f"{model_name}:{model_version}"])

    if scores:
        if score >= max(list(scores.values())):
            deploy_flag = 1
        else:
            deploy_flag = 0
    else:
        deploy_flag = 1
    print(f"Deploy flag: {deploy_flag}")

    with open((Path(evaluation_output) / "deploy_flag"), 'w') as outfile:
        outfile.write(f"{int(deploy_flag)}")

    # Add current model score and predictions
    scores["current model"] = score
    predictions["current model"] = yhat_test

    perf_comparison_plot = pd.DataFrame(
        scores, index=["recall score"]).plot(kind='bar', figsize=(15, 10))
    perf_comparison_plot.figure.savefig("perf_comparison.png")
    perf_comparison_plot.figure.savefig(Path(evaluation_output) / "perf_comparison.png")

    mlflow.log_metric("deploy flag", bool(deploy_flag))
    mlflow.log_artifact("perf_comparison.png")

    return predictions, deploy_flag

if __name__ == "__main__":

    mlflow.start_run()

    args = parse_args()

    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_input}",
        f"Test data path: {args.test_data}",
        f"Evaluation output path: {args.evaluation_output}",
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()
