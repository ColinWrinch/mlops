# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train data")
    parser.add_argument("--test_data", type=str, help="Path to test data")
    parser.add_argument("--n_estimators", type=int, default=100, help='The number of trees in the forest')
    parser.add_argument("--max_depth", type=str, default=None, help='The maximum depth of the tree')
    parser.add_argument("--model_output", type=str, help="Path of output model")
    args = parser.parse_args()
    return args

def main(args):
    '''Read train and test datasets, train model, evaluate model, save trained model'''
    # Log randomforest input parameters
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth
    }
    mlflow.log_params(params)

    # Load datasets   
    train_df = pd.read_csv(Path(args.train_data)/"train.csv")
    test_df = pd.read_csv(Path(args.test_data)/"test.csv")    
    

    # Dropping the label column and assigning it to y_train
    y_train = train_df["price"].values  # 'price' is the target variable in this case study

    # Dropping the 'price' column from train_df to get the features and converting to array for model training
    X_train = train_df.drop("price", axis=1).values

    # Dropping the label column and assigning it to y_test
    y_test = test_df["price"].values  # 'price' is the target variable for testing

    # Dropping the 'price' column from test_df to get the features and converting to array for model testing
    X_test = test_df.drop("price", axis=1).values
    
    # Convert types for our input parameters as we are sending in strings to handle None if neccessary on either of them
    n_estimators = None
    if args.n_estimators and isinstance(args.n_estimators, str):
        if args.n_estimators != "None":
            n_estimators = int(args.n_estimators)        
    max_depth = None
    if args.max_depth and isinstance(args.max_depth, str):
        if args.max_depth != "None":
            max_depth = int(args.max_depth)
    
    # Initialize and train a Random Forest Regressor
    random_forest_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=99)
    random_forest_model = random_forest_model.fit(X_train, y_train)
    random_forest_predictions = random_forest_model.predict(X_test)

    # Compute and log Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, random_forest_predictions)
    print(f'Mean Squared Error of Random Forest Regressor on test set: {mse:.2f}')
    mlflow.log_metric("MSE", float(mse))

    # Output the trained model
    mlflow.sklearn.save_model(random_forest_model, args.model_output)


if __name__ == "__main__":
    
    mlflow.start_run()
    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Test dataset input path: {args.test_data}",
        f"Model output path: {args.model_output}",
        f"Number of Estimators: {args.n_estimators}",
        f"Max Depth: {args.max_depth}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()

