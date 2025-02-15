# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import os
import argparse
import logging
import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to input data")
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--train_data", type=str, help="Path to save train data")
    parser.add_argument("--test_data", type=str, help="Path to save test data")
    args = parser.parse_args()
    return args

def main(args):  # Write the function name for the main data preparation logic
    '''Read, preprocess, split, and save datasets'''
    try:
        # Reading Data
        df = pd.read_csv(args.raw_data)
        logger.info("Data read successfully")

        # Encode our categorical features Segment 
        labelEncoder = LabelEncoder()
        df['Segment'] = labelEncoder.fit_transform(df['Segment'])
        logger.info("Encoding categorical feature Segment, encoded successfully")

        # Split our data into training and testing datasets
        train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=99)
        logger.info("Data split into training and testing datasets")

        # Save our split data into the file system
        os.makedirs(args.train_data, exist_ok=True)
        os.makedirs(args.test_data, exist_ok=True)
        train_df.to_csv(os.path.join(args.train_data, "used_cars.csv"), index=False)
        test_df.to_csv(os.path.join(args.test_data, "used_cars.csv"), index=False)
        logger.info(f"Training and test data saved successfully! training data path {args.train_data} testing data path {args.test_data}")

        # Configure ML flow log metrics
        mlflow.log_metric("train_size", len(train_df))
        mlflow.log_metric("test_size", len(test_df))
        logger.info(f"Training data size: {len(train_df)}, Testing data size: {len(test_df)}") 
    except Exception as e:
        logger.exception("Error in data preparation: {e}", exc_info=True)
        raise e 


if __name__ == "__main__":
    
    logger.info("Starting data preparation and ML flow")
    mlflow.start_run()    
    args = parse_args()    
    logger.info(f"Arguments parsed: {args}")

    lines = [
        f"data path: {args.raw_data}",  # Print the data path
        f"Train dataset output path: {args.train_data}",  # Print the train_data path
        f"Test dataset path: {args.test_data}",  # Print the test_data path
        f"Test-train ratio: {args.test_train_ratio}",  # Print the test_train_ratio
    ]

    for line in lines:
        print(line)
    
    main(args)

    mlflow.end_run()
    logger.info("Data preparation and ML flow completed")
