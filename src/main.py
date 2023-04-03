import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def main():
    """
    Description: This script trains a Gradient Boosting Classifier model on the credit card default dataset.
    Uses MLFlow for logging.
    """

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25) # 25% of the data will be used for testing by default
    parser.add_argument("--n_estimators", required=False, default=100, type=int) # 100 estimators by default
    parser.add_argument("--learning_rate", required=False, default=0.1, type=float) # 0.1 learning rate by default
    parser.add_argument("--registered_model_name", type=str, help="model name")
    args = parser.parse_args()
   
    # MLFlow for logging
    mlflow.start_run()

    # enable autologging through mlflow.sklearn
    mlflow.sklearn.autolog()

    
    # Data preparation
    print("Preparing the data")
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)
    
    credit_df = pd.read_excel(args.data, header=1, index_col=0)

    mlflow.log_metric("num_samples", credit_df.shape[0])
    mlflow.log_metric("num_features", credit_df.shape[1] - 1)

    train_df, test_df = train_test_split(
        credit_df,
        test_size=args.test_train_ratio,
    )
    
    print(f"finished preparing the data. train_df.shape={train_df.shape}, test_df.shape={test_df.shape}")



    ## Training the model
    # Extracting the label column
    y_train = train_df.pop("default payment next month")
    y_test = test_df.pop("default payment next month")

    X_train = train_df.values
    X_test = test_df.values

    # model
    clf = GradientBoostingClassifier(
        n_estimators=args.n_estimators, learning_rate=args.learning_rate
    )
    clf.fit(X_train, y_train)
    print("finished training the model")
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    


    #saving and registering the model
    # Registering the model to the workspace first
    print("Registering the model via MLFlow")
    mlflow.sklearn.log_model(
        sk_model=clf,
        registered_model_name=args.registered_model_name,
        artifact_path=args.registered_model_name,
    )

    # Saving the model to a file
    mlflow.sklearn.save_model(
        sk_model=clf,
        path=os.path.join(args.registered_model_name, "trained_model"),
    )
    
    # Stop Logging
    mlflow.end_run()

if __name__ == "__main__":
    main()