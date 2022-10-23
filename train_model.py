import os
import git
import mlflow
import logging
import argparse
import pandas as pd
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

logging.debug(f'Start data processing')
parser = argparse.ArgumentParser(description='Pass data parameters to process')
parser.add_argument('--tracking_uri', required=True)
parser.add_argument('--tracking_username', required=True)
parser.add_argument('--tracking_password', required=True)
parser.add_argument('--registry_model_name', required=True)


mlflow.sklearn.autolog(log_models=True,
                       log_input_examples=True,
                       log_model_signatures=True)

args = parser.parse_args()

MLFLOW_TRACKING_URI = args.tracking_uri
MLFLOW_TRACKING_USERNAME = args.tracking_username
MLFLOW_TRACKING_PASSWORD = args.tracking_password
os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

clf = RandomForestClassifier(random_state=42,
                             verbose=1,
                             n_estimators=30)

pipeline = Pipeline(steps=[
    ('StandardScaler', scaler),
    ('RandomForest', clf)
])

with mlflow.start_run(run_name='RandomForestPipeline') as run:
    pipeline.fit(X_train, y_train)
    mlflow.sklearn.eval_and_log_metrics(pipeline,
                                        X_test,
                                        y_test,
                                        prefix="test_")

artifact_path = "model"
model_name = args.registry_model_name
model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run.info.run_id,
                                                    artifact_path=artifact_path)
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)