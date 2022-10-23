import os
import git
import logging
import argparse
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logging.debug(f'Start data processing')
parser = argparse.ArgumentParser(description='Pass data parameters to process')
parser.add_argument('--base_data_url', required=True)
parser.add_argument('--data_file_name', required=True)
parser.add_argument('--local_data_path', required=True)
parser.add_argument('--test_size', required=True, type=float)
parser.add_argument('--random_state', required=True, type=int)
parser.add_argument('--train_variables_file_name', required=True)
parser.add_argument('--train_response_file_name', required=True)
parser.add_argument('--test_variables_file_name', required=True)
parser.add_argument('--test_response_file_name', required=True)
parser.add_argument('--file_encoding', required=True)
parser.add_argument('--file_separator', required=True)
args = parser.parse_args()

logging.debug(f'Start to clone repository {args.base_data_url}')
git.Repo.clone_from(args.base_data_url,
                    args.local_data_path)
logging.debug('Repository cloned')

logging.debug(f'Reading file {args.local_data_path} in {args.local_data_path} folder')

data = pd.read_csv(os.path.join(os.getcwd(),
                                args.local_data_path,
                                args.data_file_name),
                   sep=',',
                   encoding='utf-8')

logging.debug('Starting feature selection')
features_to_remove = data.columns[7:]
X = data.drop(features_to_remove, axis=1)
y = data["fetal_health"]

columns_names = list(X.columns)
scaler = preprocessing.StandardScaler()
X_df = scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=columns_names)

X_train, X_test, y_train, y_test = train_test_split(X_df,
                                                    y,
                                                    test_size=args.test_size,
                                                    random_state=args.random_state)
X_train.to_csv(os.path.join(args.local_data_path,
                            args.train_variables_file_name),
               sep=args.file_separator,
               encoding=args.file_encoding)

X_test.to_csv(os.path.join(args.local_data_path,
                           args.test_variables_file_name),
              sep=args.file_separator,
              encoding=args.file_encoding)

y_train.to_csv(os.path.join(args.local_data_path,
                            args.train_response_file_name),
               sep=args.file_separator,
               encoding=args.file_encoding)

y_test.to_csv(os.path.join(args.local_data_path,
                           args.test_response_file_name),
              sep=args.file_separator,
              encoding=args.file_encoding)