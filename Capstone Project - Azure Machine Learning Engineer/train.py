from sklearn.ensemble import RandomForestRegressor
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from azureml.core import Workspace, Dataset

from azureml.core import Workspace

from azureml.core.dataset import Dataset
import azureml.dataprep as _dprep
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import explained_variance_score

# -

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

# +
#from azureml.core import Workspace, Dataset
#
#subscription_id = '3d1a56d2-7c81-4118-9790-f85d1acf0c77'
#resource_group = 'aml-quickstarts-132460'
#workspace_name = 'quick-starts-ws-132460'





from azureml.core import Workspace

subscription_id = '5a4ab2ba-6c51-4805-8155-58759ad589d8'
resource_group  = 'aml-quickstarts-138280'
workspace_name  = 'quick-starts-ws-138280'

try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    ws.write_config()
    print('Library configuration succeeded')
except:
    print('Workspace not found')

# +
#workspace = Workspace(subscription_id, resource_group, workspace_name)

#dataset = Dataset.get_by_name(workspace, name='dataset')
#dataset.to_pandas_dataframe().dropna()
# -
import joblib
from azureml.core.run import Run




from sklearn import datasets


# TODO: Split data into train and test sets.

# ## YOUR CODE HERE ###a



def clean_data(data):
    x_df = data.to_pandas_dataframe()
    y_df=x_df['y']
    x_df = x_df.drop(columns=["y"])
    return (x_df,y_df)


def main():
    # Add arguments to script
    from azureml.core.run import Run
    run_logger = Run.get_context()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_depth', type=int, default=100, help="max_depth")
    parser.add_argument('--min_samples_leaf', type=int, default=2, help="min_samples_leaf")
    parser.add_argument('--max_features', type=float, default=0.2, help="max_features")
    
    args = parser.parse_args()

    
    run_logger.log("max_depth:", np.int(args.max_depth))
    run_logger.log("min_samples_leaf:", np.int(args.min_samples_leaf))
    run_logger.log("max_features:", np.float(args.max_features))

    
    #dataset = Dataset.get_by_name(ws, name='data')
    
    dataset = ws.datasets['datalite']
    
    
    x,y=clean_data(dataset)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    clf =RandomForestRegressor(max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,max_features=args.max_features, random_state=11)
    
    clf.fit(x_train, y_train)

    #yhat_train=clf.predict(X_train)
    yhat_test=clf.predict(x_test)
    #sqrt(mean_squared_error(y_train, yhat_train))
    RMSE=sqrt(mean_squared_error(y_test, yhat_test))

    
    
    run_logger.log('RMSE', np.float(RMSE))
    
    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(clf, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
