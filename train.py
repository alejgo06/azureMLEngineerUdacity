# +
from sklearn.linear_model import LogisticRegression
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

# +
#workspace = Workspace(subscription_id, resource_group, workspace_name)

#dataset = Dataset.get_by_name(workspace, name='dataset')
#dataset.to_pandas_dataframe().dropna()
# -
import joblib
from azureml.core.run import Run
run = Run.get_context()



from sklearn import datasets


# TODO: Split data into train and test sets.

# ## YOUR CODE HERE ###a



def clean_data(data):
    # Dict for cleaning data
    months = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    weekdays = {"mon":1, "tue":2, "wed":3, "thu":4, "fri":5, "sat":6, "sun":7}
    
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    jobs = pd.get_dummies(x_df.job, prefix="job")
    x_df.drop("job", inplace=True, axis=1)
    x_df = x_df.join(jobs)
    x_df["marital"] = x_df.marital.apply(lambda s: 1 if s == "married" else 0)
    x_df["default"] = x_df.default.apply(lambda s: 1 if s == "yes" else 0)
    x_df["housing"] = x_df.housing.apply(lambda s: 1 if s == "yes" else 0)
    x_df["loan"] = x_df.loan.apply(lambda s: 1 if s == "yes" else 0)
    contact = pd.get_dummies(x_df.contact, prefix="contact")
    x_df.drop("contact", inplace=True, axis=1)
    x_df = x_df.join(contact)
    education = pd.get_dummies(x_df.education, prefix="education")
    x_df.drop("education", inplace=True, axis=1)
    x_df = x_df.join(education)
    x_df["month"] = x_df.month.map(months)
    x_df["day_of_week"] = x_df.day_of_week.map(weekdays)
    x_df["poutcome"] = x_df.poutcome.apply(lambda s: 1 if s == "success" else 0)
    
    y_df = x_df.pop("y").apply(lambda s: 1 if s == "yes" else 0)
    
    return (x_df,y_df)


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()
    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    #ws = Workspace.from_config()
    #dataset = Dataset.get_by_name(ws, name='bank')
    
    
    
    web_path ="https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"
    #dataset = TabularDatasetFactory().from_delimited_files(web_path)
    dataset = Dataset.Tabular.from_delimited_files(path=web_path)

    x,y=clean_data(dataset)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    
    
    accuracy = model.score(x_test, y_test)
    predictions=model.predict(x_test)

    print('Accuracy of classifier on test set: {:.2f}'.format(accuracy))
    run.log('Accuracy', np.float(accuracy))
    cm = confusion_matrix(y_test, predictions)
    print(cm)

    os.makedirs('outputs', exist_ok=True)
    # files saved in the "outputs" folder are automatically uploaded into run history
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()

# +

    
#subscription_id = '3d1a56d2-7c81-4118-9790-f85d1acf0c77'
#resource_group = 'aml-quickstarts-132460'
#workspace_name = 'quick-starts-ws-132460'
#
#workspace = Workspace(subscription_id, resource_group, workspace_name)
#
#dataset = Dataset.get_by_name(workspace, name='dataset')
#x,y=clean_data(dataset)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#model = LogisticRegression(C=1.0, max_iter=100).fit(x_train, y_train)
# +
#iris = datasets.load_iris()
#x = iris.data
#y = iris.target
#
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
##model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
#model = LogisticRegression().fit(x_train, y_train)
#
#
#accuracy = model.score(x_test, y_test)
#predictions=model.predict(x_test)
#
#print('Accuracy of classifier on test set: {:.2f}'.format(accuracy))
#run.log('Accuracy', np.float(accuracy))
#cm = confusion_matrix(y_test, predictions)
#print(cm)
#
#os.makedirs('outputs', exist_ok=True)
## files saved in the "outputs" folder are automatically uploaded into run history
#joblib.dump(model, 'outputs/model.joblib')
# -





