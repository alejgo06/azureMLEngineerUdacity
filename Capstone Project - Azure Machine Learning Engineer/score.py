import json
import pandas as pd
import os
import joblib, pickle
from azureml.core import Model
from sklearn.externals import joblib
import azureml.train.automl

def init():
    global daone
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'bestautomlmodel.pkl')
    daone = joblib.load(model_path)

def run(data):
    try:
        data = json.loads(data)['data']
        data = pd.DataFrame.from_dict(data)
        result = daone.predict(data)
        
    
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
