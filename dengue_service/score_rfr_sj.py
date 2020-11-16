import json
import joblib
import numpy as np
from azureml.core.model import Model

#Called when the service is loaded
def init():
    global model
    #Get the path to the deployed model file and load it
    model_path=Model.get_model_path('sj_rfr_model')
    model=joblib.load(model_path)

#Called when a request is received
def run(raw_data):
    #Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    data = json.loads(raw_data)['data']
    
    #Get a prediction from the model. This will be a single week's case count
    predictions = (model.predict(data))
    predictions=predictions.tolist()
    
    #Return the predictions as JSON
    return json.dumps(predictions)
