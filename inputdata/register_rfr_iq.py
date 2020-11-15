# Import libraries
import argparse
import joblib
from azureml.core import Workspace, Model, Run

# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, dest='model_folder')
args = parser.parse_args()
model_folder = args.model_folder
print('Model folder',str(model_folder))

# Get the experiment run context
run = Run.get_context()

# load the model
print('Loading model from ' + model_folder)
model_file = model_folder + '/iq_rfr_model.pkl'
model = joblib.load(model_file)

Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = 'iq_rfr_model',
               tags={'Training context':'Pipeline'})

run.complete()
