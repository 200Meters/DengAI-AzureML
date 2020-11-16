
# Import libraries
import argparse
import joblib
from azureml.core import Workspace, Model, Run
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

#Set run context
run=Run.get_context()
ws=run.experiment.workspace

#Get PipelineData arguments for model and deployment file locations
parser = argparse.ArgumentParser()
parser.add_argument('--deploy_folder', type=str, dest='deploy_folder')
args = parser.parse_args()
deploy_folder = args.deploy_folder

#refer to the script and env file locations
script_file=deploy_folder + '/score_rfr_sj.py'
env_file=deploy_folder + '/dengue_env.yml'

#Config the scoring environment
inference_config=InferenceConfig(runtime='python',
                                   entry_script=script_file,
                                   conda_file=env_file)

deployment_config=AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

service_name='dengue-sj-rfr-service'

service=Model.deploy(ws,service_name,[model],inference_config,deployment_config)

service.wait_for_deployment(True)

print(service.state)

