
from azureml.core import Workspace,Datastore,Dataset,Run
import pandas as pd
import numpy as np


#Get the raw file datasets
#ws=Workspace.get(name='Azure-ML-WS',subscription_id='fd2d8de8-17e1-4976-9906-fdde487edd5f',resource_group='AzureML-Learning')
#ds_a=Dataset.get_by_name(ws,'dengue-features-train-all-ds',version='latest')
#ds_h=Dataset.get_by_name(ws,'dengue-features-holdout-all-ds',version='latest')

#Create dataframes for each dataset, train and holdout
#df_a=ds_a.to_pandas_dataframe()
#df_h=ds_h.to_pandas_dataframe()

#Set run context
run=Run.get_context()

# Get PipelineData argument
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, dest='folder')
args = parser.parse_args()
output_folder = args.folder

#create a dataframe for each dataset, train and holdout
df_a=pd.read_csv(output_folder+'/dengue_train_all.csv')
df_h=pd.read_csv(output_folder+'/dengue_holdout_all.csv')

#create a diferrent df for each city
df_sj=df_a[df_a['city']=='sj']
df_sj_h=df_h[df_h['city']=='sj']
df_iq=df_a[df_a['city']=='iq']
df_iq_h=df_h[df_h['city']=='iq']

#Create files for each to persist them to local storage
#df_sj.to_csv('inputdata/train_all_sj.csv',index=False)
#df_sj_h.to_csv('inputdata/holdout_all_sj.csv',index=False)
#df_iq.to_csv('inputdata/train_all_iq.csv',index=False)
#df_iq_h.to_csv('inputdata/holdout_all_iq.csv',index=False)

# Save prepped data to the PipelineData location
os.makedirs(output_folder, exist_ok=True)
train_sj_output_path = os.path.join(output_folder, 'train_all_sj.csv')
df_sj.to_csv(train_sj_output_path,index=False)
test_sj_output_path = os.path.join(output_folder, 'holdout_all_sj.csv')
df_sj_h.to_csv(test_sj_output_path,index=False)

train_iq_output_path = os.path.join(output_folder, 'train_all_iq.csv')
df_iq.to_csv(train_iq_output_path,index=False)
test_iq_output_path = os.path.join(output_folder, 'holdout_all_iq.csv')
df_iq_h.to_csv(test_iq_output_path,index=False)

#upload and create datasets
#Get the default data store
ws=run.experiment.workspace
default_ds = ws.get_default_datastore()

default_ds.upload_files(files=[train_sj_output_path],
                        target_path='dengueAI/inputdata',
                        overwrite=True, 
                        show_progress=True)

#Create a tabular dataset from the path on the datastore for the file
tab_train_all_sj_ds = Dataset.Tabular.from_delimited_files(path=(default_ds, 'dengueAI/inputdata/train_all_sj.csv'))

# Register the tabular dataset
try:
    tab_train_all_sj_ds = tab_train_all_sj_ds.register(workspace=ws, 
                            name='dengue-train-all-sj-ds',
                            description='Lagged feature training data for sj',
                            tags = {'format':'CSV'},
                            create_new_version=True)
    print('Dataset registered.')
except Exception as ex:
    print(ex)

    
default_ds.upload_files(files=[test_sj_output_path],
                    target_path='dengueAI/inputdata',
                    overwrite=True, 
                    show_progress=True)

#Create a tabular dataset from the path on the datastore for the file
tab_holdout_all_sj_ds = Dataset.Tabular.from_delimited_files(path=(default_ds, 'dengueAI/inputdata/holdout_all_sj.csv'))

# Register the tabular dataset
try:
    tab_holdout_all_sj_ds = tab_holdout_all_sj_ds.register(workspace=ws, 
                            name='dengue-holdout-all-sj-ds',
                            description='Lagged dengue feature test/holdout data for sj',
                            tags = {'format':'CSV'},
                            create_new_version=True)
    print('Dataset registered.')
except Exception as ex:
    print(ex)
    


default_ds.upload_files(files=[train_iq_output_path],
                    target_path='dengueAI/inputdata',
                    overwrite=True, 
                    show_progress=True)

#Create a tabular dataset from the path on the datastore for the file
tab_train_all_iq_ds = Dataset.Tabular.from_delimited_files(path=(default_ds, 'dengueAI/inputdata/train_all_iq.csv'))

# Register the tabular dataset
try:
    tab_train_all_iq_ds = tab_train_all_iq_ds.register(workspace=ws, 
                            name='dengue-train-all-iq-ds',
                            description='Lagged feature training data for iq',
                            tags = {'format':'CSV'},
                            create_new_version=True)
    print('Dataset registered.')
except Exception as ex:
    print(ex)

    

default_ds.upload_files(files=[test_iq_output_path],
                    target_path='dengueAI/inputdata',
                    overwrite=True, 
                    show_progress=True)

#Create a tabular dataset from the path on the datastore for the file
tab_holdout_all_iq_ds = Dataset.Tabular.from_delimited_files(path=(default_ds, 'dengueAI/inputdata/holdout_all_iq.csv'))

# Register the tabular dataset
try:
    tab_holdout_all_sj_ds = tab_holdout_all_sj_ds.register(workspace=ws, 
                            name='dengue-holdout-all-iq-ds',
                            description='Lagged dengue feature test/holdout data for iq',
                            tags = {'format':'CSV'},
                            create_new_version=True)
    print('Dataset registered.')
except Exception as ex:
    print(ex)
