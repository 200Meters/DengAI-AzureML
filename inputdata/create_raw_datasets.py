
#Import needed libraries
from azureml.core import Dataset

#Create datasets for the raw training and holdout/test file
ws=Workspace.get(name='Azure-ML-WS',subscription_id='fd2d8de8-17e1-4976-9906-fdde487edd5f',resource_group='AzureML-Learning')
default_ds = ws.get_default_datastore()

default_ds.upload_files(files=['inputdata/dengue_features_train.csv'],
                    target_path='dengueAI/inputdata',
                    overwrite=True, 
                    show_progress=True)

#Create a tabular dataset from the path on the datastore for the file
tab_train_ds = Dataset.Tabular.from_delimited_files(path=(default_ds, 'dengueAI/inputdata/dengue_features_train.csv'))

# Register the tabular dataset
try:
    tab_train_ds = tab_train_ds.register(workspace=ws, 
                            name='dengue-features-train-ds',
                            description='Raw dengue feature training data',
                            tags = {'format':'CSV'},
                            create_new_version=True)
    print('Dataset registered.')
except Exception as ex:
    print(ex)

    
default_ds.upload_files(files=['inputdata/dengue_features_test.csv'],
                    target_path='dengueAI/inputdata',
                    overwrite=True, 
                    show_progress=True)

#Create a tabular dataset from the path on the datastore for the file
tab_test_ds = Dataset.Tabular.from_delimited_files(path=(default_ds, 'dengueAI/inputdata/dengue_features_test.csv'))

# Register the tabular dataset
try:
    tab_test_ds = tab_test_ds.register(workspace=ws, 
                            name='dengue-features-test-ds',
                            description='Raw dengue feature test/holdout data',
                            tags = {'format':'CSV'},
                            create_new_version=True)
    print('Dataset registered.')
except Exception as ex:
    print(ex)
