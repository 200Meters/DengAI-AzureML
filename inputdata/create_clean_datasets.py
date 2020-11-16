
from azureml.core import Workspace,Datastore,Dataset,Run
import pandas as pd
import numpy as np
import argparse
import os

#Set run context
run=Run.get_context()

# Get PipelineData argument
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, dest='folder')
args = parser.parse_args()
output_folder = args.folder

#Get the datasets from the input params
df_i=run.input_datasets['dengue_train'].to_pandas_dataframe()
df_h=run.input_datasets['dengue_test'].to_pandas_dataframe()

#interpolate missing values
df_i.interpolate(inplace=True,method='linear',limit_direction='forward')
df_h.interpolate(inplace=True,method='linear',limit_direction='forward')

#Add a total cases column to the holdout set
df_h['total_cases']=0

#split by city
df_sj=df_i[df_i['city']=='sj']
df_iq=df_i[df_i['city']=='iq']

df_sj_h=df_h[df_h['city']=='sj']
df_iq_h=df_h[df_h['city']=='iq']

len_sj=len(df_sj)
len_sj_h=len(df_sj_h)

#concat training and holdout data
df_sj=df_sj.append(df_sj_h)

#Get cumulative totals at various intervals - past 4 to past 25 weeks
for i in range(2,25):
    df_sj['cum_rain_prior_'+str(i)+'_wks']=df_sj['precipitation_amt_mm'].rolling(i).sum()

for i in range(2,25):
    df_sj['avg_min_temp_prior_'+str(i)+'_wks']=df_sj['station_min_temp_c'].rolling(i).mean()
    
for i in range(2,25):
    df_sj['avg_max_temp_prior_'+str(i)+'_wks']=df_sj['station_max_temp_c'].rolling(i).mean()
    
for i in range(2,25):
    df_sj['avg_specific_humidity_prior_'+str(i)+'_wks']=df_sj['reanalysis_specific_humidity_g_per_kg'].rolling(i).mean()
    
for i in range(2,25):
    df_sj['avg_relative_humidity_prior_'+str(i)+'_wks']=df_sj['reanalysis_relative_humidity_percent'].rolling(i).mean()

#split the files back apart
df_sj_h=df_sj.iloc[len_sj:]
df_sj=df_sj.iloc[:len_sj]

#remove the first 25 rows of the input file because of the rolling numbers and the nan's they create
df_sj=df_sj.iloc[25:,:]

#Begin IQ
#Break out source file and holdout file by city
len_iq=len(df_iq)
len_iq_h=len(df_iq_h)

#concat training and holdout data
df_iq=df_iq.append(df_iq_h)

#Get cumulative rainfall totals at various accumulations - past 4 to past 25 weeks
for i in range(2,25):
    df_iq['cum_rain_prior_'+str(i)+'_wks']=df_iq['precipitation_amt_mm'].rolling(i).sum()

for i in range(2,25):
    df_iq['avg_min_temp_prior_'+str(i)+'_wks']=df_iq['station_min_temp_c'].rolling(i).mean()
    
for i in range(2,25):
    df_iq['avg_max_temp_prior_'+str(i)+'_wks']=df_iq['station_max_temp_c'].rolling(i).mean()
    
for i in range(2,25):
    df_iq['avg_specific_humidity_prior_'+str(i)+'_wks']=df_iq['reanalysis_specific_humidity_g_per_kg'].rolling(i).mean()
    
for i in range(2,25):
    df_iq['avg_relative_humidity_prior_'+str(i)+'_wks']=df_iq['reanalysis_relative_humidity_percent'].rolling(i).mean()

#split the files back apart
df_iq_h=df_iq.iloc[len_iq:]
df_iq=df_iq.iloc[:len_iq]

#remove the first 25 rows of the input file because of the rolling numbers
df_iq=df_iq.iloc[25:,:]

#reconstructe df's
df_all=df_sj.append(df_iq)
df_holdout=df_sj_h.append(df_iq_h)
df_holdout.drop(columns=['total_cases'],inplace=True)

# Save prepped data to the PipelineData location
os.makedirs(output_folder, exist_ok=True)
train_output_path = os.path.join(output_folder, 'dengue_train_all.csv')
df_all.to_csv(train_output_path,index=False)
test_output_path = os.path.join(output_folder, 'dengue_holdout_all.csv')
df_holdout.to_csv(test_output_path,index=False)

##Create datasets for the raw training and holdout/test file
#Get the default data store
ws=run.experiment.workspace
default_ds = ws.get_default_datastore()

default_ds.upload_files(files=[train_output_path],
                    target_path='dengueAI/inputdata',
                    overwrite=True, 
                    show_progress=True)

#Create a tabular dataset from the path on the datastore for the file
tab_train_all_ds = Dataset.Tabular.from_delimited_files(path=(default_ds, 'dengueAI/inputdata/dengue_train_all.csv'))


# Register the tabular dataset
try:
    tab_train_all_ds = tab_train_all_ds.register(workspace=ws, 
                            name='dengue-features-train-all-ds',
                            description='Cleaned dengue feature training data',
                            tags = {'format':'CSV'},
                            create_new_version=True)
    print('Dataset registered.')
except Exception as ex:
    print(ex)

    

default_ds.upload_files(files=[test_output_path],
                    target_path='dengueAI/inputdata',
                    overwrite=True, 
                    show_progress=True)

#Create a tabular dataset from the path on the datastore for the file
tab_holdout_all_ds = Dataset.Tabular.from_delimited_files(path=(default_ds, 'dengueAI/inputdata/dengue_holdout_all.csv'))

# Register the tabular dataset
try:
    tab_holdout_all_ds = tab_holdout_all_ds.register(workspace=ws, 
                            name='dengue-features-holdout-all-ds',
                            description='Cleaned dengue feature test/holdout data',
                            tags = {'format':'CSV'},
                            create_new_version=True)
    print('Dataset registered.')
except Exception as ex:
    print(ex)

run.complete()
