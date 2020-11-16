
import pandas as pd
from sklearn.preprocessing import RobustScaler
from azureml.core import Workspace,Datastore,Dataset,Run
import argparse
import os

#Set run context and workspace
run=Run.get_context()
ws=run.experiment.workspace
default_ds = ws.get_default_datastore()

# Get PipelineData argument
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, dest='folder')
args = parser.parse_args()
output_folder = args.folder

#Define the fields used for each city
sj_features=[
    'year',
    'yearcount',
    'weekofyear',
    'station_max_temp_c',
    'station_min_temp_c',
    'cum_rain_prior_24_wks',
    'avg_max_temp_prior_22_wks',
    'total_cases'
]

sj_lags={
    'year':0,
    'yearcount':0,
    'weekofyear':0,
    'station_max_temp_c':0,
    'station_min_temp_c':0,
    'cum_rain_prior_24_wks':46,
    'avg_max_temp_prior_22_wks':0,
    'total_cases':0
}

iq_features=[
    'year',
    'yearcount',
    'weekofyear',
    'reanalysis_min_air_temp_k',
    'station_max_temp_c',
    'cum_rain_prior_22_wks',
    'total_cases'
]

iq_lags={
    'year':0,
    'yearcount':0,
    'weekofyear':0,
    'reanalysis_min_air_temp_k':0,
    'station_max_temp_c':0,
    'cum_rain_prior_22_wks':43,
    'total_cases':0
}

#Define a function to retrieve the features to be used in the model for each specific city
def get_feature_list(city,lag_names=True):
    if city=='sj':
        feature_list=[]
        if lag_names==True:
            feature_list=sj_features
            for key, value in sj_lags.items():
                for i in range(value): feature_list.append(str(key)+'_shift_'+str(i))
        else:
            for key, value in sj_lags.items(): feature_list.append(str(key))
    elif city=='iq':
        feature_list=[]
        if lag_names==True:
            feature_list=iq_features
            for key, value in iq_lags.items():
                for i in range(value): feature_list.append(str(key)+'_shift_'+str(i))
        else:
            for key, value in iq_lags.items(): feature_list.append(str(key))
                
    return feature_list

#Define a function to create a set of time-lagged features based on the feature and the desired lag
def create_lag_features(df,lag,end_col=0):
    for i in range(lag):
        df_lag=df.iloc[:,:end_col]
        df_lag=df_lag.shift(periods=i)
        df=df.join(df_lag,rsuffix='_shift_'+str(i))
    
    df=df.iloc[lag:,:]
    df.reset_index(inplace=True,drop=True)
    
    return df

#create sets for each city
def prep_for_model(city,lookback):
    #get train and test for sj or iq
    if city=='sj':
        train_all_sj_ds = ws.datasets.get('dengue-train-all-sj-ds')
        holdout_all_sj_ds = ws.datasets.get('dengue-holdout-all-sj-ds')
        df=train_all_sj_ds.to_pandas_dataframe()
        df_h=holdout_all_sj_ds.to_pandas_dataframe()
        df_h['total_cases']=0
    elif city=='iq':
        train_all_iq_ds = ws.datasets.get('dengue-train-all-iq-ds')
        holdout_all_iq_ds = ws.datasets.get('dengue-holdout-all-iq-ds')
        df=train_all_iq_ds.to_pandas_dataframe()
        df_h=holdout_all_iq_ds.to_pandas_dataframe()
        df_h['total_cases']=0
    
    #create single dataset
    df_all=df.append(df_h,ignore_index=True)

    #Get the lists of features to train and reduce the df to those
    training_feature_list=[]
    city_feature_list=get_feature_list(city,lag_names=False)
    for i in range(len(city_feature_list)):training_feature_list.append(city_feature_list[i])
    df_all_lag=df_all[training_feature_list].copy()

    #Create lagged data
    df_all_lag=create_lag_features(df_all_lag,lag=lookback,end_col=df_all_lag.shape[1])

    #Reduce features to just the ones needed for training plus the lagged versions of the features since we need 2d dataset
    training_feature_list=[]
    city_feature_list=get_feature_list(city,lag_names=True)
    for i in range(len(city_feature_list)):training_feature_list.append(city_feature_list[i])
    df_all_lag=df_all_lag[training_feature_list].copy()

    #Break out the label data so it does not get scaled and the drop the values for holdout since they are all 0
    y=df_all_lag['total_cases']
    y=y[:df.shape[0]-lookback]
    df_all_lag.drop(columns=['total_cases'],inplace=True)

    #scale features using desired scaler
    scaler=RobustScaler()
    df_all_lag=scaler.fit_transform(df_all_lag)

    #break out the holdout file from the input file
    np_df=df_all_lag[:df.shape[0]-lookback,:]
    np_df_h=df_all_lag[df.shape[0]-lookback:,:]

    return np_df, np_df_h, y

#Create the datasets for each city and save to intermediate data file for model use
np_sj,np_sj_h,y_sj=prep_for_model(city='sj',lookback=50)
df_sj=pd.DataFrame(np_sj)
df_sj_holdout=pd.DataFrame(np_sj_h)
df_y_sj=pd.DataFrame(y_sj)

# Save prepped data to the PipelineData location for sj
os.makedirs(output_folder, exist_ok=True)
train_sj_output_path = os.path.join(output_folder, 'train_sj_scaled.csv')
df_sj.to_csv(train_sj_output_path,index=False)

test_sj_output_path = os.path.join(output_folder, 'holdout_sj_scaled.csv')
df_sj_holdout.to_csv(test_sj_output_path,index=False)

y_sj_output_path = os.path.join(output_folder, 'y_sj.csv')
df_y_sj.to_csv(y_sj_output_path,index=False)

#Create the datasets for each city and save to intermediate data file for model use
np_iq,np_iq_h,y_iq=prep_for_model(city='iq',lookback=50)
df_iq=pd.DataFrame(np_iq)
df_iq_holdout=pd.DataFrame(np_iq_h)
df_y_iq=pd.DataFrame(y_iq)

# Save prepped data to the PipelineData location for iq
train_iq_output_path = os.path.join(output_folder, 'train_iq_scaled.csv')
df_iq.to_csv(train_iq_output_path,index=False)

test_iq_output_path = os.path.join(output_folder, 'holdout_iq_scaled.csv')
df_iq_holdout.to_csv(test_iq_output_path,index=False)

y_iq_output_path = os.path.join(output_folder, 'y_iq.csv')
df_y_iq.to_csv(y_iq_output_path,index=False)

### Create reusable datasets for the scaled holdout data. These will be needed to make predictions once the models are deployed
default_ds.upload_files(files=[test_sj_output_path],
                    target_path='dengueAI/inputdata',
                    overwrite=True, 
                    show_progress=True)

#Create a tabular dataset from the path on the datastore for the file
tab_test_sj_rfr_ds = Dataset.Tabular.from_delimited_files(path=(default_ds, 'dengueAI/inputdata/holdout_sj_scaled.csv'))


# Register the tabular dataset
try:
    tab_test_sj_rfr_ds = tab_test_sj_rfr_ds.register(workspace=ws, 
                            name='test-sj-rfr-ds',
                            description='Holdout data scaled for SJ RFR model',
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
tab_test_iq_rfr_ds = Dataset.Tabular.from_delimited_files(path=(default_ds, 'dengueAI/inputdata/holdout_iq_scaled.csv'))

# Register the tabular dataset
try:
    tab_test_iq_rfr_ds = tab_test_iq_rfr_ds.register(workspace=ws, 
                            name='test-iq-rfr-ds',
                            description='Holdout data scaled for IQ RFR model',
                            tags = {'format':'CSV'},
                            create_new_version=True)
    print('Dataset registered.')
except Exception as ex:
    print(ex)



run.complete
