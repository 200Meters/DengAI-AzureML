#Import libraries
from azureml.core import Run
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

#Set run context
run=Run.get_context()

# Get PipelineData argument
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, dest='folder')
args = parser.parse_args()
output_folder = args.folder

#create a dataframe for each dataset, train and holdout
df_sj=pd.read_csv(output_folder+'/train_sj_scaled.csv')
df_sj_h=pd.read_csv(output_folder+'/holdout_sj_scaled.csv')
df_sj_y=pd.read_csv(output_folder+'/y_sj.csv')

#get the datasets for the city
#np_sj,np_sj_h,y_sj=prep_for_model(city='sj',lookback=50)

#split the training set into train and test
x_train, x_test, y_train, y_test = train_test_split(df_sj, df_sj_y, test_size=0.30, random_state=0)

#create the model
rfr=RandomForestRegressor(n_estimators=300,max_depth=10)
rfr.fit(x_train,y_train)

#score the model
score=rfr.score(x_test,y_test)
print('SJ score: ',score)
run.log('SJ score: ',np.float(score))

#calculate MAE
y_hat=rfr.predict(x_test)
mae=mean_absolute_error(y_hat,y_test)
print('SJ MAE: ',mae)
run.log('SJ MAE: ',np.float(mae))

# Save the trained model
os.makedirs(output_folder, exist_ok=True)
output_path = output_folder + "/sj_rfr_model.pkl"
joblib.dump(value=model, filename=output_path)

run.complete()
