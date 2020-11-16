#Import libraries
from azureml.core import Run
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

#Set run context
run=Run.get_context()

#Get PipelineData argument
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, dest='folder')
parser.add_argument('--model_folder',type=str,dest='model_folder')
args = parser.parse_args()
data_folder = args.folder
model_folder= args.model_folder

#create a dataframe for each dataset, train and holdout
df_sj=pd.read_csv(data_folder+'/train_sj_scaled.csv')
df_sj_h=pd.read_csv(data_folder+'/holdout_sj_scaled.csv')
df_sj_y=pd.read_csv(data_folder+'/y_sj.csv')

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
os.makedirs(model_folder, exist_ok=True)
output_path = model_folder + "/sj_rfr_model.pkl"
joblib.dump(value=rfr, filename=output_path)

run.complete()
