#load training data and testing data
# scale the training data
# save scaled data into processed folder
from data_preprocessing import load_and_Split_data
import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle
x_train,x_test, y_train, y_test=load_and_Split_data()

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
os.makedirs("../data/processed", exist_ok=True)
os.makedirs("../artifacts", exist_ok=True)
with open("../artifacts/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
pd.DataFrame(x_train_scaled, columns=x_train.columns).to_csv("../data/processed/x_train.csv", index=False)
pd.DataFrame(x_test_scaled, columns=x_test.columns).to_csv("../data/processed/x_test.csv", index=False)
pd.DataFrame(y_train).to_csv("../data/processed/y_train.csv", index=False)
pd.DataFrame(y_test).to_csv("../data/processed/y_test.csv", index=False)