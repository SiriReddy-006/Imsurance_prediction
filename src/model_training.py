# 1. Load processed data from processed folder
# 2. create model and train data
# 3. 

# 1. Load processed data
# 2. Train model
# 3. Save trained model

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

x_train = pd.read_csv("../data/processed/x_train.csv")
y_train = pd.read_csv("../data/processed/y_train.csv").values.ravel()

x_test = pd.read_csv("../data/processed/x_test.csv")
y_test = pd.read_csv("../data/processed/y_test.csv")

model = LinearRegression()

model.fit(x_train, y_train)

with open("../artifacts/model.pkl", "wb") as f:
    pickle.dump(model, f)