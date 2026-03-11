# 1. Load scaler.pkl and model.pkl
# 2. Get the inputs from the user
# 3. Scale the inputs
# 4. Predict the output
# 5. Return the output

import pickle
import numpy as np

class Insurance_Prediction:

    def __init__(self):

        with open("artifacts/scaler.pkl", "rb") as f:
            self.scaler = pickle.load(f)

        with open("artifacts/model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def prediction(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):

        input_data = np.array([[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]])

        scaled_input = self.scaler.transform(input_data)

        result = self.model.predict(scaled_input)[0]

        # Prevent negative insurance cost
        result = max(result, 0)

        return result