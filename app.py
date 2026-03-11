import streamlit as st
from src.prediction import Insurance_Prediction
st.title("Insurance prediction")
st.write("This project predicts medical insurance costs using machine learning based on factors such as age, BMI, number of children, smoking status, and region. The model is trained using regression techniques and integrated with a simple Streamlit web interface where users can input their details to estimate their insurance charges.")
Age = st.number_input("Enter age:", min_value=18, max_value=70, value=30)
Annual_Income_LPA = st.number_input("Annual Income (LPA):", min_value=1, max_value=50, value=8)
Policy_Term_Years = st.number_input("Policy Term Years:", min_value=5, max_value=40, value=20)
Sum_Assured_Lakhs = st.number_input("Sum Assured (Lakhs):", min_value=5, max_value=200, value=50)
if st.button("predict"):
    model=Insurance_Prediction()
    result=model.prediction(Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs)
    st.success(f"Predicted Insurance Cost : {result:.2f}")