import streamlit as st
import pickle
import pandas as pd




with open("linear_regression_traffic.pkl", "rb") as file:
    model = pickle.load(file)

feature_columns = ['Time', 'Junction_2', 'Junction_3', 'Junction_4', 'Day', 'Month',
       'Year']


st.title("🚦 Traffic Congestion Prediction App")
st.write("Predict number of vehicles at a junction based on time and date inputs.")


st.header("Enter Traffic Details")

hour = st.slider("Hour of the Day", 0, 23, 8)
day = st.slider("Day", 1, 31, 1)
month = st.slider("Month", 1, 12, 1)
year = st.number_input("Year", value=2015)


junction = st.selectbox("Junction", [1, 2, 3, 4])

input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

if "hour" in input_data.columns:
    input_data["hour"] = hour
if "day" in input_data.columns:
    input_data["day"] = day
if "month" in input_data.columns:
    input_data["month"] = month
if "year" in input_data.columns:
    input_data["year"] = year


junction_col = f"junction_{junction}"
if junction_col in input_data.columns:
    input_data[junction_col] = 1

if st.button("Predict Traffic"):
    prediction = model.predict(input_data)[0]

    st.subheader("✅ Prediction Result")
    st.write(f"**Estimated Number of Vehicles:** {int(prediction)}")


    if prediction < 30:
        st.success("Congestion Level: LOW")
    elif prediction < 70:
        st.warning("Congestion Level: MEDIUM")
    else:
        st.error("Congestion Level: HIGH")
