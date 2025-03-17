import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import joblib
import numpy as np

# Load the ML model, data processor, and label encoder
model = joblib.load('yield_prediction_model.pkl')
processor = joblib.load('data_processor.pkl')
le = joblib.load('label_encoder.pkl')

# Define crop options
crops = [
    'Potatoes', 'Maize', 'Wheat', 'Rice, paddy', 'Soybeans', 'Sorghum',
    'Sweet potatoes', 'Cassava', 'Yams', 'Plantains and others'
]

# Define Prediction function
def Prediction(average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Item, threshold=0.7):
    try:
        if average_rain_fall_mm_per_year == 0 and pesticides_tonnes == 0 and avg_temp == 0:
            st.warning("Please provide valid input values. All inputs cannot be zero or empty.")
            return None, None
        if pesticides_tonnes == 0:
            st.warning("Please provide valid input values. Pesticides cannot be zero or empty.")
            return None, None
        if avg_temp <= 10 or avg_temp >= 50:
            st.warning("Please provide valid input values. Average temperature should be greater than 5°C. since we are dealing with crops.")
            return None, None
        if soil_ph < 0 or soil_ph > 14:
            st.warning("Please provide valid input values. Soil pH should be between 0 and 14.")
            return None, None
        
        features = np.array([[average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Item]], dtype=object)
        features[:, 3] = le.transform(features[:, 3])
        transformed_features = processor.transform(features)
        predicted_yield = model.predict(transformed_features)[0]
        yield_status = "High" if predicted_yield >= threshold else "Low"
        return predicted_yield, yield_status
    except Exception as e:
        st.error(f"Error in Prediction: {e}")
        return None, None

# Load environment variables
load_dotenv()
groq_key = os.getenv('GROQ_API_KEY')

# Initialize the ChatGroq model
chat_model = ChatGroq(model='Gemma2-9b-It', groq_api_key=groq_key)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages([
    ('system',
     "You are an agricultural assistant, a virtual agricultural officer trained to assist farmers. Your primary role is to take the inputs (avg_temperature:{avg_temp} in celsius, avg_rainfall:{avg_rf} in mm, soil_ph:{soil_ph}, Nitrogen:{N}, Phosphorus:{P}, Potassium:{K} values in soil, for crop:{Item}) "
     "and the output from the crop yield prediction model is {res}. Also, consider the yield status:{statu}.\n\n"
     "You are responsible for verifying the prediction's accuracy, answering user queries, providing actionable recommendations for improving soil composition, and offering advice on crop monitoring. "
     "Always ensure that your responses are accurate, helpful, and easy to understand.\n\n"
     "Instructions:\n"
     "1. Verify the predicted crop yield based on user-provided features and check for any inconsistencies or anomalies.\n"
     "2. Provide actionable recommendations if NPK levels are low, suggesting appropriate fertilizers or organic solutions.\n"
     "3. Offer crop monitoring tips, such as pest management, irrigation schedules, or growth tracking, based on the provided data.\n"
     "4. Maintain a professional, empathetic, and approachable tone in all interactions.\n"
     "5. If Yield status is low, suggest necessary steps to improve the yield status.\n"
     "Please note that the model's predictions are based on the input features provided, and the accuracy may vary based on the data quality and model performance."),
    ('user', "Question:{question}")
])

# Function for generating chatbot responses (corrected)
def generate_response(avg_rf, pesticides, avg_temp, N, P, K, soil_ph, result, status, Item, question):
    try:
        prompt_value = prompt.format_prompt(
            avg_rf=avg_rf, avg_temp=avg_temp, soil_ph=soil_ph, N=N, P=P, K=K,
            Item=Item, res=result, statu=status, question=question
        )
        response = chat_model.invoke(prompt_value)
        return response.content
    except Exception as e:
        st.error(f"Error in Chatbot Response: {e}")
        return None

# --- Streamlit UI ---

st.title('Shasya Mitra: AI for Crop Yield Prediction')
st.write('Provide crop data below, and I will predict yield and answer your agricultural queries.')

# --- Sidebar ---
st.sidebar.title("Shasya Mitra")

# --- Prediction Area in Sidebar ---
st.sidebar.header("Crop Yield Prediction")
st.sidebar.write("Enter the following details to predict crop yield:")
average_rf = st.sidebar.number_input('Average Rainfall (mm):', min_value=0.0, step=0.1)
pesticides_tonnes = st.sidebar.number_input('Pesticides (tonnes):', min_value=0.0, step=0.1)
avg_temp = st.sidebar.number_input('Average Temperature (°C):', min_value=0.0, step=0.1)
Item = st.sidebar.selectbox('Crop:', crops)

levels = ['Low', 'Medium', 'High']


N = st.sidebar.radio('Nitrogen Level', levels)
P = st.sidebar.radio('Phosphorus Level', levels)
K = st.sidebar.radio('Potassium Level', levels)

soil_ph = st.sidebar.number_input('Soil pH:', min_value=0.0, step=0.1)

if st.sidebar.button("Predict Yield"):
    result, status = Prediction(average_rf, pesticides_tonnes, avg_temp, Item)
    if result is not None:
        st.sidebar.success(f"Predicted Yield: {result}hg/ha (Status: {status})")
    st.session_state.result1, st.session_state.status1 = result, status

# --- Chatbot Area ---
st.header("Chat with Shasya Mitra")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.markdown(f"**{message['type'].capitalize()}:** {message['text']}")

user_input = st.text_input("You:", key="user_input")

if st.button("Send"):
    if user_input:
        st.session_state.messages.append({"type": "user", "text": user_input})
        if hasattr(st.session_state, 'result1') and hasattr(st.session_state, 'status1'):
            response = generate_response(average_rf, pesticides_tonnes, avg_temp, N, P, K, soil_ph, st.session_state.result1, st.session_state.status1, Item, user_input)
        else:
            response = "Please predict the yield first in the sidebar."
        if response:
            st.session_state.messages.append({"type": "bot", "text": response})
        else:
            st.session_state.messages.append({"type": "bot", "text": "Sorry, I didn’t understand."})

        st.text_input("You:", value="", key="reset_user_input")
        st.rerun()