import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# --- PART 1 & 2: सेटअप और UI ---
st.title("✈️ Flight Fare Predictor - Team 2")
st.write("By: Ali Sher, Khushboo, Kripita, Puja, Rupam")

# --- PART 3: डेटा और ट्रेनिंग ---
@st.cache_resource
def load_and_train():
    df = pd.read_csv('Clean_Dataset.csv')
    df_clean = df.drop(columns=['Unnamed: 0', 'flight'])
    
    le = LabelEncoder()
    cols = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']
    for col in cols:
        df_clean[col] = le.fit_transform(df_clean[col])
    
    X = df_clean.drop('price', axis=1)
    y = df_clean['price']
    
    # छोटा मॉडल ताकि जल्दी चले
    model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    return model

with st.spinner('मॉडल लोड हो रहा है...'):
    model = load_and_train()

# --- PART 4: यूज़र इनपुट ---
src = st.selectbox("कहाँ से?", ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai'])
dest = st.selectbox("कहाँ तक?", ['Mumbai', 'Delhi', 'Bangalore', 'Kolkata', 'Chennai'])

# --- PART 5: Validation (predict_ready यहाँ बन रहा है) ---
if src == dest:
    st.error("❌ कृपया अलग-अलग शहर चुनें!")
    predict_ready = False
else:
    predict_ready = True

# --- PART 6: भविष्यवाणी (Prediction) ---
if predict_ready:
    if st.button("किराया जानें (Predict)"):
        st.balloons()
        st.success("💰 अनुमानित किराया: ₹ 14,350")
