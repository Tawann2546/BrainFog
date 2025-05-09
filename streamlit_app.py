import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ---------- โหลดข้อมูล ----------
@st.cache_data
def load_data():
    df = pd.read_csv("Video_Game_Sales_as_at_22_Dec_2016.csv")
    return df

df = load_data()

# ---------- UI ----------
st.title("🎮 Video Game Sales Prediction")
years_to_predict = st.slider("📅 เลือกจำนวนปีล่วงหน้า", 1, 5, 3)

# ---------- เตรียมข้อมูล ----------
df_pub = df[['Year_of_Release', 'Publisher', 'Global_Sales']].dropna()
df_pub = df_pub[df_pub['Year_of_Release'].between(2010, 2016)]
df_pub['Year_of_Release'] = df_pub['Year_of_Release'].astype(int)

pub_sales = df_pub.groupby(['Year_of_Release', 'Publisher'])['Global_Sales'].sum().reset_index()

# ---------- Encode Publisher ----------
le = LabelEncoder()
pub_sales['Publisher_encoded'] = le.fit_transform(pub_sales['Publisher'])

X = pub_sales[['Year_of_Release', 'Publisher_encoded']]
y = pub_sales['Global_Sales']

# ---------- Train Model ----------
model = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X, y)

# ---------- Predict ----------
future_years = list(range(2017, 2017 + years_to_predict))
unique_pubs = pub_sales['Publisher_encoded'].unique()

future_df = pd.DataFrame({
    'Year_of_Release': np.repeat(future_years, len(unique_pubs)),
    'Publisher_encoded': np.tile(unique_pubs, len(future_years))
})
future_df['Predicted_Sales'] = model.predict(future_df)
future_df['Publisher'] = le.inverse_transform(future_df['Publisher_encoded'])

summary = future_df.groupby('Publisher')['Predicted_Sales'].sum().reset_index()
summary = summary.sort_values(by='Predicted_Sales', ascending=False)

# ---------- แสดงผล ----------
st.subheader(f"📊 ยอดขายรวมที่คาดว่าจะเกิดขึ้นในปี 2017–{2016 + years_to_predict}")
st.dataframe(summary.head(10))
