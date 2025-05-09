# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# === 1. Load and clean data ===
import os

@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "Video_Games_Sales_as_at_22_Dec_2016.csv")
    return pd.read_csv(csv_path)

df = load_data()
df = df.dropna(subset=['Year_of_Release', 'Genre', 'Global_Sales', 'Publisher', 'Platform'])
df['Year_of_Release'] = df['Year_of_Release'].astype(int)

region_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
df[region_cols] = df[region_cols].fillna(0)

# === 2. Prepare genre_year for training ===
genre_year = df.groupby(['Year_of_Release', 'Genre'])['Global_Sales'].sum().reset_index()

# Encode genre
genre_le = LabelEncoder()
genre_year['Genre_encoded'] = genre_le.fit_transform(genre_year['Genre'])

# Train model to predict global sales by year and genre
X = genre_year[['Year_of_Release', 'Genre_encoded']]
y = genre_year['Global_Sales']

model = LinearRegression()
model.fit(X, y)

# Predict future years 2025–2030
future_years = pd.DataFrame({
    'Year_of_Release': np.repeat(np.arange(2025, 2031), len(genre_le.classes_)),
    'Genre_encoded': np.tile(np.arange(len(genre_le.classes_)), 6)
})
future_years['Genre'] = genre_le.inverse_transform(future_years['Genre_encoded'])
future_years['Predicted_Sales'] = model.predict(future_years[['Year_of_Release', 'Genre_encoded']])

# === 3. UI for Genre Trend Prediction ===
st.title("🎮 คาดการณ์แนวเกมที่จะมาแรงในอนาคต")

# Combine historical + predicted data
genre_all_years = pd.concat([
    genre_year[['Year_of_Release', 'Genre', 'Global_Sales']],
    future_years.rename(columns={'Predicted_Sales': 'Global_Sales'})[['Year_of_Release', 'Genre', 'Global_Sales']]
])

# Calculate growth trend (slope) for each genre
growth_results = []
for genre in genre_all_years['Genre'].unique():
    subset = genre_all_years[genre_all_years['Genre'] == genre]
    X = subset[['Year_of_Release']]
    y = subset['Global_Sales']
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    growth_results.append((genre, slope))

trend_df = pd.DataFrame(growth_results, columns=['Genre', 'Growth_Trend'])
trend_df = trend_df.sort_values(by='Growth_Trend', ascending=False)

st.subheader("📈 แนวเกมที่มีแนวโน้มเพิ่มขึ้น (2025–2030 + อดีต)")
st.dataframe(trend_df, use_container_width=True)

# Optional chart
top_n = st.slider("📊 เลือกจำนวนแนวเกมที่จะแสดงในกราฟ", min_value=1, max_value=10, value=5)
st.bar_chart(trend_df.head(top_n).set_index('Genre'))