
# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import os

# === 1. Load and clean data ===
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "Video_Games_Sales_as_at_22_Dec_2016.csv")
    return pd.read_csv(csv_path)

df = load_data()
df = df.dropna(subset=['Year_of_Release', 'Genre', 'Global_Sales', 'Publisher', 'Platform'])
df['Year_of_Release'] = df['Year_of_Release'].astype(int)

region_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
df[region_cols] = df[region_cols].fillna(0)

# === 2. คาดการณ์แนวเกมที่จะเพิ่มขึ้นในอนาคต โดยใช้ slope จากอดีต ===
st.title("🎮 คาดการณ์แนวเกมที่จะมาแรงในอนาคต")

# สร้างข้อมูลยอดขายรวมต่อแนวเกมในแต่ละปี
genre_year = df.groupby(['Year_of_Release', 'Genre'])['Global_Sales'].sum().reset_index()

# คำนวณ slope (แนวโน้ม) ของแต่ละแนวเกมจากข้อมูลในอดีต
growth_results = []
for genre in genre_year['Genre'].unique():
    subset = genre_year[genre_year['Genre'] == genre]
    X = subset[['Year_of_Release']]
    y = subset['Global_Sales']
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    growth_results.append((genre, slope))

# สร้าง DataFrame แสดงแนวโน้ม
trend_df = pd.DataFrame(growth_results, columns=['Genre', 'Growth_Trend'])
trend_df = trend_df.sort_values(by='Growth_Trend', ascending=False)

st.subheader("📈 แนวเกมที่มีแนวโน้มเพิ่มขึ้น (จากข้อมูลอดีต)")
st.dataframe(trend_df, use_container_width=True)

# กราฟแนวโน้มแนวเกม
top_n = st.slider("📊 เลือกจำนวนแนวเกมที่จะแสดงในกราฟ", min_value=1, max_value=10, value=5)
st.bar_chart(trend_df.head(top_n).set_index('Genre'))
